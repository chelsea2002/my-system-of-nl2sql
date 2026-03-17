<template>
  <div class="nl2sql-page">
    <div class="page-shell">
      <!-- 顶部 -->
      <section class="page-head">
        <div>
          <div class="page-title">Text-to-SQL 查询</div>
          <div class="page-sub">
            输入自然语言问题，生成候选 SQL，选择执行结果并提交反馈，沉淀为历史与样本数据。
          </div>
        </div>

        <div class="page-head-actions">
          <el-button @click="router.push('/dashboard')">返回工作台</el-button>
          <el-button type="primary" plain @click="router.push('/history')">历史管理</el-button>
          <el-button type="primary" plain @click="router.push('/sar')">样本库管理</el-button>
        </div>
      </section>

      <!-- 提问区 -->
      <el-card class="panel-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">1. 提问与生成</div>
              <div class="panel-subtitle">选择数据库并输入自然语言问题</div>
            </div>
            <el-tag effect="plain">POST /nl2sql/generate</el-tag>
          </div>
        </template>

        <div class="ask-grid">
          <div class="field db-field">
            <div class="label">db_id</div>
            <el-select v-model="dbId" class="full-width" placeholder="选择数据库">
              <el-option
                v-for="db in dbList"
                :key="db.db_id"
                :label="db.name || db.db_id"
                :value="db.db_id"
              />
            </el-select>
          </div>

          <div class="field question-field">
            <div class="label">自然语言问题</div>
            <el-input
              v-model="question"
              placeholder="例如：按地区统计活跃用户数（Top10）"
              @keyup.enter="generate"
            />
          </div>

          <div class="field action-field">
            <div class="label">&nbsp;</div>
            <el-button type="primary" :loading="genLoading" @click="generate">
              生成 SQL
            </el-button>
          </div>
        </div>

        <div class="tips-row">
          <el-alert
            type="info"
            show-icon
            :closable="false"
            title="系统会返回候选 SQL 及评分信息，包括 schema 匹配、样例相似性和可执行性。"
          />
        </div>
      </el-card>

      <!-- 中间：左候选，右执行摘要 -->
      <section class="work-grid">
        <!-- 候选 -->
        <el-card class="panel-card" shadow="never">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">2. 候选 SQL</div>
                <div class="panel-subtitle">选择最合适的 SQL 进入执行阶段</div>
              </div>
              <el-tag effect="plain">POST /nl2sql/execute</el-tag>
            </div>
          </template>

          <el-empty v-if="candidates.length === 0" description="暂无候选，请先生成" />

          <div v-else class="candidate-list">
            <div
              v-for="c in candidates"
              :key="c.id"
              class="candidate-card"
              :class="{ active: selectedId === c.id }"
              @click="selectCandidate(c)"
            >
              <div class="candidate-head">
                <div class="candidate-left">
                  <div class="candidate-title-row">
                    <div class="candidate-title">候选 #{{ c.id }}</div>
                    <el-tag size="small" type="primary" effect="light">
                      final={{ (c.scores?.final_score ?? 0).toFixed(2) }}
                    </el-tag>
                  </div>

                  <div class="candidate-tags">
                    <el-tag
                      size="small"
                      :type="c.scores?.executability?.ok ? 'success' : 'danger'"
                      effect="light"
                    >
                      {{ c.scores?.executability?.ok ? '可执行' : '不可执行' }}
                    </el-tag>
                    <el-tag size="small" effect="plain">
                      schema={{ (c.scores?.schema?.score ?? 0).toFixed(2) }}
                    </el-tag>
                    <el-tag size="small" effect="plain">
                      example={{ (c.scores?.examples?.score ?? 0).toFixed(2) }}
                    </el-tag>
                  </div>
                </div>

                <el-radio :model-value="selectedId === c.id" />
              </div>

              <el-input type="textarea" :rows="4" :model-value="c.sql" readonly />

              <div class="candidate-meta">
                <div class="meta-block">
                  <div class="meta-k">schema.used</div>
                  <div class="meta-v">{{ (c.scores?.schema?.used || []).join(', ') || '-' }}</div>
                </div>
                <div class="meta-block">
                  <div class="meta-k">examples.nearest</div>
                  <div class="meta-v">{{ c.scores?.examples?.nearest || '-' }}</div>
                </div>
              </div>
            </div>
          </div>

          <div class="candidate-actions" v-if="selectedSql">
            <el-button @click="copy(selectedSql)">复制 SQL</el-button>
            <el-button type="primary" :loading="execLoading" @click="execute">
              执行 SQL
            </el-button>
          </div>
        </el-card>

        <!-- 执行摘要 -->
        <el-card class="panel-card side-card" shadow="never">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">执行摘要</div>
                <div class="panel-subtitle">当前执行状态与耗时信息</div>
              </div>
            </div>
          </template>

          <div class="summary-list">
            <div class="summary-item">
              <span class="summary-k">已选候选</span>
              <span class="summary-v">{{ selectedId || '-' }}</span>
            </div>
            <div class="summary-item">
              <span class="summary-k">生成耗时</span>
              <span class="summary-v">{{ genTimeMs || 0 }} ms</span>
            </div>
            <div class="summary-item">
              <span class="summary-k">执行耗时</span>
              <span class="summary-v">{{ execTimeMs || 0 }} ms</span>
            </div>
            <div class="summary-item">
              <span class="summary-k">总耗时</span>
              <span class="summary-v">{{ totalTimeMs || 0 }} ms</span>
            </div>
            <div class="summary-item">
              <span class="summary-k">执行状态</span>
              <span class="summary-v">
                <el-tag v-if="execOk" type="success" effect="light">成功</el-tag>
                <el-tag v-else-if="execError" type="danger" effect="light">失败</el-tag>
                <el-tag v-else effect="plain">未执行</el-tag>
              </span>
            </div>
            <div class="summary-item">
              <span class="summary-k">结果行数</span>
              <span class="summary-v">{{ rows.length }}</span>
            </div>
          </div>

          <el-alert
            class="mt16"
            type="info"
            show-icon
            :closable="false"
            title="建议先查看候选评分，再选择最优 SQL 执行。"
          />
        </el-card>
      </section>

      <!-- 执行结果 -->
      <el-card class="panel-card result-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">3. 执行结果</div>
              <div class="panel-subtitle">查看 SQL 执行后的表格结果与可视化建议</div>
            </div>
            <el-tag effect="plain">POST /history/feedback</el-tag>
          </div>
        </template>

        <el-alert
          v-if="execError"
          type="error"
          show-icon
          :closable="false"
          :title="execError"
        />

        <div v-if="execOk" class="result-meta">
          <el-tag effect="light" type="success">执行成功</el-tag>
          <el-tag effect="plain">rows={{ rows.length }}</el-tag>
        </div>

        <div class="result-panel" v-if="execOk">
          <el-table
            v-if="columns.length > 0"
            :data="tableData"
            size="small"
            style="width: 100%"
            stripe
          >
            <el-table-column
              v-for="col in columns"
              :key="col"
              :prop="col"
              :label="col"
              min-width="140"
              show-overflow-tooltip
            />
          </el-table>

          <el-empty v-if="columns.length === 0" description="无表格数据" />
        </div>

        <div class="chart-block">
          <div class="section-title">可视化</div>
          <el-alert
            type="info"
            show-icon
            :closable="false"
            title="系统可根据执行结果自动推荐图表类型与字段映射。"
          />
          <div class="mt16">
            <ChartPanel
              :columns="columns"
              :rows="chartRows"
              :suggests="chartsSuggest"
            />
          </div>
        </div>
      </el-card>

      <!-- 反馈 -->
      <el-card class="panel-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">4. 结果反馈</div>
              <div class="panel-subtitle">提交正确性反馈，支持进入历史和 SAR 样本库</div>
            </div>
          </div>
        </template>

        <div class="feedback-box">
          <div class="feedback-row">
            <div class="feedback-left">
              <div class="label">结果是否正确</div>
              <el-radio-group v-model="feedbackOk">
                <el-radio :label="true">正确</el-radio>
                <el-radio :label="false">不正确</el-radio>
              </el-radio-group>
            </div>

            <el-button type="primary" :loading="fbLoading" @click="submitFeedback">
              提交反馈
            </el-button>
          </div>

          <div v-if="feedbackOk === false" class="mt16">
            <div class="label">修正 SQL（可选）</div>
            <el-input
              type="textarea"
              :rows="6"
              v-model="correctedSql"
              placeholder="如果你有正确 SQL，可粘贴在这里用于纠错与入库"
            />
          </div>

          <el-alert
            class="mt16"
            type="warning"
            show-icon
            :closable="false"
            title="正确反馈可写入 supervised_data，并触发后续 SAR 索引更新。"
          />
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter, useRoute } from 'vue-router'
import { apiDbList } from '@/api/sar'
import { apiNl2sqlGenerate, apiNl2sqlExecute, apiHistoryFeedback } from '@/api/nl2sql'
import ChartPanel from '@/views/ChartPanel.vue'

const router = useRouter()
const route = useRoute()
// ---------- state ----------
const dbList = ref([])
const dbId = ref('')
const question = ref('')

const genLoading = ref(false)
const candidates = ref([])
const selectedId = ref('')
const selectedSql = computed(() => candidates.value.find(x => x.id === selectedId.value)?.sql || '')

const execLoading = ref(false)
const execOk = ref(false)
const execError = ref('')
const execTimeMs = ref(0)

const genTimeMs = ref(0) // ✅ 生成耗时（ms）
const totalTimeMs = computed(() => (genTimeMs.value || 0) + (execTimeMs.value || 0))

const columns = ref([])
const rows = ref([])
const chartsSuggest = ref([])
const chartRows = computed(() => (rows.value || []).slice(0, 200))

const feedbackOk = ref(true)
const correctedSql = ref('')
const fbLoading = ref(false)

const tableData = computed(() => {
  return (rows.value || []).map(r => {
    const obj = {}
    columns.value.forEach((c, i) => (obj[c] = r[i]))
    return obj
  })
})

// ---------- lifecycle ----------
onMounted(async () => {
  dbList.value = await apiDbList()
  if (dbList.value.length) dbId.value = dbList.value[0].db_id

  // ✅ 复用：优先读 localStorage，其次读 query（你也可以只保留一种）
  const reuseFlag = route.query?.reuse
  let payload = null

  const s = localStorage.getItem('nl2sql_reuse_payload')
  if (s) {
    try { payload = JSON.parse(s) } catch {}
    localStorage.removeItem('nl2sql_reuse_payload')
  }

  // 如果你也想支持 url 参数：/nl2sql?db_id=xxx&q=yyy
  const qDb = route.query?.db_id
  const qQ = route.query?.q

  const useDb = payload?.db_id || (typeof qDb === 'string' ? qDb : '')
  const useQ  = payload?.question || (typeof qQ === 'string' ? qQ : '')

  if (reuseFlag || useDb || useQ) {
    if (useDb) dbId.value = useDb
    if (useQ) question.value = useQ
  }
})

// ---------- helpers ----------
function resetExecOnly() {
  // ✅ 只清执行结果，不动 genTimeMs（否则你 generate() 刚算的耗时会被清掉）
  execOk.value = false
  execError.value = ''
  execTimeMs.value = 0
  columns.value = []
  rows.value = []
  chartsSuggest.value = []
  feedbackOk.value = true
  correctedSql.value = ''
}

function resetAll() {
  // 如果你确实想全清（包括 genTimeMs），用这个
  resetExecOnly()
  genTimeMs.value = 0
}

// ---------- actions ----------
async function generate() {
  if (!dbId.value) return ElMessage.error('请选择 db_id')
  if (!question.value.trim()) return ElMessage.error('请输入问题')

  genLoading.value = true
  const t0 = performance.now() // ✅ 前端兜底计时
  try {
    const res = await apiNl2sqlGenerate({ db_id: dbId.value, question: question.value.trim() })
    candidates.value = res?.candidates || []
    selectedId.value = candidates.value[0]?.id || ''

    // ✅ 优先用后端返回的耗时，否则用前端计时
    const backendGen = res?.gen_time_ms ?? res?.gen_ms
    genTimeMs.value = Number.isFinite(backendGen) ? Number(backendGen) : Math.round(performance.now() - t0)

    // 清空上一次执行结果（但不要清 genTimeMs）
    resetExecOnly()

    ElMessage.success(`已生成 ${candidates.value.length} 条候选`)
  } catch (e) {
    ElMessage.error(e?.message || '生成失败')
  } finally {
    genLoading.value = false
  }
}

function selectCandidate(c) {
  selectedId.value = c.id
  resetExecOnly()
}

async function execute() {
  if (!selectedSql.value) return ElMessage.error('请先选择候选 SQL')
  execLoading.value = true
  execError.value = ''

  try {
    const res = await apiNl2sqlExecute({ db_id: dbId.value, sql: selectedSql.value })

    console.log('[apiNl2sqlExecute raw res]', res)
    console.log('[keys]', res && typeof res === 'object' ? Object.keys(res) : res)

    const body = res?.data ?? res?.result ?? res

    execOk.value = !!body?.ok

    const bag = [body, res, res?.data, res?.result, res?.payload, res?.item].filter(Boolean)
    let ms =
      bag.find(x => x.exec_time_ms != null)?.exec_time_ms ??
      bag.find(x => x.exec_ms != null)?.exec_ms ??
      bag.find(x => x.elapsed_ms != null)?.elapsed_ms ??
      bag.find(x => x.latency_ms != null)?.latency_ms ??
      bag.find(x => x.time_ms != null)?.time_ms ??
      0
    execTimeMs.value = Number(ms) || 0

    columns.value = body?.columns || []
    rows.value = body?.rows || []
    chartsSuggest.value = body?.charts_suggest || []
    execError.value = body?.ok ? '' : (body?.error || '执行失败')

    if (body?.ok) ElMessage.success('执行成功')
    else ElMessage.error(execError.value)
  } catch (e) {
    execOk.value = false
    execError.value = e?.message || '执行失败'
    ElMessage.error(execError.value)
  } finally {
    execLoading.value = false
  }
}



async function submitFeedback() {
  if (!dbId.value) return ElMessage.error('缺少 db_id')
  if (!question.value.trim()) return ElMessage.error('缺少 question')
  if (!selectedSql.value) return ElMessage.error('缺少 selected_sql')

  if (feedbackOk.value === false && correctedSql.value.trim() === '') {
    ElMessage.warning('你选择了“不正确”，建议填写修正 SQL（可选）')
  }

  fbLoading.value = true
  try {
    const payload = {
      db_id: dbId.value,
      question: question.value.trim(),
      selected_sql: selectedSql.value,
      ok: feedbackOk.value,
      corrected_sql: correctedSql.value.trim(),

      // ✅ 耗时字段：用于 /history/stats 的 avgLatencyMs
      gen_ms: Number(genTimeMs.value || 0),
      exec_ms: Number(execTimeMs.value || 0),
      total_ms: Number(totalTimeMs.value || 0),
    }

    const res = await apiHistoryFeedback(payload)

    // ✅ 按后端真实返回展示（res 可能是 {ok:true, history_id:...} 或更多）
    const parts = ['反馈已提交']
    if (res?.history_id != null) parts.push(`history_id=${res.history_id}`)
    if (res?.saved_history) parts.push('已写入历史')
    if (res?.saved_supervised) parts.push('已写入 supervised_data')
    if (res?.reindex_triggered) parts.push('已触发 SAR 增量索引')

    ElMessage.success(parts.join('；'))

    // ✅ 可选：正确反馈后跳转历史页
    // if (feedbackOk.value === true) router.push('/history')
  } catch (e) {
    ElMessage.error(e?.message || '反馈提交失败')
  } finally {
    fbLoading.value = false
  }
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text || '')
    ElMessage.success('已复制')
  } catch {
    ElMessage.error('复制失败（浏览器权限限制）')
  }
}
</script>

<style scoped>
.nl2sql-page {
  min-height: 100vh;
  background: #f5f7fb;
  padding: 20px;
  box-sizing: border-box;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei";
}

.page-shell {
  max-width: 1440px;
  margin: 0 auto;
}

.page-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  margin-bottom: 18px;
}

.page-title {
  font-size: 26px;
  font-weight: 800;
  color: #111827;
  line-height: 1.2;
}

.page-sub {
  margin-top: 8px;
  color: #6b7280;
  font-size: 14px;
}

.page-head-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.panel-card {
  border: 1px solid #e8edf5;
  border-radius: 18px;
  background: #fff;
  box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
  margin-bottom: 16px;
}

.panel-card :deep(.el-card__header) {
  padding: 16px 18px 12px;
  border-bottom: 1px solid #eef2f7;
}

.panel-card :deep(.el-card__body) {
  padding: 18px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
}

.panel-title {
  font-size: 17px;
  font-weight: 700;
  color: #111827;
}

.panel-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: #6b7280;
}

.ask-grid {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr) 140px;
  gap: 14px;
  align-items: end;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.label {
  font-size: 12px;
  color: #6b7280;
  font-weight: 600;
}

.full-width {
  width: 100%;
}

.tips-row {
  margin-top: 16px;
}

.work-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) 320px;
  gap: 16px;
  align-items: start;
}

.side-card {
  position: sticky;
  top: 20px;
}

.candidate-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.candidate-card {
  border: 1px solid #e9eef5;
  border-radius: 16px;
  padding: 14px;
  background: #fff;
  cursor: pointer;
  transition: all 0.18s ease;
}

.candidate-card:hover {
  border-color: #cfe0ff;
  box-shadow: 0 8px 22px rgba(37, 99, 235, 0.08);
}

.candidate-card.active {
  border-color: #93c5fd;
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.12);
  background: #f8fbff;
}

.candidate-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 10px;
}

.candidate-left {
  flex: 1;
  min-width: 0;
}

.candidate-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.candidate-title {
  font-size: 15px;
  font-weight: 700;
  color: #111827;
}

.candidate-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 8px;
}

.candidate-meta {
  margin-top: 10px;
  display: grid;
  gap: 8px;
}

.meta-block {
  padding: 10px 12px;
  border-radius: 12px;
  background: #f8fafc;
}

.meta-k {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 4px;
}

.meta-v {
  font-size: 12px;
  color: #111827;
  line-height: 1.6;
  word-break: break-word;
}

.candidate-actions {
  margin-top: 14px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.summary-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  gap: 14px;
  padding: 12px 0;
  border-bottom: 1px dashed #e5e7eb;
}

.summary-item:last-child {
  border-bottom: none;
}

.summary-k {
  color: #6b7280;
  font-size: 13px;
  white-space: nowrap;
}

.summary-v {
  color: #111827;
  font-weight: 600;
  text-align: right;
  word-break: break-word;
}

.result-card {
  margin-top: 0;
}

.result-meta {
  display: flex;
  gap: 10px;
  align-items: center;
  margin: 12px 0 10px;
  flex-wrap: wrap;
}

.result-panel {
  margin-top: 10px;
  border: 1px solid #edf2f7;
  border-radius: 14px;
  padding: 12px;
  background: #fcfdff;
}

.chart-block {
  margin-top: 18px;
}

.section-title {
  margin-bottom: 10px;
  font-size: 14px;
  font-weight: 700;
  color: #111827;
}

.feedback-box {
  border: 1px solid #edf2f7;
  border-radius: 14px;
  padding: 16px;
  background: #fcfdff;
}

.feedback-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  flex-wrap: wrap;
}

.feedback-left {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.mt16 {
  margin-top: 16px;
}

:deep(.el-table) {
  --el-table-header-bg-color: #f8fafc;
  --el-table-row-hover-bg-color: #f6faff;
  border-radius: 12px;
}

:deep(.el-table th.el-table__cell) {
  font-weight: 700;
  color: #374151;
}

:deep(.el-button + .el-button) {
  margin-left: 0;
}

@media (max-width: 1180px) {
  .work-grid {
    grid-template-columns: 1fr;
  }

  .side-card {
    position: static;
  }
}

@media (max-width: 900px) {
  .nl2sql-page {
    padding: 14px;
  }

  .page-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .page-title {
    font-size: 22px;
  }

  .panel-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .ask-grid {
    grid-template-columns: 1fr;
  }

  .summary-item,
  .feedback-row {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
