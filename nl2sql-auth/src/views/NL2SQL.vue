<template>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">Text-to-SQL 查询</div>
        <div class="sub">输入自然语言 → 生成候选 SQL → 选择执行 → 反馈进入历史（并可进入 SAR 样本库）。</div>
      </div>
      <div class="actions">
        <el-button @click="router.push('/dashboard')">返回工作台</el-button>
        <el-button type="primary" plain @click="router.push('/history')">历史管理</el-button>
        <el-button type="primary" plain @click="router.push('/sar')">样本库管理</el-button>
      </div>
    </div>

    <!-- 输入区 -->
    <el-card>
      <template #header>
        <div class="card-header">
          <span>提问</span>
          <el-tag effect="plain">接口：POST /nl2sql/generate</el-tag>
        </div>
      </template>

      <div class="row">
        <div class="field" style="width: 260px">
          <div class="label">db_id</div>
          <el-select v-model="dbId" style="width: 260px" placeholder="选择数据库">
            <el-option v-for="db in dbList" :key="db.db_id" :label="db.name" :value="db.db_id" />
          </el-select>
        </div>

        <div class="field" style="flex:1; min-width: 420px">
          <div class="label">自然语言问题</div>
          <el-input v-model="question" placeholder="例如：按地区统计活跃用户数（Top10）" @keyup.enter="generate" />
        </div>

        <div class="field">
          <div class="label">&nbsp;</div>
          <el-button type="primary" :loading="genLoading" @click="generate">生成 SQL</el-button>
        </div>
      </div>

      <el-alert
        style="margin-top: 12px"
        type="info"
        show-icon
        :closable="false"
        title="提示：后端可以在 /nl2sql/generate 返回 candidates[] + 每条候选的可解释评分（schema/examples/executability等）。"
      />
    </el-card>

    <!-- 候选区 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>候选 SQL</span>
          <el-tag effect="plain">接口：POST /nl2sql/execute</el-tag>
        </div>
      </template>

      <el-empty v-if="candidates.length === 0" description="暂无候选，请先生成" />

      <div v-else class="candidates">
        <div
          v-for="c in candidates"
          :key="c.id"
          class="cand"
          :class="{ active: selectedId === c.id }"
          @click="selectCandidate(c)"
        >
          <div class="cand-top">
            <div class="cand-title">
              <b>候选 #{{ c.id }}</b>
              <el-tag size="small" effect="plain" style="margin-left: 8px">
                final={{ (c.scores?.final_score ?? 0).toFixed(2) }}
              </el-tag>
            </div>

            <div class="cand-tags">
              <el-tag size="small" :type="c.scores?.executability?.ok ? 'success' : 'danger'" effect="plain">
                {{ c.scores?.executability?.ok ? '可执行' : '不可执行' }}
              </el-tag>
              <el-tag size="small" effect="plain">schema={{ (c.scores?.schema?.score ?? 0).toFixed(2) }}</el-tag>
              <el-tag size="small" effect="plain">example={{ (c.scores?.examples?.score ?? 0).toFixed(2) }}</el-tag>
            </div>
          </div>

          <el-input type="textarea" :rows="4" :model-value="c.sql" readonly />

          <div class="cand-bottom">
            <div class="mini">
              <div class="k">schema.used</div>
              <div class="v">{{ (c.scores?.schema?.used || []).join(', ') || '-' }}</div>
            </div>
            <div class="mini">
              <div class="k">examples.nearest</div>
              <div class="v">{{ c.scores?.examples?.nearest || '-' }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="cand-actions" v-if="selectedSql">
        <el-button @click="copy(selectedSql)">复制 SQL</el-button>
        <el-button type="primary" :loading="execLoading" @click="execute">执行 SQL</el-button>
      </div>
    </el-card>

    <!-- 结果区 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>执行结果</span>
          <el-tag effect="plain">接口：POST /history/feedback</el-tag>
        </div>
      </template>

      <el-alert v-if="execError" type="error" show-icon :closable="false" :title="execError" />

      <div v-if="execOk" class="result-meta">
        <el-tag effect="plain" type="success">执行成功</el-tag>
        <el-tag effect="plain">rows={{ rows.length }}</el-tag>
      </div>

      <el-table
        v-if="execOk && columns.length > 0"
        :data="tableData"
        size="small"
        style="width: 100%; margin-top: 10px"
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

      <el-empty v-if="execOk && columns.length === 0" description="无表格数据" />

      <!-- 图表占位 -->
      <el-divider />
      <div class="chart">
        <div class="chart-title">可视化</div>
        <el-alert
          type="info"
          show-icon
          :closable="false"
          title="根据执行结果自动给出可视化（ECharts）。后端返回 charts_suggest：bar/line/pie + x/y 字段。"
        />
        <div style="margin-top: 10px">
          <ChartPanel
            :columns="columns"
            :rows="chartRows"
            :suggests="chartsSuggest"
          />
        </div>
      </div>

      <!-- 反馈 -->
      <el-divider />
      <div class="feedback">
        <div class="fb-title">结果反馈</div>
        <div class="fb-row">
          <el-radio-group v-model="feedbackOk">
            <el-radio :label="true">正确</el-radio>
            <el-radio :label="false">不正确</el-radio>
          </el-radio-group>
          <el-button type="primary" :loading="fbLoading" @click="submitFeedback">提交反馈</el-button>
        </div>

        <div v-if="feedbackOk === false" style="margin-top: 10px">
          <div class="label">修正 SQL（可选）</div>
          <el-input type="textarea" :rows="6" v-model="correctedSql" placeholder="如果你有正确 SQL，可粘贴在这里用于纠错与入库" />
        </div>

        <el-alert
          style="margin-top: 10px"
          type="warning"
          show-icon
          :closable="false"
          title="已实现：选择“正确”后，后端会写入 supervised_data.json 并触发 SAR 增量索引（如 dirty 标记/立即重建视后端实现）。"
        />
      </div>
    </el-card>
  </div>
</template>bu

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { apiDbList } from '@/api/sar'
import { apiNl2sqlGenerate, apiNl2sqlExecute, apiHistoryFeedback } from '@/api/nl2sql'

const router = useRouter()

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
.wrap {
  padding: 20px;
  background: #f5f7fb;
  min-height: 100vh;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei";
}
.top {
  display:flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}
.title { font-size: 22px; font-weight: 900; color:#111827; }
.sub { margin-top: 6px; color:#6b7280; }
.actions { display:flex; gap:10px; }

.card-header {
  display:flex;
  align-items:center;
  justify-content: space-between;
  gap: 12px;
}

.row {
  display:flex;
  gap:14px;
  flex-wrap: wrap;
  align-items: end;
}
.field { display:flex; flex-direction: column; gap:6px; }
.label { font-size: 12px; color:#6b7280; }

.candidates {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
.cand {
  border: 1px solid #eef2f7;
  border-radius: 14px;
  padding: 12px;
  background: #fff;
  cursor: pointer;
  transition: all .15s ease;
}
.cand:hover { box-shadow: 0 8px 18px rgba(17,24,39,0.06); transform: translateY(-1px); }
.cand.active { border-color: #93c5fd; box-shadow: 0 8px 18px rgba(37,99,235,0.10); }

.cand-top {
  display:flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  gap: 10px;
}
.cand-tags { display:flex; gap:8px; flex-wrap: wrap; }
.cand-bottom { margin-top: 8px; display:flex; flex-direction: column; gap:6px; }
.mini .k { font-size: 12px; color:#6b7280; }
.mini .v { font-size: 12px; color:#111827; }

.cand-actions { margin-top: 12px; display:flex; gap: 10px; }

.result-meta { display:flex; gap:10px; align-items:center; margin-bottom: 6px; }

.chart-title, .fb-title { font-weight: 900; color:#111827; margin-bottom: 8px; }
.fb-row { display:flex; align-items:center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
</style>
