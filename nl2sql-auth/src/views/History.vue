<template>
  <div class="history-page">
    <div class="page-shell">
      <!-- 顶部 -->
      <section class="page-head">
        <div>
          <div class="page-title">历史管理</div>
          <div class="page-sub">
            查看查询历史、SQL 日志，并将确认正确的样本加入 SAR 样本库。
          </div>
        </div>

        <div class="page-head-actions">
          <el-button @click="goDashboard">返回工作台</el-button>
          <el-button type="primary" plain @click="goSar">去样本库管理</el-button>
        </div>
      </section>

      <!-- 主列表 -->
      <el-card class="panel-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">历史查询列表</div>
              <div class="panel-subtitle">支持按数据库、问题和 SQL 进行筛选</div>
            </div>
            <el-tag effect="plain">GET /history</el-tag>
          </div>
        </template>

        <!-- 筛选区 -->
        <div class="filter-bar">
          <div class="field db-field">
            <div class="label">db_id</div>
            <el-select
              v-model="dbId"
              placeholder="全部数据库"
              class="full-width"
              @change="onSearch"
            >
              <el-option label="全部" value="" />
              <el-option
                v-for="db in dbList"
                :key="db.db_id"
                :label="db.name || db.db_id"
                :value="db.db_id"
              />
            </el-select>
          </div>

          <div class="field keyword-field">
            <div class="label">搜索（question / sql）</div>
            <el-input
              v-model="keyword"
              placeholder="输入关键词后回车搜索"
              @keyup.enter="onSearch"
              clearable
            />
          </div>

          <div class="field size-field">
            <div class="label">每页</div>
            <el-select v-model="pageSize" class="full-width" @change="onPageSizeChange">
              <el-option :value="10" label="10" />
              <el-option :value="20" label="20" />
              <el-option :value="50" label="50" />
            </el-select>
          </div>

          <div class="filter-actions">
            <el-button type="primary" plain @click="onSearch">搜索</el-button>
            <el-button @click="reset">重置</el-button>
          </div>
        </div>

        <!-- 表格 -->
        <div class="table-shell">
          <el-table
            v-loading="loading"
            :data="items"
            size="small"
            style="width: 100%"
            stripe
          >
            <el-table-column prop="created_at" label="时间" width="170" />
            <el-table-column prop="db_id" label="db_id" width="130" />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="row.ok ? 'success' : 'danger'" effect="light">
                  {{ row.ok ? '成功' : '失败' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column
              prop="question"
              label="Question"
              min-width="280"
              show-overflow-tooltip
            />
            <el-table-column
              prop="selected_sql"
              label="SQL"
              min-width="280"
              show-overflow-tooltip
            />

            <el-table-column label="操作" width="280" fixed="right">
              <template #default="{ row }">
                <div class="table-actions">
                  <el-button size="small" @click="openDetail(row)">详情</el-button>
                  <el-button size="small" @click="copy(row.selected_sql)">复制 SQL</el-button>
                  <el-button size="small" type="primary" plain @click="openAddSar(row)">
                    加入 SAR
                  </el-button>
                </div>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <div class="pager">
          <el-pagination
            background
            layout="prev, pager, next, total"
            :total="total"
            :page-size="pageSize"
            :current-page="page"
            @current-change="onPageChange"
          />
        </div>

        <el-empty v-if="!loading && items.length === 0" description="暂无历史记录" />
      </el-card>

      <!-- 详情弹窗 -->
      <el-dialog v-model="detailVisible" title="历史详情" width="820">
        <div class="detail-panel">
          <div class="detail-row">
            <span class="detail-k">id</span>
            <span class="detail-v">{{ detail?.id }}</span>
          </div>
          <div class="detail-row">
            <span class="detail-k">db_id</span>
            <span class="detail-v">{{ detail?.db_id }}</span>
          </div>
          <div class="detail-row">
            <span class="detail-k">时间</span>
            <span class="detail-v">{{ detail?.created_at }}</span>
          </div>
          <div class="detail-row">
            <span class="detail-k">状态</span>
            <span class="detail-v">
              <el-tag :type="detail?.ok ? 'success' : 'danger'" effect="light">
                {{ detail?.ok ? '成功' : '失败' }}
              </el-tag>
              <span class="detail-extra">rows={{ detail?.rows ?? '-' }}</span>
            </span>
          </div>

          <el-divider />

          <div class="detail-block-title">question</div>
          <el-input type="textarea" :rows="4" :model-value="detail?.question" readonly />

          <div class="detail-block-title mt16">sql</div>
          <el-input type="textarea" :rows="7" :model-value="detail?.selected_sql" readonly />

          <div class="detail-block-title mt16">备注</div>
          <el-input type="textarea" :rows="3" :model-value="detail?.note || ''" readonly />

          <div class="dialog-actions mt16">
            <el-button @click="copy(detail?.question || '')">复制问题</el-button>
            <el-button @click="copy(detail?.selected_sql || '')">复制 SQL</el-button>
          </div>
        </div>

        <template #footer>
          <el-button @click="detailVisible = false">关闭</el-button>
        </template>
      </el-dialog>

      <!-- 加入 SAR 弹窗 -->
      <el-dialog v-model="addSarVisible" title="加入 SAR 样本库" width="860">
        <el-alert
          type="info"
          show-icon
          :closable="false"
          title="建议只把确认正确的问答加入 SAR。你可以在这里微调 question 和 sql 后再提交。"
        />

        <div class="sar-form">
          <div class="sar-top-grid">
            <div class="field">
              <div class="label">db_id</div>
              <el-input v-model="sarForm.db_id" readonly />
            </div>
            <div class="field">
              <div class="label">source</div>
              <el-input v-model="sarForm.source" readonly />
            </div>
            <div class="field">
              <div class="label">verified</div>
              <div class="switch-wrap">
                <el-switch v-model="sarForm.verified" />
              </div>
            </div>
          </div>

          <div class="field mt16">
            <div class="label">question（可编辑）</div>
            <el-input v-model="sarForm.question" />
          </div>

          <div class="field mt16">
            <div class="label">sql（可编辑）</div>
            <el-input type="textarea" :rows="8" v-model="sarForm.sql" />
          </div>

          <el-alert
            class="mt16"
            type="warning"
            show-icon
            :closable="false"
            title="schema 快照建议由后端根据 db_id 自动补齐；当前页面先保留占位。"
          />
        </div>

        <template #footer>
          <el-button @click="addSarVisible = false">取消</el-button>
          <el-button type="primary" :loading="addingSar" @click="submitAddSar">
            提交
          </el-button>
        </template>
      </el-dialog>
    </div>
  </div>
</template> 

<script setup>
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { apiDbList } from '@/api/sar'
import { apiHistoryList, apiHistoryDetail } from '@/api/history'
import { apiSarSampleAdd } from '@/api/sar'

const router = useRouter()

const dbList = ref([])
const dbId = ref('')
const keyword = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)
const items = ref([])
const loading = ref(false)

const detailVisible = ref(false)
const detail = ref(null)

const addSarVisible = ref(false)
const addingSar = ref(false)
const sarForm = ref({
  db_id: '',
  question: '',
  sql: '',
  source: 'history',
  verified: true,
  schema: null,
})

onMounted(async () => {
  dbList.value = await apiDbList()
  await load()
})

function goDashboard() {
  router.push('/dashboard')
}
function goSar() {
  router.push('/sar')
}

/** 兼容后端字段：selected_sql / sql / query */
function getRowSql(row) {
  return (row?.selected_sql || row?.sql || row?.query || '').toString()
}
function getRowQuestion(row) {
  return (row?.question || '').toString()
}

/** 时间戳显示友好一点：秒/毫秒都兼容 */
function formatTs(ts) {
  if (!ts) return '-'
  const n = Number(ts)
  if (!Number.isFinite(n)) return String(ts)
  const ms = n < 1e12 ? n * 1000 : n
  const d = new Date(ms)
  const pad = (x) => String(x).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

async function load() {
  loading.value = true
  try {
    const res = await apiHistoryList({
      db_id: dbId.value,
      q: keyword.value,
      page: page.value,
      page_size: pageSize.value,
    })

    // ✅ 统一补一个 selected_sql，保证模板 prop 能显示
    const list = (res?.items || []).map((r) => ({
      ...r,
      selected_sql: getRowSql(r),
      created_at: formatTs(r.created_at),
    }))

    items.value = list
    total.value = res?.total ?? 0
  } catch (e) {
    ElMessage.error(e?.message || '加载历史失败')
  } finally {
    loading.value = false
  }
}

function onSearch() {
  page.value = 1
  load()
}
function reset() {
  dbId.value = ''
  keyword.value = ''
  page.value = 1
  load()
}
function onPageChange(p) {
  page.value = p
  load()
}
function onPageSizeChange() {
  page.value = 1
  load()
}

async function openDetail(row) {
  try {
    const res = await apiHistoryDetail(row.id)
    const obj = res?.item || res || row
    // ✅ 对齐字段
    detail.value = {
      ...obj,
      selected_sql: getRowSql(obj),
      created_at: formatTs(obj.created_at),
    }
    detailVisible.value = true
  } catch (e) {
    ElMessage.error(e?.message || '获取详情失败')
  }
}

function openAddSar(row) {
  sarForm.value = {
    db_id: row?.db_id || '',
    question: getRowQuestion(row),
    sql: getRowSql(row),
    source: 'history',
    verified: true,
    schema: null,
  }
  addSarVisible.value = true
}

async function submitAddSar() {
  const _db = (sarForm.value.db_id || '').trim()
  const _q = (sarForm.value.question || '').trim()
  const _sql = (sarForm.value.sql || '').trim()

  if (!_db || !_q || !_sql) {
    ElMessage.error('db_id / question / sql 不能为空')
    return
  }

  addingSar.value = true
  try {
    // 你目前仍然走 apiSarSampleAdd；后面你要改成 /history/promote 也行
    await apiSarSampleAdd({ ...sarForm.value, db_id: _db, question: _q, sql: _sql })
    ElMessage.success('已加入 SAR 样本库')
    addSarVisible.value = false
    // 你想的话可以 reload，让列表刷新 imported_to_sar
    // await load()
  } catch (e) {
    ElMessage.error(e?.message || '加入失败')
  } finally {
    addingSar.value = false
  }
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText((text || '').toString())
    ElMessage.success('已复制')
  } catch {
    ElMessage.error('复制失败（浏览器权限限制）')
  }
}
</script>


<style scoped>
.history-page {
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

.filter-bar {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr) 120px auto;
  gap: 14px;
  align-items: end;
  margin-bottom: 14px;
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

.filter-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.table-shell {
  border-radius: 12px;
  overflow: hidden;
}

.table-actions {
  display: flex;
  gap: 8px;
}

.pager {
  margin-top: 14px;
  display: flex;
  justify-content: flex-end;
}

.detail-panel {
  border: 1px solid #edf2f7;
  border-radius: 14px;
  padding: 16px;
  background: #fafcff;
}

.detail-row {
  display: flex;
  gap: 12px;
  margin: 8px 0;
}

.detail-k {
  width: 90px;
  color: #6b7280;
  flex-shrink: 0;
}

.detail-v {
  color: #111827;
  font-weight: 600;
  word-break: break-word;
}

.detail-extra {
  margin-left: 10px;
  color: #6b7280;
  font-weight: 400;
}

.detail-block-title {
  font-size: 13px;
  color: #6b7280;
  font-weight: 700;
  margin-bottom: 8px;
}

.dialog-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.sar-form {
  margin-top: 12px;
}

.sar-top-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}

.switch-wrap {
  height: 32px;
  display: flex;
  align-items: center;
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

@media (max-width: 980px) {
  .filter-bar {
    grid-template-columns: 1fr;
  }

  .sar-top-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .history-page {
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

  .detail-row {
    flex-direction: column;
    gap: 4px;
  }

  .detail-k {
    width: auto;
  }
}
</style>
