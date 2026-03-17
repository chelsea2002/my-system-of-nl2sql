<template>
  <div class="sar-page">
    <div class="page-shell">
      <!-- 顶部 -->
      <section class="page-head">
        <div>
          <div class="page-title">SAR 样本库管理</div>
          <div class="page-sub">
            导入 supervised_data（JSON / JSONL），管理样本数据，并触发索引构建与刷新。
          </div>
        </div>

        <div class="page-head-actions">
          <div class="head-chip" v-if="dbId">
            当前数据库：<b>{{ dbId }}</b>
          </div>
          <div class="head-chip">
            索引状态：<b>{{ indexStatusLabel }}</b>
          </div>
          <el-button @click="goDashboard">返回工作台</el-button>
        </div>
      </section>

      <!-- 顶部三块 -->
      <section class="top-grid">
        <!-- 数据库与概览 -->
        <el-card class="panel-card" shadow="never">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">数据库与样本库概览</div>
                <div class="panel-subtitle">选择目标数据库并查看当前样本集状态</div>
              </div>
              <el-tag effect="plain">GET /db/list · GET /sar/datasets/summary</el-tag>
            </div>
          </template>

          <div class="overview-form">
            <div class="field span-2">
              <div class="label">选择数据库（db_id）</div>
              <el-select
                v-model="dbId"
                placeholder="请选择 db_id"
                class="full-width"
                @change="onDbChange"
              >
                <el-option
                  v-for="db in dbList"
                  :key="db.db_id"
                  :label="db.name"
                  :value="db.db_id"
                />
              </el-select>
            </div>

            <div class="field action-field">
              <div class="label">&nbsp;</div>
              <el-button
                :loading="loadingSummary"
                :disabled="!dbId"
                @click="reloadAll"
              >
                刷新数据
              </el-button>
            </div>
          </div>

          <div class="summary-grid">
            <div class="summary-kpi">
              <div class="summary-label">Dataset</div>
              <div class="summary-value small">
                {{ summary?.dataset?.dataset_id || '-' }}
              </div>
            </div>

            <div class="summary-kpi">
              <div class="summary-label">版本</div>
              <div class="summary-value">
                v{{ summary?.dataset?.version ?? '-' }}
              </div>
            </div>

            <div class="summary-kpi">
              <div class="summary-label">样本数</div>
              <div class="summary-value">
                {{ summary?.dataset?.total_samples ?? 0 }}
              </div>
            </div>

            <div class="summary-kpi">
              <div class="summary-label">最近更新</div>
              <div class="summary-value small">
                {{ summary?.dataset?.updated_at || '-' }}
              </div>
            </div>
          </div>
        </el-card>

        <!-- 导入 -->
        <el-card class="panel-card" shadow="never">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">导入 supervised_data</div>
                <div class="panel-subtitle">支持 JSON 数组和 JSONL，每条至少包含 question + sql</div>
              </div>
              <el-tag effect="plain">POST /sar/datasets/import</el-tag>
            </div>
          </template>

          <div class="import-form">
            <div class="field">
              <div class="label">导入模式</div>
              <el-radio-group v-model="importMode">
                <el-radio label="merge">合并导入</el-radio>
                <el-radio label="replace">替换导入</el-radio>
              </el-radio-group>
            </div>

            <div class="field">
              <div class="label">选择文件（.json / .jsonl）</div>
              <el-upload
                :auto-upload="false"
                :show-file-list="true"
                :limit="1"
                :on-change="onFileChange"
                :on-remove="onFileRemove"
                accept=".json,.jsonl"
              >
                <el-button type="primary" plain>选择文件</el-button>
              </el-upload>
            </div>

            <div class="field action-field">
              <div class="label">&nbsp;</div>
              <el-button
                type="primary"
                :disabled="!dbId || !importFile"
                :loading="importing"
                @click="doImport"
              >
                开始导入
              </el-button>
            </div>
          </div>

          <el-alert
            class="mt16"
            type="info"
            show-icon
            :closable="false"
            title="支持格式：JSON 数组或 JSONL。建议导入前确认 question 与 sql 字段齐全。"
          />

          <el-alert
            v-if="importResult"
            class="mt16"
            type="success"
            show-icon
            :closable="false"
            :title="`导入完成：总 ${importResult.imported_total}，有效 ${importResult.imported_valid}，无效 ${importResult.imported_invalid}，新增 ${importResult.added}`"
            :description="`Dataset 当前版本：v${importResult.dataset.version}`"
          />
        </el-card>

        <!-- 索引构建 -->
        <el-card class="panel-card" shadow="never">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">SAR 索引构建</div>
                <div class="panel-subtitle">查看嵌入状态并执行增量或全量构建</div>
              </div>
              <el-tag effect="plain">POST /sar/index/build · GET /sar/index/status</el-tag>
            </div>
          </template>

          <div class="index-status-grid">
            <div class="index-item">
              <div class="index-label">索引状态</div>
              <div class="index-value">
                <el-tag :type="indexTagType" effect="light">{{ indexStatusLabel }}</el-tag>
              </div>
            </div>

            <div class="index-item">
              <div class="index-label">Embedding 数量</div>
              <div class="index-value">{{ indexStatus?.total_embeddings ?? 0 }}</div>
            </div>

            <div class="index-item">
              <div class="index-label">最近构建</div>
              <div class="index-value small">{{ indexStatus?.updated_at || '-' }}</div>
            </div>

            <div class="index-item">
              <div class="index-label">构建模式</div>
              <div class="index-value small">{{ indexStatus?.last_build_mode || '-' }}</div>
            </div>
          </div>

          <div class="index-actions">
            <el-button
              type="primary"
              plain
              :disabled="!dbId"
              :loading="buildingIndex"
              @click="buildIndex('incremental')"
            >
              增量构建
            </el-button>
            <el-button
              type="primary"
              :disabled="!dbId"
              :loading="buildingIndex"
              @click="buildIndex('full')"
            >
              全量重建
            </el-button>
          </div>
        </el-card>
      </section>

      <!-- 样本列表 -->
      <el-card class="panel-card samples-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">样本列表</div>
              <div class="panel-subtitle">按 question / sql 搜索、分页浏览并管理样本</div>
            </div>
            <div class="panel-actions">
              <el-tag effect="plain">GET /sar/samples · DELETE /sar/samples</el-tag>
              <el-button :disabled="!dbId" :loading="loadingSamples" @click="loadSamples">
                刷新
              </el-button>
            </div>
          </div>
        </template>

        <div class="table-toolbar">
          <div class="field search-field">
            <div class="label">搜索（question / sql）</div>
            <el-input
              v-model="keyword"
              placeholder="输入关键词后回车搜索"
              @keyup.enter="onSearch"
              clearable
            />
          </div>

          <div class="field page-size-field">
            <div class="label">每页</div>
            <el-select v-model="pageSize" class="full-width" @change="onPageSizeChange">
              <el-option :value="10" label="10" />
              <el-option :value="20" label="20" />
              <el-option :value="50" label="50" />
            </el-select>
          </div>

          <div class="toolbar-actions">
            <el-button type="primary" plain :disabled="!dbId" @click="onSearch">搜索</el-button>
            <el-button :disabled="!dbId" @click="resetSearch">重置</el-button>
          </div>
        </div>

        <div class="table-shell">
          <el-table
            v-loading="loadingSamples"
            :data="samples"
            size="small"
            style="width: 100%"
            stripe
          >
            <el-table-column prop="created_at" label="时间" width="170" />
            <el-table-column prop="source" label="来源" width="100">
              <template #default="{ row }">
                <el-tag effect="plain">{{ row.source || '-' }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="verified" label="已验证" width="90">
              <template #default="{ row }">
                <el-tag v-if="row.verified" type="success" effect="light">是</el-tag>
                <el-tag v-else type="warning" effect="light">否</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="question" label="Question" min-width="280" show-overflow-tooltip />
            <el-table-column prop="sql" label="SQL" min-width="260" show-overflow-tooltip />

            <el-table-column label="操作" width="210" fixed="right">
              <template #default="{ row }">
                <div class="table-actions">
                  <el-button size="small" @click="openDetail(row)">详情</el-button>
                  <el-button size="small" type="danger" plain @click="delSample(row)">删除</el-button>
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

        <el-empty v-if="!loadingSamples && samples.length === 0" description="暂无样本" />
      </el-card>

      <!-- 详情弹窗 -->
      <el-dialog v-model="detailVisible" title="样本详情" width="760">
        <div class="detail-panel">
          <div class="detail-row"><span class="detail-k">sample_id</span><span class="detail-v">{{ detail?.sample_id }}</span></div>
          <div class="detail-row"><span class="detail-k">db_id</span><span class="detail-v">{{ detail?.db_id }}</span></div>
          <div class="detail-row"><span class="detail-k">source</span><span class="detail-v">{{ detail?.source }}</span></div>
          <div class="detail-row"><span class="detail-k">created_at</span><span class="detail-v">{{ detail?.created_at }}</span></div>
          <div class="detail-row"><span class="detail-k">verified</span><span class="detail-v">{{ detail?.verified ? 'true' : 'false' }}</span></div>

          <el-divider />

          <div class="detail-block-title">question</div>
          <el-input type="textarea" :rows="4" :model-value="detail?.question" readonly />

          <div class="detail-block-title mt16">sql</div>
          <el-input type="textarea" :rows="6" :model-value="detail?.sql" readonly />

          <div class="panel-actions mt16">
            <el-button @click="copy(detail?.question || '')">复制问题</el-button>
            <el-button @click="copy(detail?.sql || '')">复制 SQL</el-button>
          </div>
        </div>

        <template #footer>
          <el-button @click="detailVisible = false">关闭</el-button>
        </template>
      </el-dialog>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  apiDbList,
  apiSarDatasetSummary,
  apiSarSamplesList,
  apiSarImportFile,
  apiSarSampleDelete,
  apiSarBuildIndex,
  apiSarIndexStatus,
} from '@/api/sar'

const router = useRouter()

// DB selection
const dbList = ref([])
const dbId = ref('')

// Summary & index status
const loadingSummary = ref(false)
const summary = ref(null)
const indexStatus = ref(null)

// Import
const importMode = ref('merge') // merge | replace
const importFile = ref(null)
const importing = ref(false)
const importResult = ref(null)

// Samples list
const keyword = ref('')
const page = ref(1)
const pageSize = ref(10)
const total = ref(0)
const samples = ref([])
const loadingSamples = ref(false)

// Detail dialog
const detailVisible = ref(false)
const detail = ref(null)

onMounted(async () => {
  dbList.value = await apiDbList()
  // 默认选第一个 db（你也可以改成不默认）
  if (dbList.value.length) {
    dbId.value = dbList.value[0].db_id
    await reloadAll()
  }
})

function goDashboard() {
  router.push('/dashboard')
}

async function onDbChange() {
  importResult.value = null
  keyword.value = ''
  page.value = 1
  await reloadAll()
}

async function reloadAll() {
  if (!dbId.value) return
  await Promise.all([loadSummary(), loadIndexStatus(), loadSamples()])
}

async function loadSummary() {
  loadingSummary.value = true
  try {
    summary.value = await apiSarDatasetSummary(dbId.value)
  } finally {
    loadingSummary.value = false
  }
}

async function loadIndexStatus() {
  indexStatus.value = await apiSarIndexStatus(dbId.value)
}

const indexStatusLabel = computed(() => {
  const s = indexStatus.value?.status
  if (!s || s === 'NOT_BUILT') return '未构建'
  if (s === 'BUILDING') return '构建中'
  if (s === 'READY') return '就绪'
  if (s === 'FAILED') return '失败'
  return s
})
const indexTagType = computed(() => {
  const s = indexStatus.value?.status
  if (!s || s === 'NOT_BUILT') return 'info'
  if (s === 'BUILDING') return 'warning'
  if (s === 'READY') return 'success'
  if (s === 'FAILED') return 'danger'
  return 'info'
})

function onFileChange(uploadFile) {
  importFile.value = uploadFile.raw
}
function onFileRemove() {
  importFile.value = null
}

async function doImport() {
  if (!dbId.value || !importFile.value) return
  importing.value = true
  importResult.value = null
  try {
    const res = await apiSarImportFile({ db_id: dbId.value, file: importFile.value, mode: importMode.value })
    importResult.value = res
    ElMessage.success('导入完成')
    // 导入后刷新概览/列表/索引状态
    page.value = 1
    await reloadAll()
  } catch (e) {
    ElMessage.error(e?.message || '导入失败')
  } finally {
    importing.value = false
  }
}

const buildingIndex = ref(false)
async function buildIndex(mode) {
  if (!dbId.value) return
  buildingIndex.value = true
  try {
    await apiSarBuildIndex({ db_id: dbId.value, mode })
    ElMessage.success(mode === 'full' ? '已触发全量重建（mock）' : '已触发增量构建（mock）')
    await loadIndexStatus()
  } finally {
    buildingIndex.value = false
  }
}

async function loadSamples() {
  if (!dbId.value) return
  loadingSamples.value = true
  try {
    const res = await apiSarSamplesList({
      db_id: dbId.value,
      q: keyword.value,
      page: page.value,
      page_size: pageSize.value,
    })
    samples.value = res.items
    total.value = res.total
  } finally {
    loadingSamples.value = false
  }
}

function onSearch() {
  page.value = 1
  loadSamples()
}
function resetSearch() {
  keyword.value = ''
  page.value = 1
  loadSamples()
}
function onPageChange(p) {
  page.value = p
  loadSamples()
}
function onPageSizeChange() {
  page.value = 1
  loadSamples()
}

function openDetail(row) {
  detail.value = row
  detailVisible.value = true
}

async function delSample(row) {
  await ElMessageBox.confirm(
    '确定删除该样本吗？（mock 删除，后端接入后可真实删除）',
    '删除确认',
    { type: 'warning' }
  )
  await apiSarSampleDelete({ db_id: dbId.value, sample_id: row.sample_id })
  ElMessage.success('已删除')
  // 删除后刷新
  await reloadAll()
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
.sar-page {
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
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.head-chip {
  height: 34px;
  padding: 0 12px;
  border-radius: 999px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  display: inline-flex;
  align-items: center;
  color: #4b5563;
  font-size: 13px;
}

.top-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-bottom: 16px;
  align-items: start;
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

.panel-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.overview-form {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
  align-items: end;
}

.import-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
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

.action-field {
  justify-content: flex-end;
}

.span-2 {
  grid-column: span 2;
}

.full-width {
  width: 100%;
}

.summary-grid {
  margin-top: 16px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.summary-kpi {
  padding: 14px;
  border: 1px solid #edf2f7;
  border-radius: 14px;
  background: #fafcff;
}

.summary-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 8px;
}

.summary-value {
  font-size: 24px;
  font-weight: 700;
  color: #111827;
  line-height: 1.2;
  word-break: break-word;
}

.summary-value.small {
  font-size: 14px;
  font-weight: 600;
}

.index-status-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.index-item {
  padding: 14px;
  border: 1px solid #edf2f7;
  border-radius: 14px;
  background: #fafcff;
}

.index-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 8px;
}

.index-value {
  font-size: 20px;
  font-weight: 700;
  color: #111827;
  line-height: 1.2;
  word-break: break-word;
}

.index-value.small {
  font-size: 14px;
  font-weight: 600;
}

.index-actions {
  margin-top: 16px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.samples-card {
  margin-bottom: 16px;
}

.table-toolbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 120px auto;
  gap: 14px;
  align-items: end;
  margin-bottom: 14px;
}

.search-field {
  min-width: 0;
}

.page-size-field {
  width: 120px;
}

.toolbar-actions {
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
  width: 100px;
  color: #6b7280;
  flex-shrink: 0;
}

.detail-v {
  color: #111827;
  font-weight: 600;
  word-break: break-word;
}

.detail-block-title {
  font-size: 13px;
  color: #6b7280;
  font-weight: 700;
  margin-bottom: 8px;
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

@media (max-width: 1280px) {
  .top-grid {
    grid-template-columns: 1fr;
  }

  .overview-form {
    grid-template-columns: 1fr 1fr;
  }
}

@media (max-width: 900px) {
  .sar-page {
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

  .overview-form,
  .table-toolbar {
    grid-template-columns: 1fr;
  }

  .span-2 {
    grid-column: span 1;
  }

  .summary-grid,
  .index-status-grid {
    grid-template-columns: 1fr;
  }

  .page-size-field {
    width: 100%;
  }

  .toolbar-actions,
  .index-actions {
    flex-wrap: wrap;
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
