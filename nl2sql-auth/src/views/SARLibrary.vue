<template>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">SAR 样本库管理</div>
        <div class="sub">
          导入 supervised_data（JSON/JSONL），管理样本，并触发索引构建。
        </div>
      </div>
      <div class="top-actions">
        <el-button @click="goDashboard">返回工作台</el-button>
      </div>
    </div>

    <!-- 选择数据库 + 概览 -->
    <el-card>
      <template #header>
        <div class="card-header">
          <span>数据库与样本库概览</span>
          <el-tag effect="plain">GET /db/list，GET /sar/datasets/summary</el-tag>
        </div>
      </template>

      <div class="row">
        <div class="field">
          <div class="label">选择数据库（db_id）</div>
          <el-select v-model="dbId" placeholder="请选择 db_id" style="width: 260px" @change="onDbChange">
            <el-option v-for="db in dbList" :key="db.db_id" :label="db.name" :value="db.db_id" />
          </el-select>
        </div>

        <div class="field">
          <div class="label">Dataset</div>
          <el-tag effect="plain">
            {{ summary?.dataset?.dataset_id || '-' }} / v{{ summary?.dataset?.version ?? '-' }}
          </el-tag>
        </div>

        <div class="field">
          <div class="label">样本数</div>
          <el-tag type="success" effect="plain">{{ summary?.dataset?.total_samples ?? 0 }}</el-tag>
        </div>

        <div class="field">
          <div class="label">最近更新</div>
          <el-tag effect="plain">{{ summary?.dataset?.updated_at || '-' }}</el-tag>
        </div>

        <div class="field" style="margin-left:auto">
          <div class="label">&nbsp;</div>
          <el-button :loading="loadingSummary" :disabled="!dbId" @click="reloadAll">刷新</el-button>
        </div>
      </div>
    </el-card>

    <!-- 导入 supervised_data -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>导入 supervised_data</span>
          <el-tag effect="plain">POST /sar/datasets/import</el-tag>
        </div>
      </template>

      <div class="row" style="align-items: end">
        <div class="field">
          <div class="label">导入模式</div>
          <el-radio-group v-model="importMode">
            <el-radio label="merge">合并导入（去重）</el-radio>
            <el-radio label="replace">替换导入（覆盖）</el-radio>
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

        <div class="field">
          <div class="label">&nbsp;</div>
          <el-button type="primary" :disabled="!dbId || !importFile" :loading="importing" @click="doImport">
            开始导入
          </el-button>
        </div>
      </div>

      <el-alert
        style="margin-top: 12px"
        type="info"
        show-icon
        :closable="false"
        title="支持格式：JSON 数组 或 JSONL（每行一个 JSON）。每条至少需包含 question + sql（或 query）。"
      />

      <el-alert
        v-if="importResult"
        style="margin-top: 12px"
        type="success"
        show-icon
        :closable="false"
        :title="`导入完成：总 ${importResult.imported_total}，有效 ${importResult.imported_valid}，无效 ${importResult.imported_invalid}，新增 ${importResult.added}；Dataset v${importResult.dataset.version}`"
      />
    </el-card>

    <!-- 索引构建 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>SAR 索引构建</span>
          <el-tag effect="plain">POST /sar/index/build，GET /sar/index/status</el-tag>
        </div>
      </template>

      <div class="row">
        <div class="field">
          <div class="label">索引状态</div>
          <el-tag :type="indexTagType" effect="plain">{{ indexStatusLabel }}</el-tag>
        </div>

        <div class="field">
          <div class="label">Embedding 数量</div>
          <el-tag type="success" effect="plain">{{ indexStatus?.total_embeddings ?? 0 }}</el-tag>
        </div>

        <div class="field">
          <div class="label">最近构建</div>
          <el-tag effect="plain">{{ indexStatus?.updated_at || '-' }}</el-tag>
        </div>

        <div class="field">
          <div class="label">构建模式</div>
          <el-tag effect="plain">{{ indexStatus?.last_build_mode || '-' }}</el-tag>
        </div>

        <div class="field" style="margin-left:auto">
          <div class="label">&nbsp;</div>
          <el-button type="primary" plain :disabled="!dbId" :loading="buildingIndex" @click="buildIndex('incremental')">
            增量构建
          </el-button>
          <el-button type="primary" :disabled="!dbId" :loading="buildingIndex" @click="buildIndex('full')" style="margin-left: 8px">
            全量重建
          </el-button>
        </div>
      </div>
    </el-card>

    <!-- 样本列表 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>样本列表</span>
          <div style="display:flex; gap:10px; align-items:center;">
            <el-tag effect="plain">GET /sar/samples，DELETE /sar/samples</el-tag>
            <el-button :disabled="!dbId" :loading="loadingSamples" @click="loadSamples">刷新</el-button>
          </div>
        </div>
      </template>

      <div class="row" style="align-items: end">
        <div class="field" style="flex: 1; min-width: 320px">
          <div class="label">搜索（question / sql）</div>
          <el-input v-model="keyword" placeholder="输入关键词回车搜索" @keyup.enter="onSearch" clearable />
        </div>
        <div class="field">
          <div class="label">每页</div>
          <el-select v-model="pageSize" style="width: 120px" @change="onPageSizeChange">
            <el-option :value="10" label="10" />
            <el-option :value="20" label="20" />
            <el-option :value="50" label="50" />
          </el-select>
        </div>
        <div class="field">
          <div class="label">&nbsp;</div>
          <el-button type="primary" plain :disabled="!dbId" @click="onSearch">搜索</el-button>
          <el-button :disabled="!dbId" @click="resetSearch" style="margin-left: 8px">重置</el-button>
        </div>
      </div>

      <el-table
        v-loading="loadingSamples"
        :data="samples"
        size="small"
        style="width: 100%; margin-top: 12px"
      >
        <el-table-column prop="created_at" label="时间" width="170" />
        <el-table-column prop="source" label="来源" width="90">
          <template #default="{ row }">
            <el-tag effect="plain">{{ row.source || '-' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="verified" label="已验证" width="90">
          <template #default="{ row }">
            <el-tag v-if="row.verified" type="success" effect="plain">是</el-tag>
            <el-tag v-else type="warning" effect="plain">否</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="question" label="Question" min-width="260" show-overflow-tooltip />
        <el-table-column prop="sql" label="SQL" min-width="240" show-overflow-tooltip />

        <el-table-column label="操作" width="210" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="openDetail(row)">详情</el-button>
            <el-button size="small" type="danger" plain @click="delSample(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

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
      <div class="detail">
        <div class="drow"><span class="k">sample_id：</span><span class="v">{{ detail?.sample_id }}</span></div>
        <div class="drow"><span class="k">db_id：</span><span class="v">{{ detail?.db_id }}</span></div>
        <div class="drow"><span class="k">source：</span><span class="v">{{ detail?.source }}</span></div>
        <div class="drow"><span class="k">created_at：</span><span class="v">{{ detail?.created_at }}</span></div>
        <div class="drow"><span class="k">verified：</span><span class="v">{{ detail?.verified ? 'true' : 'false' }}</span></div>

        <el-divider />

        <div class="k2">question</div>
        <el-input type="textarea" :rows="4" :model-value="detail?.question" readonly />

        <div class="k2" style="margin-top: 10px">sql</div>
        <el-input type="textarea" :rows="6" :model-value="detail?.sql" readonly />

        <div style="margin-top: 10px; display:flex; gap:10px;">
          <el-button @click="copy(detail?.question || '')">复制问题</el-button>
          <el-button @click="copy(detail?.sql || '')">复制 SQL</el-button>
        </div>
      </div>

      <template #footer>
        <el-button @click="detailVisible=false">关闭</el-button>
      </template>
    </el-dialog>
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
.wrap {
  padding: 20px;
  background: #f5f7fb;
  min-height: 100vh;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei";
}

.top {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}
.title { font-size: 22px; font-weight: 900; color: #111827; }
.sub { margin-top: 6px; color: #6b7280; }

.top-actions { display: flex; gap: 10px; }

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.row {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  align-items: center;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.label { font-size: 12px; color: #6b7280; }

.pager {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
}

.detail .drow { margin: 6px 0; }
.detail .k { color: #6b7280; display:inline-block; width: 110px; }
.detail .v { color: #111827; font-weight: 600; }
.k2 { color:#6b7280; font-size: 12px; margin-bottom: 6px; }
</style>
