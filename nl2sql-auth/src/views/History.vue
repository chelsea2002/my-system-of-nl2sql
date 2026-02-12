<template>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">历史管理</div>
        <div class="sub">查看查询历史、SQL 日志，并将“确认正确”的样本加入 SAR 样本库。</div>
      </div>
      <div class="top-actions">
        <el-button @click="goDashboard">返回工作台</el-button>
        <el-button type="primary" plain @click="goSar">去样本库管理</el-button>
      </div>
    </div>

    <el-card>
      <template #header>
        <div class="card-header">
          <span>历史查询列表</span>
          <el-tag effect="plain">GET /history</el-tag>
        </div>
      </template>

      <div class="row" style="align-items:end">
        <div class="field">
          <div class="label">db_id（可选）</div>
          <el-select v-model="dbId" placeholder="全部数据库" style="width: 220px" @change="onSearch">
            <el-option label="全部" value="" />
            <el-option v-for="db in dbList" :key="db.db_id" :label="db.name" :value="db.db_id" />
          </el-select>
        </div>

        <div class="field" style="flex:1; min-width: 320px">
          <div class="label">搜索（question / sql）</div>
          <el-input v-model="keyword" placeholder="回车搜索" @keyup.enter="onSearch" clearable />
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
          <el-button type="primary" plain @click="onSearch">搜索</el-button>
          <el-button @click="reset" style="margin-left: 8px">重置</el-button>
        </div>
      </div>

      <el-table
        v-loading="loading"
        :data="items"
        size="small"
        style="width: 100%; margin-top: 12px"
      >
        <el-table-column prop="created_at" label="时间" width="170" />
        <el-table-column prop="db_id" label="db_id" width="120" />
        <el-table-column label="状态" width="90">
          <template #default="{ row }">
            <el-tag :type="row.ok ? 'success' : 'danger'" effect="plain">
              {{ row.ok ? '成功' : '失败' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="question" label="Question" min-width="260" show-overflow-tooltip />
        <el-table-column prop="sql" label="SQL" min-width="260" show-overflow-tooltip />

        <el-table-column label="操作" width="280" fixed="right">
          <template #default="{ row }">
            <el-button size="small" @click="openDetail(row)">详情</el-button>
            <el-button size="small" @click="copy(row.sql)">复制 SQL</el-button>
            <el-button size="small" type="primary" plain @click="openAddSar(row)">加入 SAR</el-button>
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

      <el-empty v-if="!loading && items.length === 0" description="暂无历史记录" />
    </el-card>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" title="历史详情" width="820">
      <div class="detail">
        <div class="drow"><span class="k">id：</span><span class="v">{{ detail?.id }}</span></div>
        <div class="drow"><span class="k">db_id：</span><span class="v">{{ detail?.db_id }}</span></div>
        <div class="drow"><span class="k">时间：</span><span class="v">{{ detail?.created_at }}</span></div>
        <div class="drow">
          <span class="k">状态：</span>
          <el-tag :type="detail?.ok ? 'success' : 'danger'" effect="plain">
            {{ detail?.ok ? '成功' : '失败' }}
          </el-tag>
          <span style="margin-left:10px; color:#6b7280">rows={{ detail?.rows ?? '-' }}</span>
        </div>

        <el-divider />

        <div class="k2">question</div>
        <el-input type="textarea" :rows="4" :model-value="detail?.question" readonly />

        <div class="k2" style="margin-top: 10px">sql</div>
        <el-input type="textarea" :rows="7" :model-value="detail?.sql" readonly />

        <div class="k2" style="margin-top: 10px">备注</div>
        <el-input type="textarea" :rows="3" :model-value="detail?.note || ''" readonly />

        <div style="margin-top: 10px; display:flex; gap:10px;">
          <el-button @click="copy(detail?.question || '')">复制问题</el-button>
          <el-button @click="copy(detail?.sql || '')">复制 SQL</el-button>
        </div>
      </div>

      <template #footer>
        <el-button @click="detailVisible=false">关闭</el-button>
      </template>
    </el-dialog>

    <!-- 加入 SAR 弹窗 -->
    <el-dialog v-model="addSarVisible" title="加入 SAR 样本库" width="860">
      <el-alert
        type="info"
        show-icon
        :closable="false"
        title="建议只把“确认正确”的问答加入 SAR。你可以在这里微调 question / sql 再提交。"
      />

      <div class="form" style="margin-top: 12px">
        <div class="row">
          <div class="field" style="width: 240px">
            <div class="label">db_id</div>
            <el-input v-model="sarForm.db_id" readonly />
          </div>
          <div class="field" style="width: 240px">
            <div class="label">source</div>
            <el-input v-model="sarForm.source" readonly />
          </div>
          <div class="field" style="width: 240px">
            <div class="label">verified</div>
            <el-switch v-model="sarForm.verified" />
          </div>
        </div>

        <div class="field" style="margin-top: 10px">
          <div class="label">question（可编辑）</div>
          <el-input v-model="sarForm.question" />
        </div>

        <div class="field" style="margin-top: 10px">
          <div class="label">sql（可编辑）</div>
          <el-input type="textarea" :rows="8" v-model="sarForm.sql" />
        </div>

        <el-alert
          style="margin-top: 10px"
          type="warning"
          show-icon
          :closable="false"
          title="schema 快照建议由后端根据 db_id 绑定（此处先占位，后续可在提交时由后端补齐）。"
        />
      </div>

      <template #footer>
        <el-button @click="addSarVisible=false">取消</el-button>
        <el-button type="primary" :loading="addingSar" @click="submitAddSar">提交</el-button>
      </template>
    </el-dialog>
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
.top-actions { display:flex; gap:10px; }

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
  align-items: center;
}
.field { display:flex; flex-direction: column; gap:6px; }
.label { font-size: 12px; color:#6b7280; }

.pager {
  margin-top: 12px;
  display:flex;
  justify-content: flex-end;
}

.detail .drow { margin: 6px 0; }
.detail .k { color:#6b7280; display:inline-block; width: 100px; }
.detail .v { color:#111827; font-weight: 600; }
.k2 { color:#6b7280; font-size: 12px; margin-bottom: 6px; }
</style>
