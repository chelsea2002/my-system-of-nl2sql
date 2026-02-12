<template>
  <div class="wrap">
    <!-- 顶部栏 -->
    <div class="topbar">
      <div class="left">
        <div class="title">工作台</div>
        <div class="subtitle">数据库智能查询与分析系统 · Schema / SAR / Text-to-SQL</div>
      </div>

      <div class="right">
        <el-tag v-if="dbOverview.currentDbName" effect="plain">
          当前数据库：{{ dbOverview.currentDbName }}
        </el-tag>
        <div class="user">
          <span>用户：</span>
          <b>{{ auth.user?.username || summary?.user?.username }}</b>
          <span class="role">（{{ auth.user?.role || summary?.user?.role }}）</span>
        </div>
        <el-button @click="logout">退出</el-button>
      </div>
    </div>

    <!-- 统计卡片 -->
    <div class="grid4">
      <el-card class="stat">
        <div class="stat-k">累计查询</div>
        <div class="stat-v">{{ stats.totalQueries ?? '-' }}</div>
        <div class="stat-h">历史自然语言查询总量</div>
      </el-card>

      <el-card class="stat">
        <div class="stat-k">今日查询</div>
        <div class="stat-v">{{ stats.todayQueries ?? '-' }}</div>
        <div class="stat-h">当天查询次数</div>
      </el-card>

      <el-card class="stat">
        <div class="stat-k">成功率</div>
        <div class="stat-v">
          {{ stats.successRate != null ? (stats.successRate * 100).toFixed(1) + '%' : '-' }}
        </div>
        <div class="stat-h">SQL 可执行/返回结果占比</div>
      </el-card>

      <el-card class="stat">
        <div class="stat-k">平均耗时</div>
        <div class="stat-v">{{ stats.avgLatencyMs != null ? stats.avgLatencyMs + 'ms' : '-' }}</div>
        <div class="stat-h">生成 + 执行链路平均延迟</div>
      </el-card>
    </div>

    <!-- 主体两列 -->
    <div class="grid2">
      <!-- 左侧：快捷入口 + 最近查询 -->
      <div class="col">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>快捷入口</span>
              <el-text type="info">（点击后可路由到对应模块页面）</el-text>
            </div>
          </template>

          <div class="quick">
            <div class="quick-item" @click="go('schema')">
              <div class="q-title">模式解析与对齐</div>
              <div class="q-desc">上传数据库文件，解析 Schema 结构并生成模式信息</div>
              <el-button type="primary" plain>进入</el-button>
            </div>

            <div class="quick-item" @click="go('index')">
              <div class="q-title">知识库索引构建（SAR）</div>
              <div class="q-desc">向量化模式信息/历史问答，构建检索索引</div>
              <el-button type="primary" plain>进入</el-button>
            </div>

            <div class="quick-item" @click="go('nl2sql')">
              <div class="q-title">Text-to-SQL 查询</div>
              <div class="q-desc">输入自然语言问题，生成 SQL 并返回结果与图表</div>
              <el-button type="primary" plain>进入</el-button>
            </div>

            <div class="quick-item" @click="go('history')">
              <div class="q-title">数据与历史管理</div>
              <div class="q-desc">查看数据库结构、管理历史查询记录与 SQL 日志</div>
              <el-button type="primary" plain>进入</el-button>
            </div>
          </div>
        </el-card>

        <el-card style="margin-top: 14px">
          <template #header>
            <div class="card-header">
              <span>最近查询</span>
              <div style="display:flex; gap:10px; align-items:center;">
                <el-button size="small" @click="reload" :loading="loading">刷新</el-button>
                <el-button size="small" type="primary" plain @click="go('nl2sql')">去提问</el-button>
              </div>
            </div>
          </template>

          <el-table :data="recentQueries" size="small" style="width: 100%">
            <el-table-column prop="time" label="时间" width="170" />
            <el-table-column prop="question" label="问题" min-width="220" show-overflow-tooltip />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="row.ok ? 'success' : 'danger'" effect="plain">
                  {{ row.ok ? '成功' : '失败' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="160">
              <template #default="{ row }">
                <el-button size="small" @click="viewSQL(row)">SQL</el-button>
                <el-button size="small" type="primary" plain @click="reRun(row)">复用</el-button>
              </template>
            </el-table-column>
          </el-table>

          <el-empty v-if="!loading && recentQueries.length === 0" description="暂无查询记录" />
        </el-card>
      </div>

      <!-- 右侧：系统状态 + 数据库概览 -->
      <div class="col">
        <el-card>
          <template #header>
            <div class="card-header">
              <span>系统状态</span>
              <el-text type="info">最近同步：{{ summary?.system?.lastSyncAt || '-' }}</el-text>
            </div>
          </template>

          <div class="status">
            <div class="status-item">
              <div class="s-left">
                <div class="s-title">模式链接</div>
                <div class="s-desc">模式剪枝与字段链接准备情况</div>
              </div>
              <el-tag :type="summary?.system?.schemaLinked ? 'success' : 'warning'" effect="plain">
                {{ summary?.system?.schemaLinked ? '就绪' : '未就绪' }}
              </el-tag>
            </div>

            <div class="status-item">
              <div class="s-left">
                <div class="s-title">SAR 索引</div>
                <div class="s-desc">检索增强所需索引构建状态</div>
              </div>
              <el-tag :type="summary?.system?.indexingReady ? 'success' : 'warning'" effect="plain">
                {{ summary?.system?.indexingReady ? '就绪' : '未就绪' }}
              </el-tag>
            </div>

            <div class="status-item">
              <div class="s-left">
                <div class="s-title">POSG / 生成引擎</div>
                <div class="s-desc">Text-to-SQL 生成组件可用性</div>
              </div>
              <el-tag :type="summary?.system?.llmReady ? 'success' : 'warning'" effect="plain">
                {{ summary?.system?.llmReady ? '就绪' : '未就绪' }}
              </el-tag>
            </div>
          </div>

          <div class="status-actions">
            <el-button type="primary" plain @click="go('schema')">上传数据库并解析</el-button>
            <el-button type="primary" plain @click="go('index')">构建索引</el-button>
          </div>
        </el-card>

        <el-card style="margin-top: 14px">
          <template #header>
            <div class="card-header">
              <span>数据库概览</span>
              <div style="display:flex; align-items:center; gap:10px;">
                <el-button size="small" type="primary" plain @click="go('history')">管理</el-button>
                <el-select
                  v-model="dbOverview.currentDbId"
                  size="small"
                  style="width: 200px"
                  @change="onDbChange"
                >
                  <el-option
                    v-for="db in dbListForSelect"
                    :key="db.db_id"
                    :label="db.name || db.db_id"
                    :value="db.db_id"
                  />
                </el-select>
              </div>
            </div>
          </template>

          <div class="db">
            <div class="db-row">
              <span class="k">已连接库数量</span>
              <span class="v">{{ dbOverview.connectedCount ?? '-' }}</span>
            </div>
            <div class="db-row">
              <span class="k">当前数据库</span>
              <span class="v">{{ dbOverview.currentDbName || '-' }}</span>
            </div>
            <div class="db-row">
              <span class="k">表数量</span>
              <span class="v">{{ dbOverview.tables ?? '-' }}</span>
            </div>
            <div class="db-row">
              <span class="k">字段数量</span>
              <span class="v">{{ dbOverview.columns ?? '-' }}</span>
            </div>

            <el-divider />

            <el-alert
              title="这里后续可展示 schema 树、已上传数据库列表、连接状态、权限等"
              type="info"
              show-icon
              :closable="false"
            />
          </div>
        </el-card>

        <el-card style="margin-top: 14px">
          <template #header>
            <div class="card-header">
              <span>开发占位：接口清单</span>
            </div>
          </template>

          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="GET /history/stats">
              统计：累计/今日/成功率/平均耗时（本页使用）
            </el-descriptions-item>
            <el-descriptions-item label="GET /history">
              最近查询列表（本页使用）
            </el-descriptions-item>
            <el-descriptions-item label="GET /db/list">
              数据库列表（本页使用）
            </el-descriptions-item>
            <el-descriptions-item label="GET /sar/datasets/summary?db_id=">
              表/字段统计（本页使用）
            </el-descriptions-item>
          </el-descriptions>
        </el-card>
      </div>
    </div>

    <!-- SQL 弹窗 -->
    <el-dialog v-model="sqlDialogVisible" title="SQL 预览" width="700">
      <el-input v-model="sqlDialogText" type="textarea" :rows="10" readonly />
      <template #footer>
        <el-button @click="sqlDialogVisible=false">关闭</el-button>
        <el-button type="primary" @click="copySQL">复制</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { apiDashboardSummary } from '@/api/dashboard'
import { apiHistoryList, apiHistoryStats } from '@/api/history'
import { apiDbList, apiDbSummary } from '@/api/db' // 按你的路径调整

const router = useRouter()
const auth = useAuthStore()

const loading = ref(false)
const summary = ref(null)

const sqlDialogVisible = ref(false)
const sqlDialogText = ref('')

const recentQueries = ref([])

const stats = ref({
  totalQueries: 0,
  todayQueries: 0,
  successRate: null,
  avgLatencyMs: null,
})

const dbListForSelect = ref([])
const dbOverview = ref({
  connectedCount: 0,
  currentDbId: '',
  currentDbName: '',
  tables: null,
  columns: null,
})

onMounted(() => {
  load()
})

function onDbChange(v) {
  localStorage.setItem('current_db_id', v)
  load()
}

async function load() {
  loading.value = true
  try {
    // 1) 数据库列表 + 选中当前库
    const dbs = await apiDbList()
    dbListForSelect.value = dbs || []
    dbOverview.value.connectedCount = (dbs || []).length

    const saved = localStorage.getItem('current_db_id') || ''
    const cur = (dbs || []).find(x => x.db_id === saved) || (dbs || [])[0] || null

    dbOverview.value.currentDbId = cur?.db_id || ''
    dbOverview.value.currentDbName = cur?.name || cur?.db_id || ''

    // 2) 当前库的表/字段概览
    if (dbOverview.value.currentDbId) {
      const sum = await apiDbSummary(dbOverview.value.currentDbId)
      dbOverview.value.tables = sum?.tables_count ?? sum?.tables ?? sum?.n_tables ?? null
      dbOverview.value.columns = sum?.columns_count ?? sum?.columns ?? sum?.n_columns ?? null
    } else {
      dbOverview.value.tables = null
      dbOverview.value.columns = null
    }

    // 3) 统计：/history/stats
    const st = await apiHistoryStats({ db_id: dbOverview.value.currentDbId })
    stats.value.totalQueries = st?.totalQueries ?? 0
    stats.value.todayQueries = st?.todayQueries ?? 0
    stats.value.successRate = st?.successRate ?? null
    stats.value.avgLatencyMs = st?.avgLatencyMs ?? null

    // 4) 最近查询：/history
    const hist = await apiHistoryList({
      db_id: dbOverview.value.currentDbId,
      page: 1,
      page_size: 10,
    })
    const items = hist?.items || []
    recentQueries.value = items.map(it => ({
      id: it.id,
      time: fmtTime(it.created_at),
      question: it.question || '',
      ok: !!it.ok,
      sql: it.selected_sql || it.sql || '',
      db_id: it.db_id || '',
    }))

    // 5) 系统状态/用户信息（可最后再拉，避免影响 stats）
    summary.value = await apiDashboardSummary()
  } catch (e) {
    ElMessage.error(e?.message || '加载失败')
  } finally {
    loading.value = false
  }
}


function fmtTime(x) {
  if (!x) return '-'
  if (typeof x === 'number') {
    const d = new Date(x * 1000)
    return d.toLocaleString()
  }
  // 有些后端直接返回字符串时间
  return String(x)
}

function reload() {
  load()
}

function logout() {
  auth.logout()
  ElMessage.success('已退出')
  router.replace('/auth/login')
}

function go(module) {
  const map = {
    schema: '/schema',
    index: '/sar',
    nl2sql: '/nl2sql',
    history: '/history',
  }
  const path = map[module]
  if (path) router.push(path)
}

function viewSQL(row) {
  sqlDialogText.value = row.sql || ''
  sqlDialogVisible.value = true
}

function reRun(row) {
  // router.push({ path: '/nl2sql', query: { q: row.question } })
  ElMessage.info('占位：将问题带到 Text-to-SQL 页面复用')
}

async function copySQL() {
  try {
    await navigator.clipboard.writeText(sqlDialogText.value || '')
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

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 14px;
}

.title {
  font-size: 22px;
  font-weight: 900;
  color: #111827;
}
.subtitle {
  margin-top: 6px;
  color: #6b7280;
}

.right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.user { color: #374151; }
.role { color: #6b7280; margin-left: 4px; }

.grid4 {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 12px;
}

.stat { border-radius: 14px; }
.stat-k { color: #6b7280; font-size: 13px; }
.stat-v { font-size: 28px; font-weight: 900; margin: 10px 0 4px; }
.stat-h { color: #9ca3af; font-size: 12px; }

.grid2 {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 12px;
  align-items: start;
}

.card-header{
  display:flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.quick {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.quick-item {
  border: 1px solid #eef2f7;
  border-radius: 14px;
  padding: 14px;
  cursor: pointer;
  transition: all .15s ease;
  background: #ffffff;
}
.quick-item:hover {
  border-color: #dbeafe;
  box-shadow: 0 8px 18px rgba(17,24,39,0.06);
  transform: translateY(-1px);
}
.q-title { font-weight: 800; color: #111827; }
.q-desc { color: #6b7280; font-size: 12px; margin: 8px 0 12px; line-height: 1.6; }

.status { display: flex; flex-direction: column; gap: 10px; }
.status-item {
  display:flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid #eef2f7;
  border-radius: 12px;
}
.s-title { font-weight: 800; color: #111827; }
.s-desc { color: #6b7280; font-size: 12px; margin-top: 4px; }

.status-actions {
  margin-top: 12px;
  display: flex;
  gap: 10px;
}

.db { padding-top: 4px; }
.db-row {
  display:flex;
  justify-content: space-between;
  padding: 8px 2px;
  color: #374151;
}
.k { color: #6b7280; }
.v { font-weight: 800; color: #111827; }
</style>
