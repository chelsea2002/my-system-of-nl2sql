<template>
  <div class="dashboard-page">
    <div class="page-shell">
      <!-- 顶部欢迎区 -->
      <section class="hero-card">
        <div class="hero-left">
          <div class="hero-badge">智能数据应用平台</div>
          <h1 class="hero-title">数据库智能查询与分析系统</h1>
          <p class="hero-desc">
            面向 Schema 解析、SAR 检索增强与 Text-to-SQL 查询的一体化工作台，
            支持数据库接入、知识索引构建、自然语言查询与历史分析。
          </p>

          <div class="hero-actions">
            <el-button type="primary" @click="go('nl2sql')">开始提问</el-button>
            <el-button @click="go('history')">查看历史</el-button>
          </div>
        </div>

        <div class="hero-right">
          <div class="hero-user-card">
            <div class="hero-user-top">
              <div>
                <div class="panel-label">当前用户</div>
                <div class="panel-value">
                  {{ auth.user?.username || summary?.user?.username || '未知用户' }}
                </div>
              </div>
              <el-tag type="primary" effect="light">
                {{ auth.user?.role || summary?.user?.role || '未分配角色' }}
              </el-tag>
            </div>

            <div class="hero-meta">
              <div class="meta-item">
                <span class="meta-k">当前数据库</span>
                <span class="meta-v">{{ dbOverview.currentDbName || '未选择' }}</span>
              </div>
              <div class="meta-item">
                <span class="meta-k">最近同步</span>
                <span class="meta-v">{{ summary?.system?.lastSyncAt || '-' }}</span>
              </div>
            </div>

            <div class="hero-user-actions">
              <el-button plain @click="reload" :loading="loading">刷新数据</el-button>
              <el-button @click="logout">退出登录</el-button>
            </div>
          </div>
        </div>
      </section>

      <!-- 统计数据 -->
      <section class="stats-grid">
        <el-card class="metric-card metric-blue" shadow="hover">
          <div class="metric-head">
            <span class="metric-title">累计查询</span>
            <span class="metric-dot"></span>
          </div>
          <div class="metric-value">{{ stats.totalQueries ?? '-' }}</div>
          <div class="metric-desc">历史自然语言查询总量</div>
        </el-card>

        <el-card class="metric-card metric-green" shadow="hover">
          <div class="metric-head">
            <span class="metric-title">今日查询</span>
            <span class="metric-dot"></span>
          </div>
          <div class="metric-value">{{ stats.todayQueries ?? '-' }}</div>
          <div class="metric-desc">当天产生的查询次数</div>
        </el-card>

        <el-card class="metric-card metric-orange" shadow="hover">
          <div class="metric-head">
            <span class="metric-title">成功率</span>
            <span class="metric-dot"></span>
          </div>
          <div class="metric-value">
            {{ stats.successRate != null ? (stats.successRate * 100).toFixed(1) + '%' : '-' }}
          </div>
          <div class="metric-desc">SQL 可执行并返回结果的占比</div>
        </el-card>

        <el-card class="metric-card metric-purple" shadow="hover">
          <div class="metric-head">
            <span class="metric-title">平均耗时</span>
            <span class="metric-dot"></span>
          </div>
          <div class="metric-value">
            {{ stats.avgLatencyMs != null ? stats.avgLatencyMs + 'ms' : '-' }}
          </div>
          <div class="metric-desc">生成与执行链路平均延迟</div>
        </el-card>
      </section>

      <!-- 主体 -->
      <section class="main-grid">
        <!-- 左侧 -->
        <div class="main-col">
          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">快捷入口</div>
                  <div class="panel-subtitle">常用功能模块快速进入</div>
                </div>
              </div>
            </template>

            <div class="quick-grid">
              <div class="quick-card" @click="go('schema')">
                <div class="quick-icon quick-icon-blue">S</div>
                <div class="quick-body">
                  <div class="quick-title">模式解析与对齐</div>
                  <div class="quick-desc">上传数据库文件，解析 Schema 结构并生成模式信息</div>
                </div>
                <el-button type="primary" plain>进入</el-button>
              </div>

              <div class="quick-card" @click="go('index')">
                <div class="quick-icon quick-icon-green">I</div>
                <div class="quick-body">
                  <div class="quick-title">知识库索引构建</div>
                  <div class="quick-desc">向量化模式信息与历史问答，构建 SAR 检索索引</div>
                </div>
                <el-button type="primary" plain>进入</el-button>
              </div>

              <div class="quick-card" @click="go('nl2sql')">
                <div class="quick-icon quick-icon-orange">Q</div>
                <div class="quick-body">
                  <div class="quick-title">Text-to-SQL 查询</div>
                  <div class="quick-desc">输入自然语言问题，生成 SQL 并返回结果与图表</div>
                </div>
                <el-button type="primary" plain>进入</el-button>
              </div>

              <div class="quick-card" @click="go('history')">
                <div class="quick-icon quick-icon-purple">H</div>
                <div class="quick-body">
                  <div class="quick-title">数据与历史管理</div>
                  <div class="quick-desc">查看数据库结构、管理历史查询记录与 SQL 日志</div>
                </div>
                <el-button type="primary" plain>进入</el-button>
              </div>
            </div>
          </el-card>

          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">最近查询</div>
                  <div class="panel-subtitle">近期自然语言查询与 SQL 生成记录</div>
                </div>
                <div class="panel-actions">
                  <el-button size="small" @click="reload" :loading="loading">刷新</el-button>
                  <el-button size="small" type="primary" @click="go('nl2sql')">去提问</el-button>
                </div>
              </div>
            </template>

            <div class="table-shell">
              <el-table :data="recentQueries" size="small" style="width: 100%">
                <el-table-column prop="time" label="时间" width="180" />
                <el-table-column prop="question" label="问题" min-width="260" show-overflow-tooltip />
                <el-table-column label="状态" width="100">
                  <template #default="{ row }">
                    <el-tag :type="row.ok ? 'success' : 'danger'" effect="light">
                      {{ row.ok ? '成功' : '失败' }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="170">
                  <template #default="{ row }">
                    <div class="table-actions">
                      <el-button size="small" @click="viewSQL(row)">SQL</el-button>
                      <el-button size="small" type="primary" plain @click="reRun(row)">复用</el-button>
                    </div>
                  </template>
                </el-table-column>
              </el-table>

              <el-empty
                v-if="!loading && recentQueries.length === 0"
                description="暂无查询记录"
              />
            </div>
          </el-card>
        </div>

        <!-- 右侧 -->
        <div class="main-col right-col">
          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">系统状态</div>
                  <div class="panel-subtitle">核心模块运行与可用性概览</div>
                </div>
                <el-tag type="info" effect="plain">
                  最近同步：{{ summary?.system?.lastSyncAt || '-' }}
                </el-tag>
              </div>
            </template>

            <div class="status-list">
              <div class="status-card">
                <div>
                  <div class="status-title">模式链接</div>
                  <div class="status-desc">模式剪枝与字段链接准备情况</div>
                </div>
                <el-tag :type="summary?.system?.schemaLinked ? 'success' : 'warning'" effect="light">
                  {{ summary?.system?.schemaLinked ? '就绪' : '未就绪' }}
                </el-tag>
              </div>

              <div class="status-card">
                <div>
                  <div class="status-title">SAR 索引</div>
                  <div class="status-desc">检索增强索引构建状态</div>
                </div>
                <el-tag :type="summary?.system?.indexingReady ? 'success' : 'warning'" effect="light">
                  {{ summary?.system?.indexingReady ? '就绪' : '未就绪' }}
                </el-tag>
              </div>

              <div class="status-card">
                <div>
                  <div class="status-title">生成引擎</div>
                  <div class="status-desc">Text-to-SQL 模型服务可用性</div>
                </div>
                <el-tag :type="summary?.system?.llmReady ? 'success' : 'warning'" effect="light">
                  {{ summary?.system?.llmReady ? '就绪' : '未就绪' }}
                </el-tag>
              </div>
            </div>

            <div class="status-btns">
              <el-button type="primary" plain @click="go('schema')">上传数据库并解析</el-button>
              <el-button type="primary" plain @click="go('index')">构建索引</el-button>
            </div>
          </el-card>

          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">数据库概览</div>
                  <div class="panel-subtitle">当前连接库与统计信息</div>
                </div>
                <div class="panel-actions">
                  <el-button size="small" type="primary" plain @click="go('history')">管理</el-button>
                </div>
              </div>
            </template>

            <div class="db-select-row">
              <span class="db-select-label">选择数据库</span>
              <el-select
                v-model="dbOverview.currentDbId"
                size="default"
                class="db-select"
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

            <div class="db-overview-grid">
              <div class="db-kpi-card">
                <div class="db-kpi-label">已连接库数量</div>
                <div class="db-kpi-value">{{ dbOverview.connectedCount ?? '-' }}</div>
              </div>
              <div class="db-kpi-card">
                <div class="db-kpi-label">当前数据库</div>
                <div class="db-kpi-value db-name">{{ dbOverview.currentDbName || '-' }}</div>
              </div>
              <div class="db-kpi-card">
                <div class="db-kpi-label">表数量</div>
                <div class="db-kpi-value">{{ dbOverview.tables ?? '-' }}</div>
              </div>
              <div class="db-kpi-card">
                <div class="db-kpi-label">字段数量</div>
                <div class="db-kpi-value">{{ dbOverview.columns ?? '-' }}</div>
              </div>
            </div>

            <el-alert
              title="后续可扩展展示 Schema 树、上传数据库列表、连接状态与权限信息"
              type="info"
              show-icon
              :closable="false"
              class="db-alert"
            />
          </el-card>

          <el-card class="panel-card dev-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">接口概览</div>
                  <div class="panel-subtitle">当前首页依赖的核心接口</div>
                </div>
              </div>
            </template>

            <el-descriptions :column="1" size="small" border>
              <el-descriptions-item label="GET /history/stats">
                统计：累计/今日/成功率/平均耗时
              </el-descriptions-item>
              <el-descriptions-item label="GET /history">
                最近查询列表
              </el-descriptions-item>
              <el-descriptions-item label="GET /db/list">
                数据库列表
              </el-descriptions-item>
              <el-descriptions-item label="GET /sar/datasets/summary?db_id=">
                表/字段统计
              </el-descriptions-item>
            </el-descriptions>
          </el-card>
        </div>
      </section>
    </div>

    <!-- SQL 弹窗 -->
    <el-dialog v-model="sqlDialogVisible" title="SQL 预览" width="720">
      <el-input v-model="sqlDialogText" type="textarea" :rows="12" readonly />
      <template #footer>
        <el-button @click="sqlDialogVisible = false">关闭</el-button>
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
  // ✅ 既支持 query 传参，也支持 localStorage（避免 URL 太长）
  const payload = {
    db_id: row?.db_id || '',
    question: row?.question || '',
  }
  localStorage.setItem('nl2sql_reuse_payload', JSON.stringify(payload))

  // 跳转到 Text-to-SQL 页面，并带一个标记
  router.push({ path: '/nl2sql', query: { reuse: '1' } })
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
.dashboard-page {
  min-height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 22%),
    radial-gradient(circle at top right, rgba(16, 185, 129, 0.06), transparent 20%),
    linear-gradient(180deg, #f6f8fc 0%, #f3f6fb 100%);
  padding: 24px;
  box-sizing: border-box;
}

.page-shell {
  max-width: 1440px;
  margin: 0 auto;
}

.hero-card {
  display: grid;
  grid-template-columns: 1.3fr 0.7fr;
  gap: 20px;
  padding: 28px;
  margin-bottom: 18px;
  border-radius: 24px;
  background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #3b82f6 100%);
  color: #fff;
  box-shadow: 0 18px 45px rgba(37, 99, 235, 0.18);
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  height: 30px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.14);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.5px;
}

.hero-title {
  margin: 16px 0 10px;
  font-size: 30px;
  line-height: 1.2;
  font-weight: 800;
}

.hero-desc {
  margin: 0;
  max-width: 760px;
  font-size: 14px;
  line-height: 1.9;
  color: rgba(255, 255, 255, 0.88);
}

.hero-actions {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.hero-right {
  display: flex;
  align-items: stretch;
}

.hero-user-card {
  width: 100%;
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 20px;
  padding: 18px;
  backdrop-filter: blur(8px);
}

.hero-user-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.panel-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.72);
  margin-bottom: 6px;
}

.panel-value {
  font-size: 22px;
  font-weight: 800;
  color: #fff;
}

.hero-meta {
  margin-top: 18px;
  display: grid;
  gap: 12px;
}

.meta-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.18);
}

.meta-item:last-child {
  border-bottom: none;
}

.meta-k {
  color: rgba(255, 255, 255, 0.76);
  font-size: 13px;
}

.meta-v {
  color: #fff;
  font-weight: 600;
  text-align: right;
}

.hero-user-actions {
  display: flex;
  gap: 10px;
  margin-top: 16px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 18px;
}

.metric-card {
  border: none;
  border-radius: 20px;
  overflow: hidden;
}

.metric-card :deep(.el-card__body) {
  padding: 20px;
}

.metric-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-title {
  font-size: 13px;
  color: #6b7280;
  font-weight: 700;
}

.metric-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.9;
}

.metric-value {
  margin: 14px 0 6px;
  font-size: 34px;
  font-weight: 800;
  line-height: 1.1;
  color: #111827;
}

.metric-desc {
  color: #9ca3af;
  font-size: 12px;
  line-height: 1.7;
}

.metric-blue {
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
  color: #3b82f6;
}

.metric-green {
  background: linear-gradient(180deg, #ffffff 0%, #f5fffb 100%);
  color: #10b981;
}

.metric-orange {
  background: linear-gradient(180deg, #ffffff 0%, #fffaf3 100%);
  color: #f59e0b;
}

.metric-purple {
  background: linear-gradient(180deg, #ffffff 0%, #faf7ff 100%);
  color: #8b5cf6;
}

.main-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
  gap: 18px;
  align-items: start;
}

.main-col {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.panel-card {
  border: 1px solid #e8edf5;
  border-radius: 20px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.panel-card :deep(.el-card__header) {
  padding: 18px 20px 14px;
  border-bottom: 1px solid #edf2f7;
}

.panel-card :deep(.el-card__body) {
  padding: 18px 20px 20px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
}

.panel-title {
  font-size: 17px;
  font-weight: 800;
  color: #111827;
  line-height: 1.2;
}

.panel-subtitle {
  margin-top: 6px;
  font-size: 12px;
  color: #6b7280;
}

.panel-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}

.quick-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.quick-card {
  display: grid;
  grid-template-columns: 52px 1fr auto;
  align-items: center;
  gap: 14px;
  padding: 18px;
  border: 1px solid #edf2f7;
  border-radius: 18px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
  cursor: pointer;
  transition: all 0.2s ease;
}

.quick-card:hover {
  transform: translateY(-2px);
  border-color: #dbe7f7;
  box-shadow: 0 12px 28px rgba(59, 130, 246, 0.08);
}

.quick-icon {
  width: 52px;
  height: 52px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 800;
  color: #fff;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.24);
}

.quick-icon-blue {
  background: linear-gradient(135deg, #2563eb, #60a5fa);
}

.quick-icon-green {
  background: linear-gradient(135deg, #059669, #34d399);
}

.quick-icon-orange {
  background: linear-gradient(135deg, #ea580c, #fb923c);
}

.quick-icon-purple {
  background: linear-gradient(135deg, #7c3aed, #a78bfa);
}

.quick-body {
  min-width: 0;
}

.quick-title {
  font-size: 15px;
  font-weight: 800;
  color: #111827;
}

.quick-desc {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
  line-height: 1.7;
}

.table-shell {
  border-radius: 14px;
  overflow: hidden;
}

.table-actions {
  display: flex;
  gap: 8px;
}

.status-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-card {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 14px;
  padding: 14px 16px;
  border: 1px solid #edf2f7;
  border-radius: 16px;
  background: linear-gradient(180deg, #ffffff 0%, #fbfcff 100%);
}

.status-title {
  font-size: 14px;
  font-weight: 800;
  color: #111827;
}

.status-desc {
  margin-top: 6px;
  font-size: 12px;
  color: #6b7280;
  line-height: 1.6;
}

.status-btns {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}

.db-select-row {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 16px;
}

.db-select-label {
  font-size: 13px;
  font-weight: 700;
  color: #374151;
}

.db-select {
  width: 100%;
}

.db-overview-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.db-kpi-card {
  padding: 16px;
  border: 1px solid #edf2f7;
  border-radius: 16px;
  background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
}

.db-kpi-label {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 10px;
}

.db-kpi-value {
  font-size: 24px;
  font-weight: 800;
  color: #111827;
  line-height: 1.2;
  word-break: break-word;
}

.db-name {
  font-size: 18px;
}

.db-alert {
  margin-top: 16px;
}

.dev-card :deep(.el-descriptions__label) {
  width: 220px;
}

:deep(.el-table) {
  --el-table-header-bg-color: #f8fafc;
  --el-table-row-hover-bg-color: #f7fbff;
  border-radius: 14px;
}

:deep(.el-table th.el-table__cell) {
  font-weight: 700;
  color: #374151;
}

:deep(.el-button + .el-button) {
  margin-left: 0;
}

@media (max-width: 1280px) {
  .hero-card {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .main-grid {
    grid-template-columns: 1fr;
  }

  .right-col {
    min-width: 0;
  }
}

@media (max-width: 900px) {
  .dashboard-page {
    padding: 14px;
  }

  .hero-card {
    padding: 20px;
    border-radius: 20px;
  }

  .hero-title {
    font-size: 24px;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .quick-grid {
    grid-template-columns: 1fr;
  }

  .quick-card {
    grid-template-columns: 48px 1fr;
  }

  .quick-card .el-button {
    grid-column: 2;
    justify-self: start;
    margin-top: 6px;
  }

  .db-overview-grid {
    grid-template-columns: 1fr;
  }

  .panel-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .panel-actions,
  .status-btns,
  .hero-actions,
  .hero-user-actions {
    flex-wrap: wrap;
  }
}
</style>
