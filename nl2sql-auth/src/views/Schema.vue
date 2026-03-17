<template>
  <div class="schema-page">
    <div class="page-shell">
      <!-- 顶部 -->
      <section class="page-head">
        <div>
          <div class="page-title">模式解析与对齐</div>
          <div class="page-sub">
            上传 SQLite 数据库，解析 Schema 结构，并基于问题生成对齐后的模式文本。
          </div>
        </div>

        <div class="page-head-actions">
          <div class="head-chip" v-if="dbId">
            当前库：<b>{{ dbId }}</b>
          </div>
          <div class="head-chip">
            状态：<b>{{ taskStatusLabel }}</b>
          </div>
          <el-button @click="goDashboard">返回工作台</el-button>
        </div>
      </section>

      <!-- 主体：左操作，右状态 -->
      <section class="top-grid">
        <!-- 左侧主操作 -->
        <div class="main-col">
          <!-- 上传 -->
          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">1. 上传数据库</div>
                  <div class="panel-subtitle">选择 SQLite 文件并生成 db_id</div>
                </div>
                <el-tag effect="plain">POST /schema/upload</el-tag>
              </div>
            </template>

            <div class="form-grid">
              <div class="field span-2">
                <div class="label">db_id（可选）</div>
                <el-input
                  v-model="dbName"
                  placeholder="例如：spider_dev / demo_db"
                />
              </div>

              <div class="field span-2">
                <div class="label">选择 SQLite 文件</div>
                <el-upload
                  :auto-upload="false"
                  :show-file-list="true"
                  :limit="1"
                  :on-change="onFileChange"
                  :on-remove="onFileRemove"
                  accept=".sqlite,.db"
                >
                  <el-button type="primary" plain>选择文件</el-button>
                </el-upload>
              </div>

              <div class="field small-field">
                <div class="label">样例数量 k</div>
                <el-input-number v-model="kExamples" :min="1" :max="5" />
              </div>

              <div class="field action-field">
                <div class="label">&nbsp;</div>
                <el-button
                  type="primary"
                  :disabled="!file"
                  :loading="uploading"
                  @click="upload"
                >
                  上传数据库
                </el-button>
              </div>
            </div>

            <el-alert
              v-if="dbId"
              class="mt16"
              type="success"
              show-icon
              :closable="false"
              :title="`已上传：db_id=${dbId}`"
              :description="dbPath ? `db_path=${dbPath}` : ''"
            />
          </el-card>

          <!-- 解析 -->
          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">2. 解析 Schema</div>
                  <div class="panel-subtitle">同步执行解析任务并生成结构化结果</div>
                </div>
                <el-tag effect="plain">POST /schema/parse</el-tag>
              </div>
            </template>

            <div class="parse-toolbar">
              <div class="parse-status">
                <div class="status-pill">
                  <span class="status-label">状态</span>
                  <el-tag :type="statusTagType" effect="light">{{ taskStatusLabel }}</el-tag>
                </div>

                <div class="status-progress">
                  <span class="status-label">进度</span>
                  <el-progress :percentage="taskProgress" />
                </div>
              </div>

              <div class="parse-actions">
                <el-button
                  type="primary"
                  :disabled="!dbId || taskRunning"
                  :loading="startingTask"
                  @click="startParse"
                >
                  开始解析
                </el-button>

                <el-button :disabled="!dbId" @click="loadSchema">
                  拉取结果
                </el-button>
              </div>
            </div>

            <div class="log-box">
              <div class="section-title">任务日志</div>
              <el-input type="textarea" :rows="6" :model-value="taskLogsText" readonly />
            </div>

            <el-alert
              v-if="taskError"
              class="mt16"
              type="error"
              show-icon
              :closable="false"
              :title="taskError"
            />
          </el-card>
        </div>

        <!-- 右侧概览 -->
        <div class="side-col">
          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">任务概览</div>
                  <div class="panel-subtitle">当前解析任务与结果摘要</div>
                </div>
              </div>
            </template>

            <div class="summary-list">
              <div class="summary-item">
                <span class="summary-k">数据库 ID</span>
                <span class="summary-v">{{ dbId || '-' }}</span>
              </div>
              <div class="summary-item">
                <span class="summary-k">数据库路径</span>
                <span class="summary-v break">{{ dbPath || '-' }}</span>
              </div>
              <div class="summary-item">
                <span class="summary-k">解析状态</span>
                <span class="summary-v">
                  <el-tag :type="statusTagType" effect="light">{{ taskStatusLabel }}</el-tag>
                </span>
              </div>
              <div class="summary-item">
                <span class="summary-k">当前页签</span>
                <span class="summary-v">{{ activeTab }}</span>
              </div>
            </div>
          </el-card>

          <el-card class="panel-card" shadow="never">
            <template #header>
              <div class="panel-header">
                <div>
                  <div class="panel-title">快捷操作</div>
                  <div class="panel-subtitle">常用动作快速执行</div>
                </div>
              </div>
            </template>

            <div class="quick-actions">
              <el-button class="full-btn" plain :disabled="!dbId" @click="loadSchema">
                查看解析结果
              </el-button>
              <el-button class="full-btn" plain :disabled="!schemaText" @click="copy(schemaText)">
                复制 schema_text
              </el-button>
              <el-button class="full-btn" plain :disabled="!schemaText" @click="downloadText">
                下载 schema_text
              </el-button>
            </div>
          </el-card>
        </div>
      </section>

      <!-- 结果区 -->
      <el-card class="panel-card result-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">3. 解析结果</div>
              <div class="panel-subtitle">查看结构树、字段样例与 schema_text</div>
            </div>

            <div class="panel-actions">
              <el-tag effect="plain">GET /schema/{db_id}/json</el-tag>
              <el-button
                type="primary"
                plain
                :disabled="!dbId"
                :loading="loadingSchema"
                @click="loadSchema"
              >
                重新拉取
              </el-button>
            </div>
          </div>
        </template>

        <el-tabs v-model="activeTab" class="result-tabs">
          <el-tab-pane label="Schema Tree" name="tree">
            <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
            <div v-else class="result-panel tree-panel">
              <el-tree
                :data="treeData"
                node-key="key"
                :expand-on-click-node="false"
                default-expand-all
              />
            </div>
          </el-tab-pane>

          <el-tab-pane label="Examples" name="examples">
            <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
            <div v-else class="result-panel">
              <el-table :data="examplesRows" size="small" style="width: 100%" stripe>
                <el-table-column prop="table" label="Table" width="160" />
                <el-table-column prop="column" label="Column" width="180" />
                <el-table-column prop="type" label="Type" width="110" />
                <el-table-column prop="pk" label="PK" width="80">
                  <template #default="{ row }">
                    <el-tag v-if="row.pk" type="success" effect="light">PK</el-tag>
                    <span v-else>-</span>
                  </template>
                </el-table-column>
                <el-table-column prop="examples" label="Examples" min-width="240" show-overflow-tooltip />
              </el-table>

              <el-alert
                v-if="examplesRows.length === 0"
                class="mt16"
                type="warning"
                show-icon
                :closable="false"
                title="当前后端 schema_json 没有返回 examples/type/pk，因此这里可能为空或仅部分展示。"
              />
            </div>
          </el-tab-pane>

          <el-tab-pane label="schema_text" name="text">
            <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
            <div v-else class="result-panel">
              <el-input type="textarea" :rows="15" :model-value="schemaText" readonly />
              <div class="panel-actions mt16">
                <el-button @click="copy(schemaText)">复制</el-button>
                <el-button type="primary" plain @click="downloadText">下载 .txt</el-button>
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </el-card>

      <!-- 对齐预览 -->
      <el-card class="panel-card" shadow="never">
        <template #header>
          <div class="panel-header">
            <div>
              <div class="panel-title">4. 对齐预览</div>
              <div class="panel-subtitle">基于问题生成更贴近任务的模式文本</div>
            </div>
            <el-tag effect="plain">POST /schema/enrich</el-tag>
          </div>
        </template>

        <div class="align-form">
          <div class="field span-3">
            <div class="label">问题（question）</div>
            <el-input
              v-model="alignQuestion"
              placeholder="例如：近30天订单金额 Top10 的客户是谁？"
            />
          </div>

          <div class="field span-2">
            <div class="label">证据（evidence，可选）</div>
            <el-input
              v-model="alignEvidence"
              placeholder="例如：日期范围、字段含义说明等"
            />
          </div>

          <div class="field small-field">
            <div class="label">k</div>
            <el-input-number v-model="kSamples" :min="1" :max="5" />
          </div>

          <div class="field span-6">
            <el-button
              type="primary"
              :disabled="!dbId || !alignQuestion.trim()"
              :loading="aligning"
              @click="alignPreview"
            >
              生成对齐预览
            </el-button>
          </div>
        </div>

        <div class="result-panel mt16">
          <div class="section-title">对齐结果</div>
          <el-input
            type="textarea"
            :rows="10"
            :model-value="alignedText"
            readonly
            placeholder="这里显示对齐后的 schema 文本（后端返回）"
          />
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  apiSchemaUpload,
  apiSchemaParseStart,
  apiSchemaGet,
  apiSchemaAlignPreview,
} from '@/api/schema'

const router = useRouter()

// Step 1
const dbName = ref('') // 这里作为 db_id（可选）
const file = ref(null)
const uploading = ref(false)
const dbId = ref('')
const dbPath = ref('')

// Step 2
const kExamples = ref(2) // 后端暂时不接这个字段，页面保留占位
const startingTask = ref(false)
const taskStatus = ref('IDLE') // IDLE | RUNNING | SUCCESS | FAILED
const taskProgress = ref(0)
const taskLogs = ref([])
const taskError = ref('')

// Step 3
const loadingSchema = ref(false)
const schemaData = ref(null)
const activeTab = ref('tree')

// Align preview
const alignQuestion = ref('')
const alignEvidence = ref('')
const kSamples = ref(2)
const aligning = ref(false)
const alignedText = ref('')

function onFileChange(uploadFile) {
  file.value = uploadFile.raw
}
function onFileRemove() {
  file.value = null
}

function goDashboard() {
  router.push('/dashboard')
}

/** ✅ 从后端 schema_text 解析 type/pk/examples
 * 解析形如：
 * (aid:INT, Primary Key, Examples: [NULL, NULL]),
 * (name:TEXT, Examples: [Alice, Bob])
 */
function parseSchemaText(schemaText) {
  const tableMap = {} // table -> col -> { type, is_pk, examples: [] }
  const lines = String(schemaText || '').split('\n')
  let curTable = null

  for (const raw of lines) {
    const line = raw.trim()

    // # Table: xxx
    if (line.startsWith('# Table: ')) {
      curTable = line.slice('# Table: '.length).trim()
      if (!tableMap[curTable]) tableMap[curTable] = {}
      continue
    }

    // (col:TYPE, ... Examples: [..]) 末尾可能有 ), 或 ),,
    if (curTable && line.startsWith('(') && line.includes('Examples:')) {
      // 尽量宽松匹配
      const m = line.match(/^\(([^:]+):([^,]+),(.+?)Examples:\s*\[([^\]]*)\]/i)
      if (!m) continue

      const col = (m[1] || '').trim()
      const type = (m[2] || '').trim()
      const rest = (m[3] || '')
      const is_pk = /primary key/i.test(rest)

      const rawVals = (m[4] || '').trim()
      const examples = rawVals
        ? rawVals.split(',').map(s => s.trim()).filter(Boolean)
        : []

      tableMap[curTable][col] = { type, is_pk, examples }
    }
  }

  return tableMap
}

async function upload() {
  if (!file.value) return
  uploading.value = true
  try {
    const res = await apiSchemaUpload(file.value, dbName.value)
    dbId.value = res.db_id
    dbPath.value = res.db_path || ''
    ElMessage.success('上传成功')
  } catch (e) {
    ElMessage.error(e?.message || '上传失败')
  } finally {
    uploading.value = false
  }
}

const taskRunning = computed(() => taskStatus.value === 'RUNNING')

const taskStatusLabel = computed(() => {
  if (taskStatus.value === 'IDLE') return '未开始'
  if (taskStatus.value === 'RUNNING') return '解析中'
  if (taskStatus.value === 'SUCCESS') return '完成'
  if (taskStatus.value === 'FAILED') return '失败'
  return '未知'
})

const statusTagType = computed(() => {
  if (taskStatus.value === 'IDLE') return 'info'
  if (taskStatus.value === 'RUNNING') return 'warning'
  if (taskStatus.value === 'SUCCESS') return 'success'
  if (taskStatus.value === 'FAILED') return 'danger'
  return 'info'
})

const taskLogsText = computed(() => (taskLogs.value || []).join('\n'))

async function startParse() {
  if (!dbId.value) return
  startingTask.value = true
  taskError.value = ''
  taskLogs.value = []
  taskProgress.value = 0
  taskStatus.value = 'RUNNING'

  try {
    // 你的后端是同步 parse：这里我们当作“触发并完成”
    await apiSchemaParseStart(dbId.value, kExamples.value)
    taskProgress.value = 100
    taskStatus.value = 'SUCCESS'
    taskLogs.value = ['Parse started...', 'Backend parsing synchronously...', 'Done.']
    ElMessage.success('解析完成（同步）')
  } catch (e) {
    taskStatus.value = 'FAILED'
    taskError.value = e?.message || '解析失败'
  } finally {
    startingTask.value = false
  }
}

async function loadSchema() {
  if (!dbId.value) return
  loadingSchema.value = true
  try {
    const res = await apiSchemaGet(dbId.value)

    // 后端：{ db_id, schema_dict, schema_text }
    const schemaDict = res.schema_dict || res.schema || {}
    const schemaTextRaw = res.schema_text || res.database_text || ''

    // 从 schema_text 解析出 type/pk/examples
    const parsed = parseSchemaText(schemaTextRaw)

    // schema_dict: { tables:[], columns:{t:[c...]}, foreign_keys:[] }
    const tables = Array.isArray(schemaDict.tables) ? schemaDict.tables : []
    const columnsMap = schemaDict.columns || {}
    const fks = Array.isArray(schemaDict.foreign_keys) ? schemaDict.foreign_keys : []

    const tablesStructured = tables.map((t) => ({
      name: t,
      columns: (Array.isArray(columnsMap[t]) ? columnsMap[t] : []).map((c) => {
        const info = parsed?.[t]?.[c] || {}
        return {
          name: c,
          type: info.type || '-',             // ✅ 这里现在有 Type 了
          is_pk: !!info.is_pk,                // ✅ 这里现在有 PK 了
          examples: info.examples || [],      // ✅ 这里现在有 Examples 了
        }
      }),
    }))

    schemaData.value = {
      db_id: res.db_id || dbId.value,
      schema_dict: schemaDict,
      schema_text: schemaTextRaw,
      schema: {
        tables: tablesStructured,
        foreign_keys: fks,
      },
    }

    ElMessage.success('已拉取解析结果')
  } catch (e) {
    ElMessage.error(e?.message || '拉取失败')
  } finally {
    loadingSchema.value = false
  }
}

/** ✅ 用后端返回的 schema_text 直接展示 */
const schemaText = computed(() => schemaData.value?.schema_text || '')

/** ✅ Tree 数据：只依赖 schemaData.schema（永远是数组，不会 iterable error） */
const treeData = computed(() => {
  const s = schemaData.value?.schema
  const safeTables = Array.isArray(s?.tables) ? s.tables : []
  const safeFks = Array.isArray(s?.foreign_keys) ? s.foreign_keys : []

  return [
    { label: `DB: ${schemaData.value?.db_id || '-'}`, key: `db:${schemaData.value?.db_id || '-'}`, children: [] },
    {
      label: 'Tables',
      key: 'tables',
      children: safeTables.map((t) => ({
        label: `Table: ${t.name}`,
        key: `t:${t.name}`,
        children: (Array.isArray(t.columns) ? t.columns : []).map((c) => ({
          label: `${c.name} : ${c.type}${c.is_pk ? ' (PK)' : ''} examples=${JSON.stringify(c.examples || [])}`,
          key: `c:${t.name}.${c.name}`,
        })),
      })),
    },
    {
      label: 'Foreign Keys',
      key: 'fks',
      children: safeFks.length ? safeFks.map((fk) => ({ label: fk, key: `fk:${fk}` })) : [{ label: '(none)', key: 'fks:none' }],
    },
  ]
})

/** ✅ Examples 表格：展示 type/pk/examples */
const examplesRows = computed(() => {
  const s = schemaData.value?.schema
  const safeTables = Array.isArray(s?.tables) ? s.tables : []
  const out = []

  for (const t of safeTables) {
    const cols = Array.isArray(t.columns) ? t.columns : []
    for (const c of cols) {
      out.push({
        table: t.name,
        column: c.name,
        type: c.type || '-',
        pk: !!c.is_pk,
        examples: JSON.stringify(c.examples || []),
      })
    }
  }
  return out
})

/** ✅ alignPreview 必须存在：否则模板会 warn */
async function alignPreview() {
  if (!dbId.value || !alignQuestion.value.trim()) return
  aligning.value = true
  try {
    // 你 api/schema.js 会把 /schema/enrich 结果转成 aligned_database_text
    const res = await apiSchemaAlignPreview({
      db_id: dbId.value,
      question: alignQuestion.value,
      evidence: alignEvidence.value,
      k_samples: kSamples.value,
    })

    alignedText.value = res.aligned_database_text || ''
    ElMessage.success('已生成对齐预览')
  } catch (e) {
    ElMessage.error(e?.message || '对齐预览失败')
  } finally {
    aligning.value = false
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

function downloadText() {
  const text = schemaText.value || ''
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${schemaData.value?.db_id || 'schema'}.txt`
  a.click()
  URL.revokeObjectURL(url)
}
</script>


<style scoped>
.schema-page {
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
  grid-template-columns: minmax(0, 1.2fr) 340px;
  gap: 16px;
  margin-bottom: 16px;
  align-items: start;
}

.main-col,
.side-col {
  display: flex;
  flex-direction: column;
  gap: 16px;
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

.form-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 14px;
  align-items: end;
}

.align-form {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 14px;
  align-items: end;
}

.span-2 {
  grid-column: span 2;
}

.span-3 {
  grid-column: span 3;
}

.span-6 {
  grid-column: span 6;
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

.small-field {
  max-width: 180px;
}

.action-field {
  justify-content: flex-end;
}

.parse-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  padding: 14px 16px;
  border: 1px solid #edf2f7;
  border-radius: 14px;
  background: #f9fbff;
}

.parse-status {
  display: flex;
  align-items: center;
  gap: 18px;
  flex: 1;
  min-width: 0;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 10px;
  white-space: nowrap;
}

.status-label {
  font-size: 12px;
  color: #6b7280;
  font-weight: 600;
}

.status-progress {
  flex: 1;
  min-width: 220px;
}

.parse-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.log-box {
  margin-top: 16px;
}

.section-title {
  margin-bottom: 10px;
  font-size: 13px;
  font-weight: 700;
  color: #374151;
}

.summary-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
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

.break {
  max-width: 170px;
}

.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.full-btn {
  width: 100%;
  justify-content: center;
}

.result-card {
  margin-bottom: 16px;
}

.result-tabs {
  margin-top: -2px;
}

.result-panel {
  border: 1px solid #edf2f7;
  border-radius: 14px;
  padding: 14px;
  background: #fcfdff;
}

.tree-panel {
  min-height: 320px;
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

:deep(.el-tabs__item) {
  font-weight: 600;
}

:deep(.el-button + .el-button) {
  margin-left: 0;
}

@media (max-width: 1180px) {
  .top-grid {
    grid-template-columns: 1fr;
  }

  .form-grid,
  .align-form {
    grid-template-columns: repeat(2, 1fr);
  }

  .span-2,
  .span-3 {
    grid-column: span 2;
  }

  .span-6 {
    grid-column: span 2;
  }

  .parse-toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .parse-status {
    flex-direction: column;
    align-items: stretch;
  }
}

@media (max-width: 768px) {
  .schema-page {
    padding: 14px;
  }

  .page-head {
    flex-direction: column;
    align-items: flex-start;
  }

  .page-title {
    font-size: 22px;
  }

  .form-grid,
  .align-form {
    grid-template-columns: 1fr;
  }

  .span-2,
  .span-3,
  .span-6 {
    grid-column: span 1;
  }

  .panel-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .summary-item {
    flex-direction: column;
  }

  .break {
    max-width: 100%;
  }
}
</style>
