<template>
  <div class="wrap">
    <div class="top">
      <div>
        <div class="title">模式解析与对齐</div>
        <div class="sub">上传数据库文件（SQLite），解析 Schema，并可对齐生成问题相关样例（预览）。</div>
      </div>
      <div class="top-actions">
        <el-button @click="goDashboard">返回工作台</el-button>
      </div>
    </div>

    <!-- Step 1: 上传 -->
    <el-card>
      <template #header>
        <div class="card-header">
          <span>Step 1：上传数据库</span>
          <el-tag effect="plain">POST /schema/upload</el-tag>
        </div>
      </template>

      <div class="row">
        <div class="field">
          <div class="label">db_id（可选，默认用文件名）</div>
          <el-input v-model="dbName" placeholder="例如：spider_dev / demo_db" style="width: 360px" />
        </div>

        <div class="field">
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

        <div class="field">
          <div class="label">样例数量 k（解析阶段占位）</div>
          <el-input-number v-model="kExamples" :min="1" :max="5" />
        </div>

        <div class="field">
          <div class="label">&nbsp;</div>
          <el-button type="primary" :disabled="!file" :loading="uploading" @click="upload">
            上传
          </el-button>
        </div>
      </div>

      <el-alert
        v-if="dbId"
        type="success"
        show-icon
        :closable="false"
        style="margin-top: 12px"
        :title="`已上传：db_id=${dbId}`"
        :description="dbPath ? `db_path=${dbPath}` : ''"
      />
    </el-card>

    <!-- Step 2: 解析 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>Step 2：解析</span>
          <el-tag effect="plain">POST /schema/parse（同步执行）</el-tag>
        </div>
      </template>

      <div class="row" style="align-items: end">
        <div class="field">
          <div class="label">状态</div>
          <el-tag :type="statusTagType" effect="plain">{{ taskStatusLabel }}</el-tag>
        </div>

        <div class="field">
          <div class="label">进度</div>
          <div style="width: 360px">
            <el-progress :percentage="taskProgress" />
          </div>
        </div>

        <div class="field">
          <div class="label">&nbsp;</div>
          <el-button
            type="primary"
            :disabled="!dbId || taskRunning"
            :loading="startingTask"
            @click="startParse"
          >
            开始解析
          </el-button>

          <el-button :disabled="!dbId" @click="loadSchema" style="margin-left: 8px">
            解析后直接拉取（推荐）
          </el-button>
        </div>
      </div>

      <el-divider />

      <div class="logs">
        <div class="label">日志</div>
        <el-input type="textarea" :rows="6" :model-value="taskLogsText" readonly />
      </div>

      <el-alert
        v-if="taskError"
        type="error"
        show-icon
        :closable="false"
        style="margin-top: 12px"
        :title="taskError"
      />
    </el-card>

    <!-- Step 3: 结果预览 -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>Step 3：解析结果预览</span>
          <div style="display:flex; gap:10px; align-items:center;">
            <el-tag effect="plain">GET /schema/{db_id}/json</el-tag>
            <el-button type="primary" plain :disabled="!dbId" :loading="loadingSchema" @click="loadSchema">
              拉取结果
            </el-button>
          </div>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="结构（Schema Tree）" name="tree">
          <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
          <div v-else class="tree-wrap">
            <el-tree :data="treeData" node-key="key" :expand-on-click-node="false" default-expand-all />
          </div>
        </el-tab-pane>

        <el-tab-pane label="样例预览（Examples）" name="examples">
          <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
          <div v-else>
            <el-table :data="examplesRows" size="small" style="width: 100%">
              <el-table-column prop="table" label="Table" width="160" />
              <el-table-column prop="column" label="Column" width="180" />
              <el-table-column prop="type" label="Type" width="110" />
              <el-table-column prop="pk" label="PK" width="70">
                <template #default="{ row }">
                  <el-tag v-if="row.pk" type="success" effect="plain">PK</el-tag>
                  <span v-else>-</span>
                </template>
              </el-table-column>
              <el-table-column prop="examples" label="Examples" min-width="220" show-overflow-tooltip />
            </el-table>

            <el-alert
              v-if="examplesRows.length === 0"
              type="warning"
              show-icon
              :closable="false"
              style="margin-top: 10px"
              title="当前后端 schema_json 没有返回 examples/type/pk，因此这里可能为空或占位。"
            />
          </div>
        </el-tab-pane>

        <el-tab-pane label="Schema 文本（schema_text）" name="text">
          <el-empty v-if="!schemaData" description="尚未拉取解析结果" />
          <div v-else>
            <el-input type="textarea" :rows="14" :model-value="schemaText" readonly />
            <div style="margin-top: 10px; display:flex; gap:10px;">
              <el-button @click="copy(schemaText)">复制</el-button>
              <el-button type="primary" plain @click="downloadText">下载 .txt</el-button>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <!-- 对齐预览（可选） -->
    <el-card style="margin-top: 14px">
      <template #header>
        <div class="card-header">
          <span>对齐预览（可选）</span>
          <el-tag effect="plain">POST /schema/enrich</el-tag>
        </div>
      </template>

      <div class="row">
        <div class="field" style="flex: 1">
          <div class="label">问题（question）</div>
          <el-input v-model="alignQuestion" placeholder="例如：近30天订单金额Top10客户？" />
        </div>
        <div class="field" style="flex: 1">
          <div class="label">证据（evidence，可选）</div>
          <el-input v-model="alignEvidence" placeholder="例如：限定日期范围、字段含义说明等" />
        </div>
        <div class="field">
          <div class="label">k</div>
          <el-input-number v-model="kSamples" :min="1" :max="5" />
        </div>
        <div class="field">
          <div class="label">&nbsp;</div>
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

      <el-input
        type="textarea"
        :rows="10"
        :model-value="alignedText"
        readonly
        placeholder="这里显示对齐后的 schema 文本（后端返回）"
        style="margin-top: 12px"
      />
    </el-card>
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
.title {
  font-size: 22px;
  font-weight: 900;
  color: #111827;
}
.sub {
  margin-top: 6px;
  color: #6b7280;
}
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

.tree-wrap {
  border: 1px solid #eef2f7;
  border-radius: 12px;
  padding: 10px;
}

.logs { margin-top: 8px; }
</style>
