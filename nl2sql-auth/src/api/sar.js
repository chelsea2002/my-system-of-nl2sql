// src/api/sar.js
import http from './http'

/**
 * 混合模式：
 * - 已实现的后端接口：走 http
 * - 未实现的接口：保留 mock（避免 404）
 */

const __sarStore = window.__sarStore || (window.__sarStore = {
  datasets: {},
  samples: {},
  index: {},
})

function nowStr() {
  const d = new Date()
  const pad = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function hashSample(dbId, question, sql) {
  return `s_${dbId}_${btoa(unescape(encodeURIComponent(question + '|' + sql))).slice(0, 12)}_${Date.now()}`
}

// ✅ 关键：统一前端字段为 sql（兼容后端 examples.json 的 query）
function normalizeSample(x = {}) {
  const sql = x.sql ?? x.query ?? x.rag_sql ?? ''
  return { ...x, sql }
}

/** ✅ 获取系统内可选数据库列表（后端已实现） */
export async function apiDbList() {
  return http.get('/db/list')
}

/** 获取样本库概览（dataset 版本/统计） */
export async function apiSarDatasetSummary(db_id) {
  return http.get('/sar/datasets/summary', { params: { db_id } })
}

/** 分页获取样本列表 */
export async function apiSarSamplesList({ db_id, q = '', page = 1, page_size = 10 }) {
  return http.get('/sar/samples', { params: { db_id, q, page, page_size } })
}

/** 导入 supervised_data 文件 */
export async function apiSarImportFile({ db_id, file, mode = 'merge' }) {
  const fd = new FormData()
  fd.append('db_id', db_id)
  fd.append('mode', mode) // merge / replace
  fd.append('file', file)

  // ✅ 后端接口：POST /sar/datasets/import
  return http.post('/sar/datasets/import', fd)
}

/** 删除样本 */
export async function apiSarSampleDelete({ db_id, sample_id }) {
  return http.delete('/sar/samples', { data: { db_id, sample_id } })
}

/** 触发索引构建 */
export async function apiSarBuildIndex({ db_id, mode = 'incremental' }) {
  return http.post('/sar/index/build', { db_id, mode })
}

/** 获取索引状态 */
export async function apiSarIndexStatus(dbId) {
  return http.get('/sar/index/status', { params: { db_id: dbId } })
}

/** ✅ 追加单条样本（后端已实现） */
export async function apiSarSampleAdd({
  db_id,
  question,
  sql,
  query, // ✅ 兼容 query 传参
  schema = null,
  source = 'history',
  verified = true,
}) {
  const finalSql = sql ?? query ?? ''
  return http.post('/sar/samples/add', { db_id, question, sql: finalSql, schema, source, verified })
}
