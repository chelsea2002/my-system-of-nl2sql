import http from './http'

// 把用户输入的 dbName 转成后端需要的 db_id
function normalizeDbId(name) {
  const raw = (name || '').trim()
  if (!raw) return ''
  return raw
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
}

export async function apiSchemaUpload(file, dbName = '') {
  const fd = new FormData()
  fd.append('file', file)

  // ✅ 后端必填：db_id
  const fromName = normalizeDbId(dbName)
  const fromFile = normalizeDbId(file?.name?.replace(/\.(sqlite|db)$/i, '') || '')
  const db_id = fromName || fromFile || `db_${Date.now()}`
  fd.append('db_id', db_id)

  // 可选字段，不影响
  if (dbName) fd.append('db_name', dbName)

  // ✅ 不要手动设置 Content-Type
  return http.post('/schema/upload', fd)
}

/**
 * POST /schema/parse
 * body: { db_id, k_examples? }
 */
export async function apiSchemaParseStart(dbId, kExamples = 2) {
  return http.post('/schema/parse', { db_id: dbId, k_examples: kExamples })
}

/**
 * 如果没有 /tasks 系统，就直接成功（你现在后端看起来是同步 parse）
 */
export async function apiTaskGet(taskId) {
  return Promise.resolve({
    status: 'SUCCESS',
    progress: 100,
    logs: [`Task ${taskId} finished (no task system, mocked SUCCESS)`],
    error: null,
  })
}

/**
 * GET /schema/{db_id}/json
 * 返回：{ db_id, schema_dict, schema_text, ... }
 */
export async function apiSchemaGet(dbId) {
  return http.get(`/schema/${dbId}/json`)
}

/**
 * POST /schema/enrich
 * body: { db_id, question, evidence, k }
 * 返回：{ db_id, schema_text_enriched, sample_text }
 */
export async function apiSchemaAlignPreview(payload) {
  const {
    db_id,
    question,
    evidence = '',
    // ✅ 兼容两种写法：k 或 k_samples
    k,
    k_samples,
  } = payload || {}

  const kk = Number.isFinite(k) ? k : Number.isFinite(k_samples) ? k_samples : 2

  const res = await http.post('/schema/enrich', {
    db_id,
    question,
    evidence,
    k: kk,
  })

  // ✅ 同时返回两套字段名，Schema.vue/旧代码都能用
  return {
    db_id: res.db_id || db_id,
    schema_text_enriched: res.schema_text_enriched || '',
    sample_text: res.sample_text || '',
    aligned_database_text: res.schema_text_enriched || '',
  }
}
