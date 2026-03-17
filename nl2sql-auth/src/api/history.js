// src/api/history.js
import http from './http'

/** 分页获取历史列表 */
export function apiHistoryList({ db_id = '', q = '', ok = null, imported = null, page = 1, page_size = 10 } = {}) {
  const params = { page, page_size }
  if (db_id) params.db_id = db_id
  if (q) params.q = q
  if (ok !== null) params.ok = ok
  if (imported !== null) params.imported = imported

  return http.get('/history', { params })
}

/** 将某条历史样本加入 SAR 样本库（后端：POST /history/promote） */
export function apiHistoryPromote({ history_id }) {
  return http.post('/history/promote', { history_id })
}


/** 获取单条历史详情 */
export function apiHistoryDetail(id) {
  return http.get(`/history/${id}`)
}

export function apiHistoryStats({ db_id = '' } = {}) {
  return http.get('/history/stats', { params: { db_id } })
}