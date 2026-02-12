// src/api/db.js
import http from './http'

/** 获取数据库列表：GET /db/list */
export function apiDbList() {
  return http.get('/db/list')
}

/** 获取某库的摘要信息：GET /sar/datasets/summary?db_id=xxx */
export function apiDbSummary(db_id) {
  return http.get('/sar/datasets/summary', { params: { db_id } })
}