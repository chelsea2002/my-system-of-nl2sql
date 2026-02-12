// src/api/nl2sql.js
import http from './http'

/** POST /nl2sql/generate */
export function apiNl2sqlGenerate({ db_id, question, options = {} } = {}) {
  return http.post('/nl2sql/generate', { db_id, question, options })
}

/** POST /nl2sql/execute */
export function apiNl2sqlExecute({ db_id, sql } = {}) {
  return http.post('/nl2sql/execute', { db_id, sql })
}

/** POST /history/feedback */
export function apiHistoryFeedback({
  db_id,
  question,
  selected_sql,
  ok,
  corrected_sql = '',
  gen_ms = 0,
  exec_ms = 0,
  total_ms = 0,
} = {}) {
  return http.post('/history/feedback', {
    db_id,
    question,
    selected_sql,
    ok,
    corrected_sql,
    gen_ms,
    exec_ms,
    total_ms,
  })
}
