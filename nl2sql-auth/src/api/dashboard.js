import http from './http'

/**
 * 首页汇总数据：用户信息、系统状态、数据库概览、最近查询等
 * 后续你把 URL 替换成真实后端即可
 */
export function apiDashboardSummary() {
  // return http.get('/dashboard/summary')
  return Promise.resolve({
    user: { username: 'admin', role: 'admin' },
    system: {
      schemaLinked: true,
      indexingReady: true,
      llmReady: true,
      lastSyncAt: '2026-01-27 10:20:00',
    },
    db: {
      connectedCount: 1,
      currentDbName: 'demo_db',
      tables: 24,
      columns: 312,
    },
    stats: {
      totalQueries: 128,
      todayQueries: 6,
      successRate: 0.93,
      avgLatencyMs: 680,
    },
    recentQueries: [
      { id: 1, question: '近30天订单金额Top10客户？', sql: 'SELECT ...', time: '10:12', ok: true },
      { id: 2, question: '按地区统计活跃用户数', sql: 'SELECT ...', time: '09:40', ok: true },
      { id: 3, question: '找出异常退款订单', sql: 'SELECT ...', time: '昨天', ok: false },
    ],
  })
}

/** 快捷：获取最近查询（可拆分接口） */
export function apiRecentQueries() {
  // return http.get('/queries/recent')
  return Promise.resolve([])
}
