// src/api/auth.js
import http from './http'

// 你后端暂时没做 auth，所以先继续 mock
const USE_MOCK = true

const mockDB = {
  users: [
    { id: 1, username: 'admin', contact: 'admin@example.com', password: 'Admin12345', role: 'admin' },
  ],
}

function mockDelay(ms = 500) {
  return new Promise((r) => setTimeout(r, ms))
}

export async function apiLogin({ account, password }) {
  if (!USE_MOCK) return http.post('/auth/login', { account, password })

  await mockDelay()
  const user = mockDB.users.find((u) => u.username === account || u.contact === account) || null

  if (!user) {
    const e = new Error('账号不存在')
    e.response = { status: 400, data: { message: '账号不存在' } }
    throw e
  }
  if (user.password !== password) {
    const e = new Error('密码错误')
    e.response = { status: 400, data: { message: '密码错误' } }
    throw e
  }

  return {
    token: `mock-token-${user.id}-${Date.now()}`,
    user: { id: user.id, username: user.username, role: user.role },
    expiresIn: 3600,
  }
}

export async function apiRegister({ username, contact, password }) {
  if (!USE_MOCK) return http.post('/auth/register', { username, contact, password })

  await mockDelay()
  const exists = mockDB.users.some((u) => u.username === username || u.contact === contact)
  if (exists) {
    const e = new Error('用户名或联系方式已存在')
    e.response = { status: 400, data: { message: '用户名或联系方式已存在' } }
    throw e
  }

  const id = mockDB.users.length + 1
  mockDB.users.push({ id, username, contact, password, role: 'user' })

  return {
    token: `mock-token-${id}-${Date.now()}`,
    user: { id, username, role: 'user' },
    expiresIn: 3600,
  }
}

export async function apiMe() {
  if (!USE_MOCK) return http.get('/auth/me')
  await mockDelay(200)
  return { ok: true }
}
