import { defineStore } from 'pinia'

const TOKEN_KEY = 'nl2sql_token'
const USER_KEY = 'nl2sql_user'

export const useAuthStore = defineStore('auth', {
  state: () => ({
  token: sessionStorage.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY) || '',
  user: JSON.parse(sessionStorage.getItem(USER_KEY) || localStorage.getItem(USER_KEY) || 'null'),
  }),
  getters: {
    isAuthed: (s) => !!s.token,
  },
  actions: {
    setAuth(token, user, remember = true) {
      this.token = token
      this.user = user
      const storage = remember ? localStorage : sessionStorage

      // 清理两边，避免“记住我”切换时残留
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
      sessionStorage.removeItem(TOKEN_KEY)
      sessionStorage.removeItem(USER_KEY)

      storage.setItem(TOKEN_KEY, token)
      storage.setItem(USER_KEY, JSON.stringify(user))
    },
    restore() {
      // 优先 session，其次 local
      const t = sessionStorage.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY)
      const u = sessionStorage.getItem(USER_KEY) || localStorage.getItem(USER_KEY)
      this.token = t || ''
      this.user = u ? JSON.parse(u) : null
    },
    logout() {
      this.token = ''
      this.user = null
      localStorage.removeItem(TOKEN_KEY)
      localStorage.removeItem(USER_KEY)
      sessionStorage.removeItem(TOKEN_KEY)
      sessionStorage.removeItem(USER_KEY)
    },
  },
})
