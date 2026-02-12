import axios from 'axios'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'

// ✅ 业务后端地址：优先读环境变量，没有就用 localhost:8000
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
})

http.interceptors.request.use((config) => {
  const auth = useAuthStore()
  auth.restore()

  // ✅ 后端目前不校验 token 也没关系；未来加鉴权可直接用
  if (auth.token) {
    config.headers.Authorization = `Bearer ${auth.token}`
  }
  return config
})

http.interceptors.response.use(
  (res) => res.data,
  (err) => {
    const status = err?.response?.status
    const msg = err?.response?.data?.detail || err?.response?.data?.message || err.message || '请求失败'

    // ✅ 401：只有你后端以后加鉴权才会触发
    if (status === 401) {
      const auth = useAuthStore()
      auth.logout()
      ElMessage.error('登录已过期，请重新登录')
      window.location.href = '/auth/login'
      return Promise.reject(err)
    }

    ElMessage.error(msg)
    return Promise.reject(err)
  }
)

export default http
