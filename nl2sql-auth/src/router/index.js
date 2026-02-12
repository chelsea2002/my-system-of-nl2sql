import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

import AuthLayout from '@/views/AuthLayout.vue'
import Login from '@/views/Login.vue'
import Register from '@/views/Register.vue'
import Dashboard from '@/views/Dashboard.vue'
import Schema from '@/views/Schema.vue'
import SARLibrary from '@/views/SARLibrary.vue'
import History from '@/views/History.vue'
import NL2SQL from '@/views/NL2SQL.vue'


const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/dashboard' },

    // Auth
    {
      path: '/auth',
      component: AuthLayout,
      children: [
        { path: 'login', component: Login },
        { path: 'register', component: Register },
      ],
    },

    // App pages
    { path: '/dashboard', component: Dashboard, meta: { requiresAuth: true } },
    { path: '/schema', component: Schema, meta: { requiresAuth: true } },
    { path: '/sar', component: SARLibrary, meta: { requiresAuth: true } },
    { path: '/history', component: History, meta: { requiresAuth: true } },
    { path: '/nl2sql', component: NL2SQL, meta: { requiresAuth: true } },

    // ✅ 404 必须放最后，否则会吞掉后续所有路由
    { path: '/:pathMatch(.*)*', redirect: '/auth/login' },
  ],
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  auth.restore() // ✅ 关键：先恢复

  if (to.meta.requiresAuth && !auth.isAuthed) {
    return '/auth/login'
  }

  if (to.path.startsWith('/auth') && auth.isAuthed) {
    return '/dashboard'
  }

  return true
})

export default router
