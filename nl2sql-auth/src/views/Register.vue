<template>
  <div>
    <div class="header">
      <div class="h1">注册</div>
      <div class="sub">创建独立的数据库管理空间（账号名 / 密码 / 联系方式）</div>
    </div>

    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-position="top"
      @keyup.enter="onSubmit"
    >
      <el-form-item label="用户名" prop="username">
        <el-input v-model="form.username" placeholder="4~20，字母/数字/下划线" clearable />
      </el-form-item>

      <el-form-item label="联系方式" prop="contact">
        <el-input v-model="form.contact" placeholder="邮箱或手机号" clearable />
      </el-form-item>

      <el-form-item label="密码" prop="password">
        <el-input
          v-model="form.password"
          type="password"
          show-password
          placeholder="8~32"
          clearable
        />
      </el-form-item>

      <el-form-item label="确认密码" prop="confirmPassword">
        <el-input
          v-model="form.confirmPassword"
          type="password"
          show-password
          placeholder="再次输入密码"
          clearable
        />
      </el-form-item>

      <el-form-item prop="agree">
        <el-checkbox v-model="form.agree">
          我已阅读并同意《用户协议》和《隐私政策》
        </el-checkbox>
      </el-form-item>

      <el-button
        type="primary"
        :loading="loading"
        style="width: 100%; margin-top: 6px"
        @click="onSubmit"
      >
        注册并进入系统
      </el-button>

      <div class="footer">
        <span>已有账号？</span>
        <el-link type="primary" :underline="false" @click="goLogin">去登录</el-link>
      </div>
    </el-form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { apiRegister } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()
const formRef = ref()
const loading = ref(false)

const form = reactive({
  username: '',
  contact: '',
  password: '',
  confirmPassword: '',
  agree: false,
})

const isEmail = (v) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v)
const isPhone = (v) => /^1[3-9]\d{9}$/.test(v) // 简化版（中国手机号）

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 4, max: 20, message: '长度 4~20', trigger: 'blur' },
    {
      validator: (_, v, cb) => {
        if (!/^[a-zA-Z0-9_]+$/.test(v)) return cb(new Error('仅支持字母/数字/下划线'))
        if (/^\d+$/.test(v)) return cb(new Error('用户名不能全为数字'))
        cb()
      },
      trigger: 'blur',
    },
  ],
  contact: [
    { required: true, message: '请输入联系方式', trigger: 'blur' },
    {
      validator: (_, v, cb) => {
        if (!isEmail(v) && !isPhone(v)) return cb(new Error('请输入合法邮箱或手机号'))
        cb()
      },
      trigger: 'blur',
    },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, max: 32, message: '长度 8~32', trigger: 'blur' },
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (_, v, cb) => {
        if (v !== form.password) return cb(new Error('两次密码不一致'))
        cb()
      },
      trigger: 'blur',
    },
  ],
  agree: [
    {
      validator: (_, v, cb) => {
        if (!v) return cb(new Error('请先同意协议'))
        cb()
      },
      trigger: 'change',
    },
  ],
}

function goLogin() {
  router.push('/auth/login')
}

async function onSubmit() {
  await formRef.value?.validate(async (ok) => {
    if (!ok) return
    loading.value = true
    try {
      const res = await apiRegister({
        username: form.username,
        contact: form.contact,
        password: form.password,
      })
      auth.setAuth(res.token, res.user, true)
      ElMessage.success('注册成功，已自动登录')
      router.replace('/dashboard')
    } finally {
      loading.value = false
    }
  })
}
</script>

<style scoped>
.header { margin-bottom: 18px; }
.h1 { font-size: 24px; font-weight: 800; }
.sub { margin-top: 6px; color: #6b7280; line-height: 1.6; }
.footer {
  margin-top: 12px;
  display: flex;
  gap: 6px;
  justify-content: center;
  color: #6b7280;
}
</style>
