<template>
  <div>
    <div class="header">
      <div class="h1">登录</div>
      <div class="sub">输入账号与密码获取系统访问权限</div>
    </div>

    <el-form
      ref="formRef"
      :model="form"
      :rules="rules"
      label-position="top"
      @keyup.enter="onSubmit"
    >
      <el-form-item label="账号" prop="account">
        <el-input v-model="form.account" placeholder="用户名 / 邮箱 / 手机号" clearable />
      </el-form-item>

      <el-form-item label="密码" prop="password">
        <el-input
          v-model="form.password"
          type="password"
          show-password
          placeholder="请输入密码"
          clearable
        />
      </el-form-item>

      <div class="row">
        <el-checkbox v-model="form.remember">记住我</el-checkbox>
        <el-link type="primary" :underline="false" @click="onForgot">忘记密码？</el-link>
      </div>

      <el-button
        type="primary"
        :loading="loading"
        style="width: 100%; margin-top: 10px"
        @click="onSubmit"
      >
        登录
      </el-button>

      <div class="footer">
        <span>没有账号？</span>
        <el-link type="primary" :underline="false" @click="goRegister">去注册</el-link>
      </div>
    </el-form>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { apiLogin } from '@/api/auth'
import { useAuthStore } from '@/stores/auth'

const router = useRouter()
const auth = useAuthStore()
const formRef = ref()

const form = reactive({
  account: '',
  password: '',
  remember: true,
})

const rules = {
  account: [
    { required: true, message: '请输入账号', trigger: 'blur' },
    { min: 3, max: 50, message: '长度 3~50', trigger: 'blur' },
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 8, max: 32, message: '长度 8~32', trigger: 'blur' },
  ],
}

const loading = ref(false)

function goRegister() {
  router.push('/auth/register')
}

function onForgot() {
  ElMessage.info('忘记密码功能可后续接入（当前占位）')
}

async function onSubmit() {
  await formRef.value?.validate(async (ok) => {
    if (!ok) return

    loading.value = true
    try {
      const res = await apiLogin({ account: form.account, password: form.password })
      auth.setAuth(res.token, res.user, form.remember)
      ElMessage.success('登录成功')
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
.sub { margin-top: 6px; color: #6b7280; }
.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 6px 0 4px;
}
.footer {
  margin-top: 12px;
  display: flex;
  gap: 6px;
  justify-content: center;
  color: #6b7280;
}
</style>
