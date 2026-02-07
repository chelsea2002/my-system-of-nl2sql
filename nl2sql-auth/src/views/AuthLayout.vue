<template>
  <div class="page">
    <div class="left">
      <div class="brand">
        <div class="logo">NL2SQL</div>
        <div class="title">数据库智能查询与分析系统</div>
        <div class="desc">
          集成 模式剪枝、SchemaLinker、SAR、POSG，实现自然语言到 SQL 的智能生成与分析。
        </div>
      </div>

      <div class="feature">
        <div class="item">• 用户权限管理</div>
        <div class="item">• 模式解析与对齐</div>
        <div class="item">• 知识库索引构建（SAR）</div>
        <div class="item">• Text-to-SQL 生成与可视化</div>
        <div class="item">• 查询历史与 SQL 日志管理</div>
      </div>
    </div>

    <div class="right">
      <div class="card">
        <router-view />
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  height: 100vh;
  display: flex;
  background: #f5f7fb;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei";
}
.left {
  flex: 1;
  padding: 64px;
  background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
  color: white;
}
.brand .logo {
  width: 72px;
  height: 72px;
  border-radius: 16px;
  background: rgba(255,255,255,0.18);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  letter-spacing: 1px;
  margin-bottom: 18px;
}
.brand .title {
  font-size: 28px;
  font-weight: 800;
  margin-bottom: 12px;
}
.brand .desc {
  opacity: 0.92;
  line-height: 1.7;
  max-width: 520px;
}
.feature {
  margin-top: 36px;
  padding-top: 24px;
  border-top: 1px solid rgba(255,255,255,0.25);
  max-width: 520px;
}
.feature .item {
  margin: 10px 0;
  opacity: 0.95;
}
.right {
  width: 520px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.card {
  width: 100%;
  background: white;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(17,24,39,0.08);
  padding: 28px 28px 18px;
}
</style>
