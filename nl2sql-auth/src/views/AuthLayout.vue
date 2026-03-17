<template>
  <div class="page">
    <div class="left">
      <div class="left-inner">
        <div class="top-badge">SchemaRAG&AP-SQL Platform</div>

        <div class="brand">
          <div class="logo">NL2SQL</div>
          <div class="title">数据库智能查询与分析系统</div>
          <div class="desc">
            集成模式剪枝、SchemaLinker、SAR 与 POSG，支持从数据库接入、
            结构解析、自然语言查询到反馈沉淀的完整智能分析流程。
          </div>
        </div>

        <div class="feature-card">
          <div class="feature-title">核心能力</div>

          <div class="feature-list">
            <div class="item">
              <span class="dot"></span>
              <span>用户权限管理与访问控制</span>
            </div>
            <div class="item">
              <span class="dot"></span>
              <span>模式解析与 Schema 对齐</span>
            </div>
            <div class="item">
              <span class="dot"></span>
              <span>知识库索引构建（SAR）</span>
            </div>
            <div class="item">
              <span class="dot"></span>
              <span>Text-to-SQL 生成与执行分析</span>
            </div>
            <div class="item">
              <span class="dot"></span>
              <span>查询历史、反馈与样本沉淀</span>
            </div>
          </div>
        </div>

        <div class="info-grid">
          <div class="info-card">
            <div class="info-k">适用场景</div>
            <div class="info-v">教学演示 / 智能问数 / 数据分析</div>
          </div>

          <div class="info-card">
            <div class="info-k">系统特点</div>
            <div class="info-v">结构化、可扩展、支持反馈闭环</div>
          </div>
        </div>
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
  min-height: 100vh;
  display: flex;
  background:
    radial-gradient(circle at top left, rgba(59, 130, 246, 0.08), transparent 26%),
    linear-gradient(180deg, #f5f7fb 0%, #eef3f9 100%);
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "PingFang SC", "Microsoft YaHei";
}

.left {
  position: relative;
  flex: 1;
  min-width: 0;
  overflow: auto;
  background:
    radial-gradient(circle at 20% 20%, rgba(255,255,255,0.12), transparent 18%),
    radial-gradient(circle at 80% 30%, rgba(255,255,255,0.10), transparent 20%),
    linear-gradient(135deg, #1e3a8a 0%, #2563eb 52%, #38bdf8 100%);
  color: white;
}

.left::before {
  content: "";
  position: absolute;
  right: -80px;
  top: -80px;
  width: 260px;
  height: 260px;
  border-radius: 50%;
  background: rgba(255,255,255,0.08);
}

.left::after {
  content: "";
  position: absolute;
  left: 40px;
  bottom: 40px;
  width: 180px;
  height: 180px;
  border-radius: 50%;
  background: rgba(255,255,255,0.06);
}

.left-inner {
  position: relative;
  z-index: 1;
  min-height: 100%;
  padding: 48px 48px 36px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 28px;
}

.top-badge {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  height: 32px;
  padding: 0 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.14);
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.4px;
}

.brand {
  margin-top: 8px;
}

.logo {
  width: 76px;
  height: 76px;
  border-radius: 20px;
  background: rgba(255,255,255,0.16);
  border: 1px solid rgba(255,255,255,0.14);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  letter-spacing: 1px;
  font-size: 14px;
  margin-bottom: 22px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
}

.title {
  font-size: 34px;
  font-weight: 800;
  line-height: 1.25;
  max-width: 560px;
}

.desc {
  margin-top: 16px;
  opacity: 0.9;
  line-height: 1.9;
  max-width: 560px;
  font-size: 14px;
}

.feature-card {
  margin-top: 34px;
  max-width: 560px;
  padding: 22px 24px;
  border-radius: 22px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
  backdrop-filter: blur(8px);
}

.feature-title {
  font-size: 15px;
  font-weight: 700;
  margin-bottom: 14px;
  color: rgba(255,255,255,0.96);
}

.feature-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.item {
  display: flex;
  align-items: center;
  gap: 10px;
  color: rgba(255,255,255,0.94);
  line-height: 1.6;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 0 0 5px rgba(255,255,255,0.10);
  flex-shrink: 0;
}

.info-grid {
  margin-top: 28px;
  max-width: 560px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

.info-card {
  padding: 16px 18px;
  border-radius: 18px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.14);
}

.info-k {
  font-size: 12px;
  color: rgba(255,255,255,0.74);
}

.info-v {
  margin-top: 8px;
  font-size: 14px;
  line-height: 1.7;
  font-weight: 600;
  color: #fff;
}

.right {
  width: 540px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 28px;
  box-sizing: border-box;
}

.card {
  width: 100%;
  max-width: 420px;
  background: rgba(255,255,255,0.96);
  border: 1px solid #e8edf5;
  border-radius: 22px;
  box-shadow: 0 16px 40px rgba(17,24,39,0.08);
  padding: 30px 28px 22px;
  backdrop-filter: blur(8px);
}

@media (max-width: 1100px) {
  .left-inner {
    padding: 44px 40px 36px;
  }

  .title {
    font-size: 30px;
  }

  .right {
    width: 500px;
    padding: 20px;
  }
}

@media (max-width: 900px) {
  .page {
    flex-direction: column;
  }

  .left {
    min-height: 300px;
  }

  .left-inner {
    padding: 32px 24px 26px;
  }

  .title {
    font-size: 26px;
    max-width: 100%;
  }

  .desc,
  .feature-card,
  .info-grid {
    max-width: 100%;
  }

  .info-grid {
    grid-template-columns: 1fr;
  }

  .right {
    width: 100%;
    padding: 18px 14px 24px;
  }

  .card {
    max-width: 100%;
    border-radius: 18px;
  }
}

@media (max-width: 640px) {
  .left {
    min-height: auto;
  }

  .left-inner {
    padding: 24px 18px 18px;
  }

  .logo {
    width: 64px;
    height: 64px;
    border-radius: 16px;
    margin-bottom: 18px;
  }

  .title {
    font-size: 22px;
  }

  .desc {
    font-size: 13px;
    line-height: 1.8;
  }

  .feature-card {
    padding: 18px;
    border-radius: 18px;
  }

  .right {
    padding: 12px;
  }

  .card {
    padding: 24px 18px 18px;
  }
}
</style>