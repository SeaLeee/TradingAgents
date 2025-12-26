# GitHub OAuth 和数据库配置指南

本指南介绍如何为 TradingAgents 配置 GitHub OAuth 登录和数据库存储。

## 目录

1. [GitHub OAuth 配置](#github-oauth-配置)
2. [数据库配置](#数据库配置)
3. [Railway 部署](#railway-部署)
4. [本地开发](#本地开发)

---

## GitHub OAuth 配置

### 步骤 1: 创建 GitHub OAuth App

1. 登录 GitHub，访问 [Settings > Developer settings > OAuth Apps](https://github.com/settings/developers)
2. 点击 **"New OAuth App"**
3. 填写信息：
   - **Application name**: `TradingAgents` (或你喜欢的名称)
   - **Homepage URL**: `https://your-app.railway.app` (生产环境) 或 `http://localhost:8000` (开发环境)
   - **Authorization callback URL**: `https://your-app.railway.app/auth/github/callback` (生产环境) 或 `http://localhost:8000/auth/github/callback` (开发环境)
4. 点击 **"Register application"**
5. 创建后，点击 **"Generate a new client secret"**
6. 复制 **Client ID** 和 **Client Secret**

### 步骤 2: 配置环境变量

添加以下环境变量：

```bash
# GitHub OAuth
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
GITHUB_REDIRECT_URI=https://your-app.railway.app/auth/github/callback

# 可选：保留密码登录
AUTH_USERNAME=admin
AUTH_PASSWORD=your_secure_password
```

---

## 数据库配置

### 方案 1: Railway PostgreSQL (推荐生产环境)

1. 在 Railway 项目中，点击 **"+ New"** > **"Database"** > **"PostgreSQL"**
2. Railway 会自动创建 `DATABASE_URL` 环境变量
3. 应用会自动检测并使用 PostgreSQL

**Railway PostgreSQL 特点：**
- 免费 500MB 存储
- 自动备份
- 与应用在同一平台，延迟低
- 无需额外账号

### 方案 2: SQLite (本地开发)

如果未设置 `DATABASE_URL`，应用会自动使用 SQLite：
- 数据库文件位置: `web/data/tradingagents.db`
- 适合本地开发和测试
- 无需安装额外软件

### 方案 3: 外部 PostgreSQL

你也可以使用其他 PostgreSQL 服务：

**Supabase (免费 500MB):**
```bash
DATABASE_URL=postgresql://user:password@db.xxx.supabase.co:5432/postgres
```

**Neon (免费):**
```bash
DATABASE_URL=postgresql://user:password@xxx.neon.tech/neondb
```

**PlanetScale (MySQL):**
> 注意：本项目使用 PostgreSQL，不支持 MySQL

---

## Railway 部署

### 步骤 1: 添加 PostgreSQL 数据库

1. 打开 Railway 项目
2. 点击 **"+ New"** > **"Database"** > **"PostgreSQL"**
3. 等待数据库创建完成

### 步骤 2: 配置环境变量

在 Railway 项目设置中添加：

```
GITHUB_CLIENT_ID=xxx
GITHUB_CLIENT_SECRET=xxx
GITHUB_REDIRECT_URI=https://your-app.railway.app/auth/github/callback
AUTH_USERNAME=admin
AUTH_PASSWORD=your_password
DEEPSEEK_API_KEY=sk-xxx
```

**重要**: 更新 GitHub OAuth App 的 callback URL 为 Railway 生产地址。

### 步骤 3: 部署

```bash
git add .
git commit -m "Add GitHub OAuth and database support"
git push origin main
```

Railway 会自动部署更新。

---

## 本地开发

### 步骤 1: 创建 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# GitHub OAuth (开发环境)
GITHUB_CLIENT_ID=your_client_id
GITHUB_CLIENT_SECRET=your_client_secret
GITHUB_REDIRECT_URI=http://localhost:8000/auth/github/callback

# 密码登录
AUTH_USERNAME=admin
AUTH_PASSWORD=trading123

# LLM API Keys
DEEPSEEK_API_KEY=sk-xxx

# 测试模式 (跳过登录和使用模拟数据)
TEST_MODE=false
```

### 步骤 2: 安装依赖

```bash
pip install sqlalchemy psycopg2-binary httpx
```

### 步骤 3: 运行应用

```bash
cd web
python -m uvicorn app:app --reload --port 8000
```

### 步骤 4: 测试登录

1. 访问 http://localhost:8000
2. 你可以使用密码登录或 GitHub 登录
3. GitHub 登录需要先在 GitHub 创建 OAuth App

---

## 数据库表结构

### users 表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| github_id | INTEGER | GitHub 用户 ID (唯一) |
| username | VARCHAR(100) | 用户名 |
| email | VARCHAR(255) | 邮箱 |
| avatar_url | VARCHAR(500) | 头像 URL |
| name | VARCHAR(200) | 显示名称 |
| is_active | BOOLEAN | 是否活跃 |
| is_admin | BOOLEAN | 是否管理员 |
| created_at | DATETIME | 创建时间 |
| last_login | DATETIME | 最后登录时间 |

### sessions 表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| user_id | INTEGER | 用户 ID (外键) |
| token | VARCHAR(100) | 会话令牌 |
| ip_address | VARCHAR(50) | 客户端 IP |
| user_agent | VARCHAR(500) | 用户代理 |
| created_at | DATETIME | 创建时间 |
| expires_at | DATETIME | 过期时间 |

### analysis_history 表
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| user_id | INTEGER | 用户 ID (外键) |
| ticker | VARCHAR(20) | 股票代码 |
| analysis_date | VARCHAR(20) | 分析日期 |
| llm_provider | VARCHAR(50) | LLM 提供商 |
| decision | TEXT | 决策 (英文) |
| decision_cn | TEXT | 决策 (中文) |
| reports | TEXT | 报告 (JSON) |
| reports_cn | TEXT | 报告 (中文) |
| created_at | DATETIME | 创建时间 |

---

## 常见问题

### Q: GitHub OAuth 回调失败？
A: 检查 `GITHUB_REDIRECT_URI` 是否与 GitHub OAuth App 设置的 callback URL 一致。

### Q: 数据库连接失败？
A: 确保 `DATABASE_URL` 格式正确。Railway PostgreSQL 会自动设置此变量。

### Q: 如何迁移数据？
A: 数据库表会自动创建。如需迁移旧数据，可以编写迁移脚本。

### Q: 如何只使用密码登录？
A: 不设置 `GITHUB_CLIENT_ID` 和 `GITHUB_CLIENT_SECRET`，页面只会显示密码登录表单。

---

## 安全建议

1. **生产环境务必更改默认密码**
2. **不要将敏感信息提交到 Git**
3. **定期轮换 GitHub Client Secret**
4. **使用 HTTPS** (Railway 自动提供)
5. **定期清理过期会话**
