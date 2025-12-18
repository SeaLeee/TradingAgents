# TradingAgents Web Deployment

## Railway 部署步骤

### 1. 准备工作

确保你有：
- [Railway](https://railway.app/) 账号
- GitHub 账号（代码已推送到 GitHub）

### 2. 在 Railway 上部署

1. 登录 [Railway](https://railway.app/)
2. 点击 "New Project"
3. 选择 "Deploy from GitHub repo"
4. 选择你的 TradingAgents 仓库
5. Railway 会自动检测并使用 Dockerfile 或 railway.toml 配置

### 3. 配置环境变量

在 Railway 项目设置中添加以下环境变量：

```
# 必需 - 至少配置一个 LLM 提供商的 API Key
GOOGLE_API_KEY=your_google_api_key
# 或
OPENAI_API_KEY=your_openai_api_key
# 或
ANTHROPIC_API_KEY=your_anthropic_api_key
# 或
OPENROUTER_API_KEY=your_openrouter_api_key

# 可选 - 用于获取金融数据
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

### 4. 访问应用

部署完成后，Railway 会提供一个 URL，如：
`https://your-app-name.up.railway.app`

### 5. 本地测试

在部署前，你可以本地测试：

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Web 服务
python -m web.app
```

然后访问 http://localhost:8000

## API 端点

- `GET /` - Web 界面
- `GET /api/health` - 健康检查
- `GET /api/config` - 获取配置选项
- `POST /api/analyze` - 开始分析
- `GET /api/status/{task_id}` - 获取分析状态

## 注意事项

1. **API 配额**: 免费的 LLM API 通常有速率限制，建议使用付费 API 或 OpenRouter 的免费模型
2. **超时**: 分析可能需要几分钟，Railway 默认超时是 300 秒
3. **内存**: 建议至少 512MB RAM
