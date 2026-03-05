# 多任务问答助手 (Multitask QA Assistant)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.7+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 LangChain 构建的智能问答助手，支持多任务处理、工具调用和会话管理

[功能特性](#功能特性) • [快速开始](#快速开始) • [使用示例](#使用示例) • [配置说明](#配置说明) • [项目结构](#项目结构)

</div>

---

## 功能特性

- **智能问答** - 基于 DeepSeek 大语言模型的智能对话能力
- **工具调用** - 支持集成外部工具扩展能力
  - 🌤️ 高德地图天气查询 - 实时获取城市天气信息
- **会话管理** - 支持多会话隔离，每个会话独立维护上下文
- **内存管理** - 灵活的对话历史管理策略
  - `trim` - 裁剪旧消息
  - `summary` - 摘要压缩历史
  - `delete` - 删除旧消息
- **存储后端** - 支持多种会话状态存储方式
  - `in_memory` - 内存存储（默认）
  - `redis` - Redis 持久化存储
- **提示词模板** - YAML 配置的提示词模板，支持热加载
- **模块化设计** - 清晰的模块划分，易于扩展

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- DeepSeek API 密钥
- 高德地图 API 密钥（用于天气查询功能）
- Redis 服务（可选，如需使用 Redis 存储）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/your-username/multitask-qa-assistant.git
cd multitask-qa-assistant
```

2. **安装依赖**

```bash
# 使用 pip
pip install -e .

# 或使用 uv
uv sync
```

3. **配置环境变量**

创建 `.env` 文件：

```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
MODEL=deepseek-chat
TEMPERATURE=0.7
MAX_TOKENS=1024

# 高德地图 API 配置
AMAP_API_KEY=your_amap_api_key
AMAP_API_URL=https://restapi.amap.com/v3/weather/weatherInfo

# 应用配置
LOG_LEVEL=INFO
MAX_SESSION_HISTORY=2
MAX_TOKENS_BEFORE_SUMMARY=4000
OVERFLOW_MEMORY_METHOD=summary

# 存储后端配置
STORAGE_BACKEND=in_memory  # 可选值: in_memory/redis

# Redis 连接配置（当 STORAGE_BACKEND=redis 时需要配置）
REDIS_URL=redis://localhost:6379/0
# 或单独配置
# REDIS_HOST=localhost
# REDIS_PORT=6379
# REDIS_DB=0
# REDIS_PASSWORD=your_redis_password
# REDIS_SOCKET_TIMEOUT=5
# REDIS_SOCKET_CONNECT_TIMEOUT=5
# REDIS_RETRY_ON_TIMEOUT=True
# REDIS_MAX_CONNECTIONS=10
```

4. **运行程序**

```bash
python main.py
```

## 使用示例

### 基本对话

```python
from agent.qa_agent import create_qa_agent

# 创建问答代理实例
agent = create_qa_agent()

# 发送查询并获取响应
response = agent.chat("你好，请介绍一下你自己")
print(response)
```

### 天气查询

```python
from agent.qa_agent import create_qa_agent

agent = create_qa_agent()

# 查询天气（代理会自动调用天气查询工具）
response = agent.chat("北京今天天气怎么样？")
print(response)
```

### 会话管理

```python
from agent.qa_agent import create_qa_agent

# 创建带会话ID的代理实例
session_id = "user_123"
agent = create_qa_agent(session_id=session_id)

# 多轮对话 - 上下文会被保持
response1 = agent.chat("我叫小明")
response2 = agent.chat("你还记得我的名字吗？")  # 助手会记住"小明"
```

### 命令行交互

```bash
python main.py
```

```
🤖 多任务问答助手启动中...
✅ 代理初始化完成，开始对话
💡 提示：输入 'exit' 或 'quit' 退出对话
==================================================
👤 用户: 上海今天天气怎么样？
🤖 助手: 正在思考...
🤖 助手: 上海今天天气晴朗，气温25°C...
==================================================
```

## 配置说明

### API 配置

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 必填 |
| `DEEPSEEK_API_BASE` | DeepSeek API 基础 URL | `https://api.deepseek.com/v1` |
| `MODEL` | 使用的模型名称 | `deepseek-chat` |
| `TEMPERATURE` | 模型温度参数 | `0.7` |
| `MAX_TOKENS` | 模型最大令牌数 | `1024` |
| `AMAP_API_KEY` | 高德地图 API 密钥 | 必填 |
| `AMAP_API_URL` | 高德地图天气 API URL | `https://restapi.amap.com/v3/weather/weatherInfo` |

### 应用配置

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `MAX_SESSION_HISTORY` | 最大会话历史长度 | `2` |
| `MAX_TOKENS_BEFORE_SUMMARY` | 触发摘要的最大令牌数 | `4000` |
| `OVERFLOW_MEMORY_METHOD` | 溢出内存管理方法 | `summary` |
| `STORAGE_BACKEND` | 会话存储后端 | `in_memory` |
| `REDIS_URL` | Redis 连接 URL（当使用 Redis 存储时） | `redis://localhost:6379/0` |
| `REDIS_HOST` | Redis 主机地址（当使用 Redis 存储时） | `localhost` |
| `REDIS_PORT` | Redis 端口（当使用 Redis 存储时） | `6379` |
| `REDIS_DB` | Redis 数据库索引（当使用 Redis 存储时） | `0` |
| `REDIS_PASSWORD` | Redis 密码（当使用 Redis 存储时） | `None` |
| `REDIS_SOCKET_TIMEOUT` | Redis 连接超时时间（秒） | `5` |
| `REDIS_SOCKET_CONNECT_TIMEOUT` | Redis 连接建立超时时间（秒） | `5` |
| `REDIS_RETRY_ON_TIMEOUT` | 超时是否重试 | `True` |
| `REDIS_MAX_CONNECTIONS` | 最大连接数 | `10` |

## 项目结构

```
multitask-qa-assistant/
├── agent/
│   ├── __init__.py
│   └── qa_agent.py           # 核心问答代理模块
├── config/
│   ├── __init__.py
│   ├── log.py                # 日志配置
│   └── settings.py           # 项目配置管理
├── enums/
│   └── enums.py              # 枚举定义
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py     # 内存管理器
│   └── middle_ware.py        # 内存中间件
├── models/
│   ├── __init__.py
│   ├── model_manager.py      # 模型管理器
│   └── openai_model_config.py # OpenAI 兼容模型配置
├── prompts/
│   ├── __init__.py
│   ├── general_qa.yaml       # 提示词模板配置
│   └── prompt_manager.py     # 提示词管理器
├── tools/
│   ├── __init__.py
│   ├── amap_weather_query.py # 高德天气查询工具
│   ├── tool_manager.py       # 工具管理器
│   ├── tool_shema.py         # 工具 Schema 定义
│   └── AMap_adcode_citycode.xlsx # 城市编码映射表
├── test/
│   └── test.py               # 测试文件
├── infrastructure/          # 新增：基础设施层
│   ├── cache/                # 新增：Redis 连接功能模块
│   │   ├── __init__.py
│   │   ├── redis_client.py   # Redis 客户端实现
│   │   ├── redis_saver.py    # LangGraph RedisCheckpointSaver 集成
│   │   ├── storage_factory.py # 存储后端工厂
│   │   └── test/
│   │       └── test_redis_client.py # Redis 功能测试
│   ├── config/               # 现有：配置管理（包含 Redis 配置类）
│   ├── log/                  # 现有：日志管理
│   ├── memory/               # 现有：内存管理（保持不变）
│   ├── model/                # 现有：模型管理
│   ├── prompt/               # 现有：提示词管理
│   └── tool/                 # 现有：工具管理
├── main.py                   # 项目入口
├── pyproject.toml            # 项目依赖配置
└── README.md                 # 项目说明文档
```

## 扩展指南

### 添加新工具

1. 在 `tools/` 目录下创建新的工具模块

```python
# tools/my_tool.py
from langchain.tools import tool

@tool("my_tool", description="工具描述")
def my_tool(param: str) -> str:
    """工具实现"""
    return "result"
```

2. 在 `tools/tool_manager.py` 中注册工具

```python
def init_tools(self) -> list:
    @tool("my_tool", description="工具描述")
    def my_tool(param: str) -> str:
        return exec_my_tool(param)
    return [get_weather, my_tool]
```

### 添加新模型

在 `models/openai_model_config.py` 中添加模型配置：

```python
MODEL_CONFIGS = {
    "deepseek-chat": ChatOpenAI(...),
    "new-model": ChatOpenAI(
        api_key=settings.api.new_api_key,
        base_url="https://api.example.com/v1",
        model="new-model-name",
        temperature=0.7,
        max_tokens=1024
    )
}
```

### 自定义提示词模板

编辑 `prompts/general_qa.yaml` 文件：

```yaml
custom_template:
  name: 自定义模板
  desc: 模板描述
  content: |
    你是一个专业的助手。
    用户问题: {query}
    请回答用户的问题。
```

## 存储后端

### 使用 Redis 存储

1. 确保 Redis 服务正在运行
2. 在 `.env` 文件中设置：
   ```env
   STORAGE_BACKEND=redis
   REDIS_URL=redis://localhost:6379/0
   ```
3. 重新启动应用程序

### 存储后端配置

- **in_memory**（默认）：会话状态存储在内存中，重启后会丢失
- **redis**：会话状态存储在 Redis 中，支持持久化和分布式部署

## 故障排除

### 常见问题

<details>
<summary><b>API 密钥错误</b></summary>

确保 `.env` 文件中的 API 密钥正确设置：
- `DEEPSEEK_API_KEY` - 从 [DeepSeek 平台](https://platform.deepseek.com/) 获取
- `AMAP_API_KEY` - 从 [高德开放平台](https://lbs.amap.com/) 获取

</details>

<details>
<summary><b>网络连接问题</b></summary>

检查网络连接和 API 基础 URL 配置。如果使用代理，请确保代理设置正确。

</details>

<details>
<summary><b>城市天气查询失败</b></summary>

确保城市名称正确，支持中文城市名称（如"北京"、"上海"）。系统会自动匹配城市编码。

</details>

<details>
<summary><b>Redis 连接失败</b></summary>

确保 Redis 服务正在运行，并检查以下配置：
- `STORAGE_BACKEND=redis`
- `REDIS_URL` 或 `REDIS_HOST`/`REDIS_PORT` 配置正确
- Redis 服务是否允许连接（防火墙配置）
- Redis 是否需要密码认证

</details>

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过 GitHub Issues 与我们联系。

---

<div align="center">

如果这个项目对你有帮助，请给一个 ⭐️ Star 支持一下！

</div>
