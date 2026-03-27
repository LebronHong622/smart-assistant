# 多任务问答助手 (Multitask QA Assistant)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.7+-green.svg)](https://github.com/langchain-ai/langchain)
[![Milvus](https://img.shields.io/badge/Milvus-2.4+-orange.svg)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个基于 LangChain + LangGraph 构建的智能问答助手，支持多任务处理、工具调用和会话管理，集成了 Milvus 向量数据库用于文档检索。

[功能特性](#功能特性) • [快速开始](#快速开始) • [使用示例](#使用示例) • [配置说明](#配置说明) • [项目结构](#项目结构)

</div>

---

## 功能特性

- **智能问答** - 基于 DeepSeek 大语言模型的智能对话能力
- **工具调用** - 支持集成外部工具扩展能力
  - 🌤️ 高德地图天气查询 - 实时获取城市天气信息
  - 📄 文档检索 - 使用 Milvus 向量数据库进行语义搜索
- **会话管理** - 支持多会话隔离，每个会话独立维护上下文
- **内存管理** - 灵活的对话历史管理策略
  - `trim` - 裁剪旧消息
  - `summary` - 摘要压缩历史
  - `delete` - 删除旧消息
- **存储后端** - 支持多种会话状态存储方式
  - `in_memory` - 内存存储（默认）
  - `redis` - Redis 持久化存储
- **向量存储** - 使用 Milvus 向量数据库存储文档嵌入向量
- **语义检索** - 基于阿里千文 text-embedding-v3 的文档相似度检索
- **提示词模板** - YAML 配置的提示词模板，支持热加载
- **模块化设计** - 清晰的模块划分，易于扩展

## 快速开始

### 环境要求

- Python 3.12 或更高版本
- DeepSeek API 密钥
- 高德地图 API 密钥（用于天气查询功能）
- Redis 服务（可选，如需使用 Redis 存储）
- Milvus 向量数据库（Docker 中运行）
- DashScope API 密钥（用于阿里千文 Embeddings）

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

# Milvus 向量数据库配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION_NAME=document_embeddings
MILVUS_DIMENSION=1536
MILVUS_METRIC_TYPE=L2
MILVUS_INDEX_TYPE=IVF_FLAT
MILVUS_N_LIST=1024

# DashScope API 配置（阿里千文 Embeddings）
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v3
```

4. **运行程序**

```bash
# 运行 CLI 界面
uv run python -m interface.cli.main

# 或运行 API 服务
uv run python -m interface.api.main
```

## 使用示例

### 基本对话（Agentic RAG架构）

项目已升级为基于LangGraph的Agentic RAG架构，提供更智能的工具选择和动态工作流编排：

```python
from application.agent import AgenticRagAgent
from interface.container import container

# 创建Agentic RAG代理实例
agent = container.get_agentic_rag_agent()
answer, documents = agent.chat_with_documents("你好，请介绍一下你自己")
print(answer)
```

**新架构特性：**
- 🧠 智能工具选择：根据查询内容自动选择最相关的工具
- 🔄 多轮对话记忆：维护完整的对话上下文
- 📊 动态工作流：根据查询复杂度自适应调整处理流程
- 🎯 自适应检索：智能判断是否需要文档检索及检索策略

### 天气查询（智能工具选择）

```python
from application.agent import AgenticRagAgent
from interface.container import container

agent = container.get_agentic_rag_agent()

# Agentic RAG会自动识别这是天气查询并调用相应工具
answer, documents = agent.chat_with_documents("北京今天天气怎么样？")
print(answer)
# 输出示例："北京今天天气晴朗，气温25°C，东南风3级..."
```

### 文档检索

```python
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService

# 初始化文档检索服务
retrieval_service = MilvusDocumentRetrievalService()

# 上传文档
document = {
    "content": "这是一个示例文档内容，关于 Python 编程的基础概念。",
    "metadata": {
        "title": "Python 编程基础",
        "author": "示例作者",
        "tags": ["Python", "编程", "基础"]
    }
}
retrieval_service.add_document_to_collection(document)

# 检索相关文档
results = retrieval_service.retrieve_similar_documents(
    query="Python 编程基础",
    limit=3,
    score_threshold=0.8
)

# 打印结果
for result in results:
    print(f"文档ID: {result.document_id}")
    print(f"内容预览: {result.content[:100]}...")
    print(f"相似度: {result.similarity_score:.4f}")
    print()
```

### 会话管理（多轮对话记忆）

```python
from application.agent import AgenticRagAgent
from interface.container import container

# 创建带会话ID的Agentic RAG代理实例
session_id = "user_123"
agent = container.get_agentic_rag_agent(session_id=session_id)

# 多轮对话 - Agentic RAG会维护完整的对话上下文
response1 = agent.chat_with_documents("我叫小明")
answer2, docs2 = agent.chat_with_documents("你还记得我的名字吗？")
print(answer2)  # 输出示例："当然记得，你叫小明！"
```

### 命令行交互

```bash
uv run python -m interface.cli.main
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

### 文档管理命令行工具

```bash
# 上传文档
uv run python -m interface.cli.document_manager upload -t "Python 编程基础" -c "这是一个关于 Python 编程的基础教程..." -p txt -s upload

# 检索文档
uv run python -m interface.cli.document_manager retrieve -q "Python 编程基础" -l 3 -s 0.8

# 获取集合信息
uv run python -m interface.cli.document_manager info

# 删除文档
uv run python -m interface.cli.document_manager delete -i "document_id"
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

### Milvus 配置

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `MILVUS_HOST` | Milvus 主机地址 | `localhost` |
| `MILVUS_PORT` | Milvus 端口 | `19530` |
| `MILVUS_URI` | Milvus 连接 URI | `http://localhost:19530` |
| `MILVUS_COLLECTION_NAME` | 集合名称 | `document_embeddings` |
| `MILVUS_DIMENSION` | 向量维度 | `1536` |
| `MILVUS_METRIC_TYPE` | 相似度度量类型 | `L2` |
| `MILVUS_INDEX_TYPE` | 索引类型 | `IVF_FLAT` |
| `MILVUS_N_LIST` | IVF 索引的 n_list 参数 | `1024` |

### DashScope 配置

| 配置项 | 描述 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | DashScope API 密钥 | 必填 |
| `DASHSCOPE_EMBEDDING_MODEL` | 嵌入模型名称 | `text-embedding-v3` |

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
├── domain/                     # 领域层
│   ├── qa/                     # 问答领域
│   │   ├── entity/             # 问答实体
│   │   ├── repository/         # 问答仓库接口
│   │   ├── service/            # 问答领域服务
│   │   └── value_object/       # 问答值对象
│   └── document/               # 文档检索领域（新增）
│       ├── entity/             # 文档和文档集合实体
│       ├── repository/         # 文档操作接口
│       ├── service/            # 文档领域服务
│       └── value_object/       # 文档元数据和检索结果
├── application/                # 应用层
│   ├── agent/                  # QA 代理创建和配置
│   └── document/               # 文档检索服务实现
├── infrastructure/             # 基础设施层
│   ├── config/                 # 配置管理（包含 Milvus 和 DashScope 配置）
│   ├── log/                    # 日志管理
│   ├── memory/                 # 内存管理（保持不变）
│   ├── model/                  # 模型管理
│   │   ├── model_manager.py    # 模型管理器
│   │   ├── openai_model_config.py # OpenAI 兼容模型配置
│   │   └── embeddings_manager.py # Embeddings 模型管理（新增）
│   ├── prompt/                 # 提示词管理
│   ├── tool/                   # 工具管理
│   │   ├── amap_weather_query.py # 高德天气查询工具
│   │   ├── document_retrieval.py # 文档检索工具（新增）
│   │   ├── tool_manager.py     # 工具管理器
│   │   └── tool_shema.py       # 工具 Schema 定义
│   ├── cache/                  # Redis 连接功能模块
│   │   ├── __init__.py
│   │   ├── redis_client.py     # Redis 客户端实现
│   │   ├── redis_saver.py      # LangGraph RedisCheckpointSaver 集成
│   │   ├── storage_factory.py  # 存储后端工厂
│   │   └── test/
│   │       └── test_redis_client.py # Redis 功能测试
│   └── vector/                 # 向量存储实现（新增）
│       ├── milvus_client.py    # Milvus 客户端封装
│       └── vector_store.py     # 向量存储仓库实现
├── interface/                  # 接口层
│   ├── api/                    # FastAPI 接口（新增文档管理路由）
│   ├── cli/                    # 命令行交互
│   │   ├── main.py             # QA 助手主入口
│   │   └── document_manager.py # 文档管理命令行工具（新增）
│   └── test/                   # 测试模块
├── enums/                      # 枚举定义
├── pyproject.toml              # 项目依赖配置
├── README.md                   # 项目说明文档
└── .env                        # 环境变量配置
```

## 扩展指南

### 添加新工具

1. 在 `infrastructure/tool/` 目录下创建新的工具模块

```python
# infrastructure/tool/my_tool.py
from langchain.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具实现"""
    return "result"
```

2. 在 `infrastructure/tool/tool_manager.py` 中注册工具

```python
def init_tools(self) -> list:
    @tool("my_tool", description="工具描述")
    def my_tool(param: str) -> str:
        return exec_my_tool(param)
    return [get_weather, document_retrieval, my_tool]
```

### 添加新模型

在 `infrastructure/model/openai_model_config.py` 中添加模型配置：

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

编辑 `infrastructure/prompt/general_qa.yaml` 文件：

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

### 使用 Milvus 存储

1. 确保 Milvus 向量数据库正在 Docker 中运行
2. 在 `.env` 文件中正确配置 Milvus 连接信息
3. 配置 DashScope API 密钥用于 Embeddings
4. 重新启动应用程序

## 文档检索功能

### 上传文档

```python
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService

retrieval_service = MilvusDocumentRetrievalService()
document = {
    "content": "文档内容",
    "metadata": {
        "title": "文档标题",
        "author": "作者",
        "tags": ["标签1", "标签2"]
    }
}
retrieval_service.add_document_to_collection(document)
```

### 检索文档

```python
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService

retrieval_service = MilvusDocumentRetrievalService()
results = retrieval_service.retrieve_similar_documents(
    query="查询文本",
    limit=5,
    score_threshold=0.5
)

for result in results:
    print(f"文档ID: {result.document_id}")
    print(f"内容预览: {result.content[:100]}...")
    print(f"相似度: {result.similarity_score:.4f}")
```

## 故障排除

### 常见问题

<details>
<summary><b>API 密钥错误</b></summary>

确保 `.env` 文件中的 API 密钥正确设置：
- `DEEPSEEK_API_KEY` - 从 [DeepSeek 平台](https://platform.deepseek.com/) 获取
- `AMAP_API_KEY` - 从 [高德开放平台](https://lbs.amap.com/) 获取
- `DASHSCOPE_API_KEY` - 从 [阿里云 DashScope 控制台](https://dashscope.aliyun.com/) 获取

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

<details>
<summary><b>Milvus 连接失败</b></summary>

确保 Milvus 向量数据库正在 Docker 中运行，并检查以下配置：
- `MILVUS_HOST` 和 `MILVUS_PORT` 配置正确
- Docker 容器是否正在运行
- 网络连接是否正常
- 防火墙配置是否允许访问 Milvus 端口

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
