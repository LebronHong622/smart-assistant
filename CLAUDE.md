```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

## 项目概述

这是一个基于 LangChain + LangGraph 构建的**多任务问答助手**，支持智能对话、工具调用（如高德地图天气查询）和会话管理。项目采用**领域驱动设计 (DDD)** 架构，代码结构清晰，易于扩展。

## 开发环境与命令

### 依赖管理

项目使用 `uv` 进行依赖管理（已弃用 Python 版本限制，使用 `uv.lock` 锁定依赖）：

```bash
# 安装依赖（同步 uv.lock）
uv sync

# 运行项目（CLI 模式）
uv run python -m interface.cli.main

# 运行 API 服务（FastAPI）
uv run python -m interface.api.main

# 运行测试
uv run pytest interface/test/test.py -v

# 运行 Redis 功能测试
uv run pytest infrastructure/cache/test/test_redis_client.py -v
```

### 项目入口

- **CLI 界面**: `interface/cli/main.py` - 命令行交互
- **API 服务**: `interface/api/main.py` - FastAPI 接口
- **核心代理**: `application/agent/qa_agent.py` - 创建和配置 QA 代理
- **文档管理**: `interface/cli/document_manager.py` - 文档管理命令行工具

## 代码架构与结构

项目采用 DDD 架构，分为四层：

### 1. Domain Layer（领域层）
**位置**: `domain/`
- 定义核心业务实体和值对象
- 包含领域服务接口
- **关键文件**:
  - `domain/qa/` - 问答领域（实体、仓库、服务、值对象）
  - `domain/document/` - 文档检索领域（新增）
    - 实体: 文档和文档集合
    - 仓库: 文档操作接口
    - 服务: 文档管理和检索服务
    - 值对象: 文档元数据和检索结果

### 2. Application Layer（应用层）
**位置**: `application/`
- 协调领域层和基础设施层
- 提供业务用例实现
- **核心**:
  - `agent/qa_agent.py` - 创建和管理智能代理实例
  - `document/document_retrieval_service_impl.py` - 文档检索服务 Milvus 实现
- 工具调用链实现: 使用 LangGraph 构建的对话流程

### 3. Infrastructure Layer（基础设施层）
**位置**: `infrastructure/`
- 提供技术实现细节
- **关键模块**:
  - `config/settings.py` - 环境变量和配置管理（包含 Milvus 和 DashScope 配置）
  - `log/` - Loguru 日志配置
  - `memory/` - 对话历史管理（memory_manager.py, middle_ware.py）
  - `model/` - 模型管理
    - `model_manager.py` - 模型管理器
    - `openai_model_config.py` - OpenAI 兼容模型配置
    - `embeddings_manager.py` - Embeddings 模型管理（新增）
  - `prompt/` - 提示词模板管理（YAML 配置）
  - `tool/` - 外部工具集成
    - `amap_weather_query.py` - 高德地图天气查询
    - `document_retrieval.py` - 文档检索工具（新增）
  - `cache/` - Redis 连接功能模块
    - `redis_client.py` - Redis 客户端实现（单例模式）
    - `redis_saver.py` - LangGraph RedisCheckpointSaver 集成
    - `storage_factory.py` - 存储后端工厂
    - `test/test_redis_client.py` - Redis 功能测试
  - `vector/` - 向量存储实现（新增）
    - `milvus_client.py` - Milvus 客户端封装
    - `vector_store.py` - 向量存储仓库实现

### 4. Interface Layer（接口层）
**位置**: `interface/`
- 对外提供服务接口
- **模块**:
  - `api/` - FastAPI 接口
    - `main.py` - 主入口（包含文档管理路由）
    - `document_routes.py` - 文档管理路由（新增）
  - `cli/` - 命令行界面
    - `main.py` - QA 代理交互
    - `document_manager.py` - 文档管理工具（新增）
  - `test/` - 测试模块

## 核心功能实现

### 智能代理创建

```python
from application.agent import create_qa_agent

# 创建代理（支持会话ID）
agent = create_qa_agent(session_id="user_123")
response = agent.chat("北京天气怎么样？")
```

### 工具调用机制

工具在 `infrastructure/tool/` 中实现，需：
1. 在 `tool_manager.py` 中注册
2. 符合 LangChain `@tool` 装饰器规范
3. 支持的工具：
   - 高德地图天气查询 (`amap_weather_query.py`)
   - 文档检索 (`document_retrieval.py`)

### 文档检索功能

```python
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService

# 初始化文档检索服务
retrieval_service = MilvusDocumentRetrievalService()

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
```

### 内存管理策略

支持三种对话历史管理方式（配置于 `settings.py`）：
- `trim`: 裁剪旧消息
- `summary`: 摘要压缩历史
- `delete`: 删除旧消息

### 存储后端

支持两种会话状态存储方式（配置于 `settings.py`）：
- `in_memory`: 内存存储（默认），服务重启后会话状态会丢失
- `redis`: Redis 持久化存储，支持分布式部署和会话共享

### 向量存储（Milvus）

使用 Milvus 向量数据库存储文档嵌入向量，支持语义搜索功能。

## 配置与环境变量

**文件**: `.env`
```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
AMAP_API_KEY=your_amap_key
MODEL=deepseek-chat
LOG_LEVEL=INFO

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

## 提示词模板

**位置**: `infrastructure/prompt/`
- YAML 格式配置
- 支持热加载
- 模板变量支持动态替换

## 常见开发任务

1. **添加新工具**: 在 `infrastructure/tool/` 中创建新文件，在 `tool_manager.py` 中注册
2. **修改代理行为**: 调整 `application/agent/qa_agent.py` 中的 LangGraph 链
3. **更新提示词**: 编辑 `infrastructure/prompt/*.yaml` 文件
4. **添加新模型**: 配置 `infrastructure/model/openai_model_config.py`
5. **使用 Redis 存储**:
   - 确保 Redis 服务正在运行（本地或 Docker）
   - 配置 `.env` 文件：`STORAGE_BACKEND=redis`
   - 配置 Redis 连接参数（`REDIS_URL` 或 `REDIS_HOST`/`REDIS_PORT`）
   - 重新启动应用程序
6. **使用 Milvus 存储**:
   - 确保 Milvus 服务正在运行（本地或 Docker）
   - 配置 `.env` 文件：添加 Milvus 连接参数
   - 配置 DashScope API 密钥（用于 Embeddings）
   - 重新启动应用程序

## 日志规范

### 日志系统概述

项目使用 **Loguru** 作为日志库，实现了单例模式的 LoggerManager 和统一的日志配置。

#### 架构设计
- **位置**: `/workspace/infrastructure/log/log.py`
- **架构**: 单例模式 (LoggerManager)
- **日志库**: Loguru
- **设计优势**: 确保整个应用使用单一日志实例，提高资源利用率和一致性

#### 输出配置
- **控制台输出**: 彩色格式化输出，支持实时调试
- **文件输出**:
  - `app_{time:YYYY-MM-DD}.log`: 应用程序日志
  - `error_{time:YYYY-MM-DD}.log`: 错误日志
  - `api_{time:YYYY-MM-DD}.log`: API 请求日志
- **轮转策略**: 每日自动轮转
- **保留期**: 普通日志 7 天，错误日志 14 天
- **压缩**: 旧日志自动压缩为 ZIP 格式

#### 日志格式
```
{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}
```

### 统一日志使用规范

#### 强制规范
- **所有操作都应该打日志**：任何函数或方法的入口、出口、关键操作和错误都应该记录日志
- **统一导入方式**：必须使用项目提供的统一 logger 实例
- **错误日志必填**：所有异常都应该记录 ERROR 级别日志

#### 导入方式规范

**正确方式**：
```python
from infrastructure.log import app_logger
app_logger.info("信息日志")
```

**错误方式**：
```python
# 禁止使用 Python 内置 logging 模块
import logging
logger = logging.getLogger(__name__)

# 禁止直接使用 loguru 模块
from loguru import logger
```

#### 日志级别说明

| 级别       | 用途                     | 示例场景                     |
|------------|--------------------------|------------------------------|
| **DEBUG**  | 调试信息，开发时使用     | 变量值、函数参数、内部状态   |
| **INFO**   | 常规信息，跟踪应用程序流程 | 函数入口/出口、重要操作完成   |
| **WARNING**| 警告信息，潜在问题       | 参数验证失败、资源即将耗尽   |
| **ERROR**  | 错误信息，需要关注的问题 | 异常捕获、API 调用失败       |
| **CRITICAL**| 严重错误，系统故障       | 数据库连接失败、系统崩溃     |

#### 日志使用场景

1. **函数入口/出口日志**
   ```python
   def my_function(param1, param2):
       app_logger.info(f"调用 my_function，参数: param1={param1}, param2={param2}")
       try:
           # 函数逻辑
           result = do_something()
           app_logger.info(f"my_function 执行成功，返回: {result}")
           return result
       except Exception as e:
           app_logger.error(f"my_function 执行失败: {str(e)}")
           raise
   ```

2. **重要操作记录**
   ```python
   app_logger.info("正在初始化数据库连接")
   app_logger.debug(f"连接参数: host={host}, port={port}")
   ```

3. **错误捕获和记录**
   ```python
   try:
       # 可能出错的操作
       response = requests.get(url, timeout=10)
       response.raise_for_status()
   except requests.exceptions.RequestException as e:
       app_logger.error(f"HTTP 请求失败: {str(e)}")
       raise
   ```

4. **性能监控**
   ```python
   import time
   start_time = time.time()
   # 执行耗时操作
   end_time = time.time()
   app_logger.info(f"操作耗时: {end_time - start_time:.2f} 秒")
   ```

5. **用户行为跟踪**
   ```python
   app_logger.info(f"用户 {user_id} 执行了 {action} 操作")
   ```

#### 最佳实践

1. **日志消息应该有意义和上下文**
   ```python
   # 好的
   app_logger.info(f"成功处理了 {len(items)} 个项目")

   # 不好的
   app_logger.info("处理完成")
   ```

2. **避免过度记录和重复日志**
   - 不要在循环中记录 DEBUG 级别的日志
   - 避免在同一函数中重复记录相同信息的日志

3. **错误日志应该包含足够的调试信息**
   ```python
   # 好的
   app_logger.error(f"查询数据库失败: SQL={sql}, 错误信息={str(e)}")

   # 不好的
   app_logger.error("查询数据库失败")
   ```

4. **敏感信息不应该记录在日志中**
   ```python
   # 禁止记录密码、API 密钥等敏感信息
   app_logger.debug(f"用户密码: {password}")  # 错误！
   ```

5. **使用结构化日志格式**
   ```python
   # 使用字典格式便于日志分析
   app_logger.info("用户登录", extra={"user_id": user_id, "ip": ip_address})
   ```

### 日志配置与环境变量

**文件**: `.env`
```env
# 日志级别配置（可选值: DEBUG/INFO/WARNING/ERROR/CRITICAL）
LOG_LEVEL=INFO
```

#### 配置说明
- 支持通过环境变量 `LOG_LEVEL` 配置全局日志级别
- 默认日志级别为 INFO
- 可以在运行时动态调整（需重启应用）

### 技术栈

- LangChain / LangGraph: LLM 应用框架
- DeepSeek API: 大语言模型服务
- 高德地图 API: 天气数据
- FastAPI: API 服务
- Pydantic: 数据验证
- Loguru: 日志管理
- Redis: 会话状态存储（可选）
- Milvus: 向量数据库（新增）
- DashScope: 阿里千文 Embeddings（新增）

## Milvus 功能文档

### 架构设计

Milvus 功能模块采用**单例模式**设计，确保整个应用使用单一 Milvus 连接实例，提高资源利用率。

### 配置方式

支持两种配置方式：
1. **URL 配置**: 使用 `MILVUS_URI` 配置完整连接地址（如 `http://localhost:19530`）
2. **分参数配置**: 分别配置 `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION_NAME` 等参数

### 功能模块

1. **MilvusClient**: 单例客户端实现，负责管理 Milvus 连接
2. **MilvusVectorStore**: 实现向量存储仓库接口，用于存储和检索文档
3. **DocumentRetrievalService**: 文档检索服务，使用 Milvus 和 DashScope Embeddings
4. **工具模块**: `document_retrieval.py` 提供文档检索工具

### 使用 Milvus 存储的优势

1. **语义检索**: 支持基于向量相似度的文档检索
2. **高性能**: 支持大规模向量检索（百万级数据秒级响应）
3. **多模态支持**: 可扩展支持文本、图片、音频等多种模态的向量化
4. **持久化存储**: 文档向量不会因服务重启而丢失
5. **高可用性**: 支持 Milvus 主从复制和集群部署

### 风险与注意事项

1. **Milvus 服务可用性**: 应用程序依赖 Milvus 服务的可用性
2. **数据一致性**: 需要考虑 Milvus 持久化策略
3. **连接管理**: 需要合理配置 Milvus 连接参数
4. **网络延迟**: 需要考虑 Milvus 服务器与应用服务器的网络延迟
5. **向量维度**: 确保 Embeddings 模型的维度与 Milvus 配置一致

## Redis 功能文档

### 架构设计

Redis 功能模块采用**单例模式**设计，确保整个应用使用单一 Redis 连接实例，提高资源利用率。

### 配置方式

支持两种配置方式：
1. **URL 配置**: 使用 `REDIS_URL` 配置完整连接地址（如 `redis://localhost:6379/0`）
2. **分参数配置**: 分别配置 `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD` 等参数

### 功能模块

1. **RedisClient**: 单例客户端实现，负责管理 Redis 连接
2. **RedisCheckpointSaver**: 实现 LangGraph 的 CheckpointSaver 接口，用于存储会话状态
3. **StorageFactory**: 存储后端工厂，支持 `in_memory` 和 `redis` 两种存储方式
4. **测试模块**: `test_redis_client.py` 包含完整的功能测试和集成测试

### 使用 Redis 存储的优势

1. **持久化存储**: 会话状态不会因服务重启而丢失
2. **分布式支持**: 多个应用实例可以共享会话状态
3. **内存管理**: 释放应用服务器内存压力
4. **高可用性**: 支持 Redis 主从复制和哨兵模式

### 风险与注意事项

1. **Redis 服务可用性**: 应用程序依赖 Redis 服务的可用性
2. **数据一致性**: 需要考虑 Redis 持久化策略（RDB/AOF）
3. **连接管理**: 需要合理配置 Redis 连接池参数
4. **网络延迟**: 需要考虑 Redis 服务器与应用服务器的网络延迟

## 验证与测试

### Redis 功能测试

```bash
# 运行 Redis 功能测试
uv run pytest infrastructure/cache/test/test_redis_client.py -v

# 运行集成测试（需要实际 Redis 服务）
uv run pytest infrastructure/cache/test/test_redis_client.py -v -m "not skip"
```

### 功能验证步骤

1. 确保 Redis 服务正在运行（本地或 Docker）
2. 配置 `.env` 文件：`STORAGE_BACKEND=redis`
3. 运行应用程序并创建会话
4. 重启应用程序并验证会话状态是否保持

### Milvus 功能测试

```bash
# 运行文档检索服务测试（需要实际 Milvus 服务）
uv run pytest application/document/test/test_document_retrieval.py -v
```

### 文档检索功能验证

1. 确保 Milvus 服务正在运行（本地或 Docker）
2. 配置 `.env` 文件中的 Milvus 和 DashScope 连接参数
3. 使用 `document_manager.py` 命令行工具上传文档
4. 使用 `retrieve` 命令检索文档并验证结果

## 文档管理命令

### 上传文档

```bash
uv run python -m interface.cli.document_manager upload -t "文档标题" -c "文档内容" -p txt -s upload
```

### 检索文档

```bash
uv run python -m interface.cli.document_manager retrieve -q "查询文本" -l 3 -s 0.8
```

### 获取集合信息

```bash
uv run python -m interface.cli.document_manager info
```

### 删除文档

```bash
uv run python -m interface.cli.document_manager delete -i "document_id"
```

## API 接口

### 文档管理接口

- **上传文档**: `POST /documents/upload`
- **检索文档**: `POST /documents/retrieve`
- **获取集合信息**: `GET /documents/info`
- **删除文档**: `DELETE /documents/{document_id}`

### API 使用示例

```bash
# 上传文档
curl -X POST "http://localhost:8000/documents/upload" -H "Content-Type: application/json" -d '{
    "content": "文档内容",
    "title": "文档标题",
    "document_type": "txt",
    "source": "upload"
}'

# 检索文档
curl -X POST "http://localhost:8000/documents/retrieve" -H "Content-Type: application/json" -d '{
    "query": "查询文本",
    "limit": 5,
    "score_threshold": 0.5
}'

# 获取集合信息
curl -X GET "http://localhost:8000/documents/info"

# 删除文档
curl -X DELETE "http://localhost:8000/documents/document_id"
```
