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

## 代码架构与结构

项目采用 DDD 架构，分为四层：

### 1. Domain Layer（领域层）
**位置**: `domain/qa/`
- 定义核心业务实体和值对象
- 包含领域服务接口
- **关键文件**: `domain/qa/entities.py`, `domain/qa/repositories.py`, `domain/qa/services.py`

### 2. Application Layer（应用层）
**位置**: `application/agent/`
- 协调领域层和基础设施层
- 提供业务用例实现
- **核心**: `qa_agent.py` - 创建和管理智能代理实例
- 工具调用链实现: 使用 LangGraph 构建的对话流程

### 3. Infrastructure Layer（基础设施层）
**位置**: `infrastructure/`
- 提供技术实现细节
- **关键模块**:
  - `config/settings.py` - 环境变量和配置管理
  - `log/` - Loguru 日志配置
  - `memory/` - 对话历史管理（memory_manager.py, middle_ware.py）
  - `model/` - 模型管理器（OpenAI 兼容接口）
  - `prompt/` - 提示词模板管理（YAML 配置）
  - `tool/` - 外部工具集成（高德地图天气查询）
  - `cache/` - 新增：Redis 连接功能模块
    - `redis_client.py` - Redis 客户端实现（单例模式）
    - `redis_saver.py` - LangGraph RedisCheckpointSaver 集成
    - `storage_factory.py` - 存储后端工厂
    - `test/test_redis_client.py` - Redis 功能测试

### 4. Interface Layer（接口层）
**位置**: `interface/`
- 对外提供服务接口
- **模块**:
  - `api/` - FastAPI 接口（main.py）
  - `cli/` - 命令行界面（main.py）
  - `test/` - 测试模块（test.py）

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

### 内存管理策略

支持三种对话历史管理方式（配置于 `settings.py`）：
- `trim`: 裁剪旧消息
- `summary`: 摘要压缩历史
- `delete`: 删除旧消息

### 存储后端

支持两种会话状态存储方式（配置于 `settings.py`）：
- `in_memory`: 内存存储（默认），服务重启后会话状态会丢失
- `redis`: Redis 持久化存储，支持分布式部署和会话共享

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

## 技术栈

- LangChain / LangGraph: LLM 应用框架
- DeepSeek API: 大语言模型服务
- 高德地图 API: 天气数据
- FastAPI: API 服务
- Pydantic: 数据验证
- Loguru: 日志管理
- Redis: 会话状态存储（可选）

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
