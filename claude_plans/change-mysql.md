# 数据库迁移：PostgreSQL → MySQL

## 任务描述

将项目中的 **PostgreSQL** 数据库依赖完全替换为 **MySQL**。需要删除所有PostgreSQL相关代码，并实现MySQL版本。

## 已有代码分析

当前项目中PostgreSQL相关代码分布在以下位置：

### 核心文件
- `infrastructure/persistence/database/postgres_client.py` - PostgreSQL客户端单例实现
- `application/common/components/postgres_component.py` - PostgreSQL组件注册
- `infrastructure/persistence/database/__init__.py` - 导出PostgreSQL客户端
- `infrastructure/persistence/eval/postgres/` - 评估模块的PostgreSQL实现
  - `eval_dataset_repository_impl.py`
  - `eval_result_repository_impl.py`
  - `eval_vector_repository_impl.py`
- `interface/container.py` - 依赖注入容器注册
- `config/settings.py` - PostgreSQL配置定义
- `infrastructure/persistence/database/test/test_postgres_client.py` - 测试文件

## 迁移要求

### 1. 配置层修改
- [ ] 在 `config/settings.py` 中：
  - 删除 `PostgresSettings` 类
  - 添加 `MySQLSettings` 类，支持 URL 配置和分参数配置两种方式
  - 修改 `Settings` 类中的 `postgres: PostgresSettings` 为 `mysql: MySQLSettings`
  - 修改 `preload_components` 默认值从 `["redis", "milvus", "postgres"]` 改为 `["redis", "milvus", "mysql"]`
- [ ] 在 `.env.example` 中：
  - 删除所有PostgreSQL配置项
  - 添加MySQL配置项示例：
    ```env
    # MySQL 数据库配置
    MYSQL_URL=mysql+pymysql://user:password@localhost:3306/database_name
    # 或单独配置
    # MYSQL_HOST=localhost
    # MYSQL_PORT=3306
    # MYSQL_USER=root
    # MYSQL_PASSWORD=your_password
    # MYSQL_DATABASE=database_name
    ```

### 2. 基础设施层修改
- [ ] 删除 `infrastructure/persistence/database/postgres_client.py`
- [ ] 创建 `infrastructure/persistence/database/mysql_client.py`，实现：
  - **单例模式**的 `MySQLClient` 类
  - 使用 `sqlalchemy` 作为ORM框架（与原PostgreSQL实现保持一致）
  - 实现连接池管理
  - 提供 `ping()` 方法检测连接可用性
  - 提供 `get_session()` 方法获取会话
  - 提供 `execute()` 方法执行原生SQL
  - 遵循项目日志规范，使用统一的 `app_logger`
- [ ] 修改 `infrastructure/persistence/database/__init__.py`：
  - 删除PostgreSQL相关导出
  - 添加 `MySQLClient` 和 `mysql_client` 单例导出
- [ ] 删除 `infrastructure/persistence/eval/postgres/` 整个目录
  - 在 `infrastructure/persistence/eval/` 下创建 `mysql/` 目录
  - 将三个Repository实现迁移到MySQL：
    - `eval_dataset_repository_impl.py`
    - `eval_result_repository_impl.py`
    - `eval_vector_repository_impl.py`
  - 保持相同的接口实现（因为都依赖domain层的接口）
  - 只改变底层SQL方言

### 3. 应用层修改
- [ ] 删除 `application/common/components/postgres_component.py`
- [ ] 创建 `application/common/components/mysql_component.py`：
  - 实现 `MySQLComponent` 类，继承自 `BaseComponent`
  - 在 `load()` 方法中初始化 `MySQLClient` 并进行连接测试
  - 遵循现有组件模式

### 4. 依赖注入修改
- [ ] 修改 `application/common/components/__init__.py`：
  - 删除PostgreSQL组件导入和注册
  - 添加MySQL组件导入和注册
- [ ] 修改 `interface/container.py`：
  - 删除三个PostgreSQL Repository的导入
  - 添加三个MySQL Repository的导入
  - 修改注册绑定，将原来绑定到PostgreSQL实现改为绑定到MySQL实现

### 5. 依赖管理
- [ ] 修改 `pyproject.toml`：
  - 删除 `psycopg2-binary` 或 `psycopg` 依赖
  - 添加 `pymysql` 依赖
- [ ] 运行 `uv sync` 更新 `uv.lock`（这一步需要用户执行）

### 6. 测试代码清理
- [ ] 删除 `infrastructure/persistence/database/test/test_postgres_client.py`
- [ ] 可选：创建 `test_mysql_client.py` （如果有测试需求）

### 7. 代码规范检查
- [ ] 严格遵守项目DDD架构规范：
  - ✅ 分层依赖：interface → application → domain ← infrastructure
  - ✅ 依赖抽象：依赖domain层接口，不直接依赖具体实现
  - ✅ 职责分离：业务逻辑保持不变，只改变底层实现
- [ ] 保持单例模式设计（与原PostgreSQL、Redis、Milvus保持一致）
- [ ] 遵循项目日志规范，使用统一的 `app_logger`
- [ ] 删除所有未使用的import

## 约束条件

1. **最小改动**：只修改与PostgreSQL→MySQL迁移相关的代码，保持其他功能不变
2. **保持架构**：保持现有的DDD分层架构和单例模式设计不变
3. **保持接口**：保持domain层定义的Repository接口不变，只更改infrastructure层实现
4. **完全清理**：确保删除所有PostgreSQL相关代码、配置和依赖，不留残余
5. **依赖处理**：确保不引入不必要的新依赖，只添加MySQL连接所需的最小依赖

## 验证检查清单

完成迁移后，请确认：

- [ ] 项目能够正常启动，没有import错误
- [ ] 所有现有测试能够正常运行
- [ ] 配置文件中没有PostgreSQL残留配置
- [ ] `pyproject.toml` 中没有PostgreSQL依赖
- [ ] 符合项目开发规范（DDD分层、日志规范等）
