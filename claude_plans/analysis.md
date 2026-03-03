# 智能问答助手项目架构分析报告

## 项目概述

这是一个基于 LangChain 和 DeepSeek API 构建的智能问答助手项目，支持工具调用、会话管理和内存管理功能。项目提供了命令行和 FastAPI 两种交互方式。

## 重构核心目标
- 仅调整现有代码的目录结构/文件归属，不修改任何业务逻辑、技术实现代码；
- 严格遵循DDD四层架构（基础设施层→领域层→应用层→接口层），每层职责边界清晰；
- 保证重构后依赖关系合规（仅上层依赖下层，禁止反向依赖），项目可正常运行；
- 保留原有核心能力，仅调整代码组织形式。

## 当前目录结构分析

```
multitask-qa-assistant/
├── agent/                    # 智能代理模块（问题所在核心）
│   ├── __init__.py
│   └── qa_agent.py         # 核心问答代理，耦合严重
├── config/                   # 配置模块
│   ├── __init__.py
│   ├── log.py              # Loguru 日志管理（单例模式）
│   └── settings.py         # Pydantic 配置管理（单例模式）
├── enums/                    # 枚举模块
│   └── enums.py
├── memory/                   # 内存管理模块
│   ├── __init__.py
│   ├── memory_manager.py    # 基于 LangChain 记忆系统
│   └── middle_ware.py       # 内存处理中间件
├── models/                   # 模型管理模块
│   ├── __init__.py
│   ├── model_manager.py     # 模型管理器
│   └── openai_model_config.py # OpenAI 兼容模型配置
├── prompts/                  # 提示词管理模块
│   ├── __init__.py
│   ├── general_qa.yaml      # 提示词模板配置
│   └── prompt_manager.py    # YAML 配置加载
├── server/                   # FastAPI 服务器模块（API接口）
│   ├── __init__.py
│   ├── handle.py            # API 路由定义
│   └── main.py              # FastAPI 应用入口
├── tools/                    # 工具模块
│   ├── __init__.py
│   ├── amap_weather_query.py # 高德天气查询
│   ├── tool_manager.py      # 工具管理器
│   ├── tool_shema.py        # 工具参数定义
│   └── AMap_adcode_citycode.xlsx
├── test/                     # 测试模块
│   └── test.py
├── main.py                   # 项目入口（命令行模式）
├── pyproject.toml           # 依赖配置
├── uv.lock                   # 依赖锁定文件
└── README.md
```

## 重构目标结构参考（DDD四层）
可以参考下面的目标结构进行重构，原来代码中没有的能力（如cache、rag等），留个空的文件夹就行
project/  # 根目录
├── infrastructure/  # 基础设施层：纯技术能力，无业务逻辑
│   ├── cache/             # Redis缓存
│   ├── rag/               # RAG检索
│   ├── model/             # LLM/规则引擎
│   ├── api/               # 商品API对接
│   ├── utils/             # 通用工具
│   └── protocol/          # 协议封装
├── domain/  # 领域层：核心业务模型 + 规则（从 Agent 抽离）
│   ├── qa/  # 意图领域
├── application/  # 应用层：流程编排（原 Agent / 核心）
│   ├── agent/  # 核心 Agent（仅保留编排逻辑）
│   └── workflow/  # 调度逻辑
├── interface/  # 接口层：对外入口（原 test.py）
└── README.md  # 补充 DDD 分层说明

## 4. 重构核心规则
### 4.1 文件迁移规则
- 基础设施层：仅迁移「纯技术工具」，不包含任何业务规则（如Redis、RAG、LLM、API）；
- 领域层：从原Agent文件中**抽离**业务模型（实体）和核心规则，不新增逻辑；
- 应用层：原Agent文件保留「调用领域层+基础设施层」的编排逻辑，删除内部业务规则；
- 接口层：原测试入口仅调整路径，保留调用应用层的逻辑，不直接依赖底层。

### 4.2 依赖修正规则
- 应用层（application/）仅允许导入 `domain/` 和 `infrastructure/` 的模块；
- 接口层（interface/）仅允许导入 `application/` 的模块；
- 领域层（domain/）仅允许导入 `infrastructure/` 的技术工具，不依赖上层；
- 所有文件的`import`语句仅修正路径，不修改调用逻辑。

### 4.3 验收标准
- 目录结构完全匹配目标结构；
- 所有文件导入路径修正完成，无导入错误；
- 运行`interface/main.py`，核心链路输出结果与重构前一致；
- 无新增/删除业务逻辑代码，仅调整组织形式。

## 5. 规划输出要求
- 输出形式：分步骤重构计划（含操作指令、文件路径、验证方式）；
- 步骤划分：目录创建→文件迁移→领域层抽离→依赖修正→整体验证；
- 每个步骤输出：
  1. 执行指令（如mkdir/移动文件的bash命令）；
  2. 需修改的`import`语句示例；
  3. 该步骤的验证方法；
- 优先保证：核心链路可运行，抽离的领域层代码与原逻辑一致。
