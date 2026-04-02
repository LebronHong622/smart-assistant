"""
评估配置模块
统一从config/eval/test_dataset_config.yaml加载，支持环境变量替换
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


def _replace_env_vars(value: Any) -> Any:
    """递归替换字符串中的环境变量占位符 ${VAR_NAME}"""
    if isinstance(value, str):
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replacer, value)
    elif isinstance(value, dict):
        return {k: _replace_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_replace_env_vars(item) for item in value]
    return value


def _replace_env_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """在字典中递归替换所有环境变量"""
    return _replace_env_vars(data)


class DocumentsConfig(BaseModel):
    """文档加载配置"""
    input_dir: str = "data/documents"
    file_pattern: str = "json"
    recursive: bool = False


class SplitterConfig(BaseModel):
    """文档分割配置"""
    type: str = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 200
    separators: list = Field(default_factory=lambda: ["\n\n", "\n", ".", " ", ""])


class LLMConfig(BaseModel):
    """LLM配置"""
    provider: str = "deepseek"
    model_name: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    """Embedding配置"""
    provider: str = "dashscope"
    model_name: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dimension: int = 1536


class RoleConfig(BaseModel):
    """问题生成角色配置"""
    enabled: bool = True
    questions_per_doc: int = 3
    description: str = ""


class TransformConfig(BaseModel):
    """单个Transform配置"""
    enable: bool = True
    class_name: str = Field(alias="class")
    module: str = "ragas.testset.transforms"
    parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class KeywordExtractionConfig(BaseModel):
    """关键词提取配置（已废弃，使用transforms_config替代）"""
    enabled: bool = True
    top_n: int = 5


class SingleHopConfig(BaseModel):
    """单跳生成配置"""
    enabled: bool = True
    max_questions_per_doc: int = 5
    include_doc_metadata: bool = True


class OutputConfig(BaseModel):
    """输出配置"""
    format: str = "parquet"
    save_dir: str = "data/test_datasets"


class GenerationConfig(BaseModel):
    """生成配置"""
    roles: Dict[str, RoleConfig] = Field(default_factory=dict)
    transforms: Dict[str, TransformConfig] = Field(default_factory=dict, alias="transforms_config")
    single_hop: SingleHopConfig = Field(default_factory=SingleHopConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config = {"populate_by_name": True}


class TestDatasetConfig(BaseModel):
    """测试数据集生成配置"""
    documents: DocumentsConfig = Field(default_factory=DocumentsConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    keyword_extraction: Optional[KeywordExtractionConfig] = None


class EvalSettings:
    """
    评估配置单例类
    负责从YAML文件加载并管理所有评估相关配置
    """
    _instance: Optional["EvalSettings"] = None
    _initialized: bool = False
    _config: Optional[TestDatasetConfig] = None
    _config_path: Optional[str] = None

    def __new__(cls, config_path: Optional[str] = None) -> "EvalSettings":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if not self._initialized or config_path != self._config_path:
            self._config_path = config_path or os.getenv(
                "EVAL_CONFIG_PATH",
                str(Path(__file__).parent / "eval" / "test_dataset_config.yaml")
            )
            self._load_config()
            self._initialized = True

    def _load_config(self) -> None:
        """从YAML文件加载配置"""
        path = Path(self._config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 替换环境变量
        data = _replace_env_in_dict(data)

        self._config = TestDatasetConfig(**data)

    @property
    def config(self) -> TestDatasetConfig:
        """获取当前配置对象"""
        if self._config is None:
            raise RuntimeError("配置未加载")
        return self._config

    def reload(self) -> None:
        """重新加载配置"""
        self._load_config()

    @classmethod
    def reload_instance(cls) -> None:
        """重置并重新加载单例实例"""
        if cls._instance is not None:
            cls._instance.reload()


def get_eval_settings() -> EvalSettings:
    """获取评估配置单例实例"""
    return EvalSettings()


# 导出所有配置类
__all__ = [
    "DocumentsConfig",
    "SplitterConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "RoleConfig",
    "TransformConfig",
    "KeywordExtractionConfig",
    "SingleHopConfig",
    "OutputConfig",
    "GenerationConfig",
    "TestDatasetConfig",
    "EvalSettings",
    "get_eval_settings",
]
