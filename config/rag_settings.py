"""
RAG 配置加载器
从 YAML 文件加载 RAG 相关配置
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


def _replace_env_vars(value: str) -> str:
    """替换字符串中的环境变量 ${VAR:-default}"""
    if "${" not in value:
        return value

    def replace_env(match):
        content = match.group(1)
        if ":-" in content:
            var_name, default = content.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(content, "")

    return re.sub(r'\$\{([^}]+)\}', replace_env, value)


def _replace_env_in_dict(d: Dict) -> Dict:
    """递归替换字典中的环境变量"""
    result = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = _replace_env_vars(v)
        elif isinstance(v, dict):
            result[k] = _replace_env_in_dict(v)
        else:
            result[k] = v
    return result


# ==================== 向量数据库配置 ====================

class BM25AnalyzerParams(BaseModel):
    """BM25 分词器参数配置"""
    type: str = Field(default="chinese", description="分词器类型: chinese, english, standard 等")
    tokenizer: Optional[str] = Field(default=None, description="自定义分词器")
    filter: Optional[List[str]] = Field(default=None, description="过滤器列表")


class BM25MultiAnalyzerParams(BaseModel):
    """BM25 多语言分词器参数配置"""
    analyzers: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "chinese": {"type": "chinese"},
            "english": {"type": "english"},
            "default": {"tokenizer": "icu"}
        },
        description="多语言分词器配置"
    )
    by_field: str = Field(default="language", description="用于选择分词器的字段名")
    alias: Optional[Dict[str, str]] = Field(default=None, description="分词器别名映射")


class BM25FunctionConfig(BaseModel):
    """BM25 内置函数配置"""
    enabled: bool = Field(default=True, description="是否启用 BM25 函数")
    input_field_names: str = Field(default="content", description="输入字段名（原始文本字段）")
    output_field_names: str = Field(default="sparse_embedding", description="输出字段名（稀疏向量字段）")
    enable_match: bool = Field(default=False, description="是否启用文本匹配")
    function_name: Optional[str] = Field(default=None, description="函数名称，None 时自动生成")
    analyzer_params: Optional[BM25AnalyzerParams] = Field(default=None, description="单语言分词器参数")
    multi_analyzer_params: Optional[BM25MultiAnalyzerParams] = Field(default=None, description="多语言分词器参数")


class LangchainMilvusConfig(BaseModel):
    """LangChain Milvus 配置"""
    auto_id: bool = True
    vector_field: str = "embedding"
    text_field: str = "content"
    metadata_field: str = "metadata"
    # 新增: 混合检索配置
    enable_hybrid_search: bool = False
    # 稠密向量字段名
    dense_vector_field: str = "embedding"
    # 稀疏向量字段名
    sparse_vector_field: str = "sparse_embedding"
    # BM25 函数配置
    bm25_function: BM25FunctionConfig = Field(default_factory=BM25FunctionConfig)


class MilvusConfig(BaseModel):
    """Milvus 配置"""
    enabled: bool = True
    connection: Dict[str, str] = Field(default_factory=lambda: {"uri": "http://localhost:19530"})
    default_dimension: int = 1536
    metric_type: str = "L2"
    index_type: str = "IVF_FLAT"
    n_list: int = 1024
    implementation: str = "langchain_milvus"
    # 新增: 启用动态字段
    enable_dynamic_field: bool = True
    langchain_config: LangchainMilvusConfig = Field(default_factory=LangchainMilvusConfig)

    def get_connection_uri(self) -> str:
        """获取连接 URI，支持环境变量替换"""
        uri = self.connection.get("uri", "http://localhost:19530")
        return _replace_env_vars(uri)


class ChromaConfig(BaseModel):
    """Chroma 配置"""
    enabled: bool = False
    persist_directory: str = "data/chroma"
    settings: Dict[str, Any] = Field(default_factory=dict)


class FAISSConfig(BaseModel):
    """FAISS 配置"""
    enabled: bool = False
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    index_path: str = "data/faiss/index.faiss"


class QdrantConnectionConfig(BaseModel):
    """Qdrant 连接配置"""
    url: str = "http://localhost:6333"
    api_key: str = ""


class QdrantConfig(BaseModel):
    """Qdrant 配置"""
    enabled: bool = False
    connection: QdrantConnectionConfig = Field(default_factory=QdrantConnectionConfig)
    distance: str = "Cosine"

    def get_url(self) -> str:
        """获取 URL，支持环境变量替换"""
        return _replace_env_vars(self.connection.url)


class VectorConfig(BaseModel):
    """向量数据库统一配置"""
    provider: Literal["milvus", "chroma", "faiss", "qdrant"] = "milvus"
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)

    def get_active_config(self) -> Union[MilvusConfig, ChromaConfig, FAISSConfig, QdrantConfig]:
        """获取当前激活的向量数据库配置"""
        configs = {
            "milvus": self.milvus,
            "chroma": self.chroma,
            "faiss": self.faiss,
            "qdrant": self.qdrant,
        }
        return configs[self.provider]

    @classmethod
    def from_yaml_dict(cls, data: Dict[str, Any]) -> "VectorConfig":
        """从 YAML 字典创建配置，自动处理环境变量"""
        data = _replace_env_in_dict(data)
        return cls(
            provider=data.get("provider", "milvus"),
            milvus=MilvusConfig(**data.get("milvus", {})),
            chroma=ChromaConfig(**data.get("chroma", {})),
            faiss=FAISSConfig(**data.get("faiss", {})),
            qdrant=QdrantConfig(**data.get("qdrant", {})),
        )


class EmbeddingModelConfig(BaseModel):
    """嵌入模型配置"""
    enabled: bool = True
    model: str = "text-embedding-v3"
    dimension: int = 1536
    batch_size: int = 25


class RAGPipelineConfig(BaseModel):
    """RAG 流程配置"""
    vector_provider: str = "milvus"  # 向量数据库提供者
    default_embedding_model: str = "dashscope"
    collection_prefix: str = "doc_"
    auto_create_collection: bool = True
    batch_size: int = 100
    enable_multimodal: bool = True


class DefaultsConfig(BaseModel):
    """默认配置"""
    loader: str = "pdf"
    splitter: str = "recursive"


class LoaderItemConfig(BaseModel):
    """加载器单项配置"""
    enabled: bool = True
    class_name: str = ""
    module: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class SplitterItemConfig(BaseModel):
    """分块器单项配置"""
    enabled: bool = True
    class_name: str = ""
    module: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)


class RAGSettings:
    """
    RAG 配置管理类
    单例模式，从 YAML 文件加载配置
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._config_dir = Path(__file__).parent
            self._rag_config = self._load_yaml("rag_config.yaml")
            self._loader_config = self._load_yaml("loader_config.yaml")
            self._splitter_config = self._load_yaml("splitter_config.yaml")
            self._vector_config_data = self._load_yaml("vector_config.yaml")
            
            # 解析配置
            self.rag_pipeline = RAGPipelineConfig(**self._rag_config.get("rag_pipeline", {}))
            self.vector = VectorConfig.from_yaml_dict(self._vector_config_data)
            self.defaults = DefaultsConfig(**self._rag_config.get("defaults", {}))
            
            # 解析嵌入模型配置
            self.embedding_models: Dict[str, EmbeddingModelConfig] = {}
            for name, config in self._rag_config.get("embedding_models", {}).items():
                self.embedding_models[name] = EmbeddingModelConfig(**config)
            
            # 解析加载器配置
            self.loaders: Dict[str, LoaderItemConfig] = {}
            for name, config in self._loader_config.get("loaders", {}).items():
                self.loaders[name] = LoaderItemConfig(
                    enabled=config.get("enabled", True),
                    class_name=config.get("class", ""),
                    module=config.get("module", ""),
                    config=config.get("config", {})
                )
            self.default_loader = self._loader_config.get("default_loader", "pdf")
            
            # 解析分块器配置
            self.splitters: Dict[str, SplitterItemConfig] = {}
            for name, config in self._splitter_config.get("splitters", {}).items():
                self.splitters[name] = SplitterItemConfig(
                    enabled=config.get("enabled", True),
                    class_name=config.get("class", ""),
                    module=config.get("module", ""),
                    config=config.get("config", {})
                )
            self.default_splitter = self._splitter_config.get("default_splitter", "recursive")
            
            # 兼容性：milvus 属性指向 vector.milvus
            self.milvus = self.vector.milvus
            
            self._initialized = True

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        filepath = self._config_dir / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_loader_config(self, loader_type: str) -> Optional[LoaderItemConfig]:
        """获取指定类型的加载器配置"""
        return self.loaders.get(loader_type)

    def get_splitter_config(self, splitter_type: str) -> Optional[SplitterItemConfig]:
        """获取指定类型的分块器配置"""
        return self.splitters.get(splitter_type)

    def get_embedding_config(self, model_name: str) -> Optional[EmbeddingModelConfig]:
        """获取指定嵌入模型的配置"""
        return self.embedding_models.get(model_name)

    def get_vector_config(self) -> VectorConfig:
        """获取向量数据库配置"""
        return self.vector

    def get_collection_name(self, domain: str) -> str:
        """根据业务领域获取 collection 名称"""
        return f"{self.rag_pipeline.collection_prefix}{domain}"


# 全局配置单例
rag_settings = RAGSettings()


def get_rag_settings() -> RAGSettings:
    """获取 RAG 配置单例"""
    return rag_settings
