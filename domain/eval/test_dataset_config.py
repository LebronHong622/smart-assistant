"""
测试数据集生成配置值对象
所有配置从YAML文件加载，支持环境变量占位符替换
"""
from typing import Optional, Dict
from pydantic import BaseModel, Field


class DocumentsConfig(BaseModel):
    """文档加载配置"""
    input_dir: str = Field("data/documents", description="输入文档目录")
    file_pattern: str = Field("*.json", description="文件匹配模式")
    recursive: bool = Field(False, description="是否递归搜索子目录")


class SplitterConfig(BaseModel):
    """文本分割配置"""
    type: str = Field("recursive", description="分割器类型")
    chunk_size: int = Field(1024, description="分块大小")
    chunk_overlap: int = Field(200, description="分块重叠大小")
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ".", " ", ""],
        description="分隔符列表"
    )


class LLMConfig(BaseModel):
    """LLM模型配置"""
    provider: str = Field("deepseek", description="LLM提供商: deepseek|dashscope|openai")
    model_name: str = Field("deepseek-chat", description="模型名称")
    api_key: str = Field(..., description="API密钥")
    base_url: str = Field(..., description="API基础URL")
    temperature: float = Field(0.1, description="采样温度")
    max_tokens: int = Field(1024, description="最大生成token数")


class EmbeddingConfig(BaseModel):
    """Embedding模型配置"""
    provider: str = Field("dashscope", description="Embedding提供商: dashscope|openai")
    model_name: str = Field("text-embedding-v3", description="模型名称")
    api_key: str = Field(..., description="API密钥")
    base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="API基础URL"
    )
    dimension: int = Field(1536, description="向量维度")


class RoleConfig(BaseModel):
    """生成角色配置"""
    enabled: bool = Field(True, description="是否启用此角色")
    questions_per_doc: int = Field(3, description="每个文档生成的问题数量")
    description: str = Field("", description="角色描述，用于提示词")


class KeywordExtractionConfig(BaseModel):
    """关键词提取配置"""
    enabled: bool = Field(True, description="是否启用关键词提取")
    top_n: int = Field(5, description="提取关键词数量")


class SingleHopConfig(BaseModel):
    """单跳生成配置"""
    enabled: bool = Field(True, description="是否启用单跳生成")
    max_questions_per_doc: int = Field(5, description="每个文档最大问题数")
    include_doc_metadata: bool = Field(True, description="是否包含文档元数据")


class OutputConfig(BaseModel):
    """输出配置"""
    format: str = Field("parquet", description="输出格式: parquet|csv|json")
    save_dir: str = Field("data/test_datasets", description="保存目录")


class GenerationConfig(BaseModel):
    """生成配置"""
    roles: Dict[str, RoleConfig] = Field(
        default_factory=dict,
        description="角色配置: concrete/abstract"
    )
    single_hop: SingleHopConfig = Field(
        default_factory=SingleHopConfig,
        description="单跳生成配置"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="输出配置"
    )


class TestDatasetConfig(BaseModel):
    """测试数据集生成完整配置"""
    documents: DocumentsConfig = Field(
        default_factory=DocumentsConfig,
        description="文档加载配置"
    )
    splitter: SplitterConfig = Field(
        default_factory=SplitterConfig,
        description="文本分割配置"
    )
    llm: LLMConfig = Field(..., description="LLM配置")
    embedding: EmbeddingConfig = Field(..., description="Embedding配置")
    keyword_extraction: KeywordExtractionConfig = Field(
        default_factory=KeywordExtractionConfig,
        description="关键词提取配置"
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="生成配置"
    )

    model_config = {"extra": "ignore"}
