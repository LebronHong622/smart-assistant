"""
文档检索工具参数 Schema
定义不同实现方式的输入参数类型和详细描述
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentRetrievalInput(BaseModel):
    """
    标准文档检索工具入参定义

    用于从 Milvus 向量数据库中，基于语义相似度检索与查询相关的文档。
    支持多集合隔离，不同业务领域可以使用不同的文档集合。
    """
    query: str = Field(
        ...,
        description="用户查询文本，基于语义进行相似性匹配，支持自然语言查询",
        examples=[
            "Python中如何使用装饰器",
            "Milvus向量数据库的基本使用方法",
            "LangChain工具调用的最佳实践"
        ]
    )
    limit: int = Field(
        5,
        description="返回结果的最大数量限制，数值越大返回结果越多，但可能增加噪声和计算开销",
        examples=[3, 5, 10]
    )
    score_threshold: float = Field(
        0.5,
        description="相似度分数阈值，只有相似度高于该阈值的文档才会被返回。"
                    "值越大，结果越相关，但返回数量可能越少；值越小，返回数量越多，但可能包含不相关结果。",
        examples=[0.3, 0.5, 0.8]
    )
    collection_name: Optional[str] = Field(
        None,
        description="向量集合名称，用于隔离不同的文档知识库。"
                    "如果不指定，使用配置文件中定义的默认集合。",
        examples=["default_collection", "product_docs", "internal_knowledge"]
    )


class EmbeddingRetrievalInput(BaseModel):
    """
    根据嵌入向量检索相似文档入参定义

    使用预计算的嵌入向量进行检索，常用于RAG流程中的多步检索。
    """
    embedding: List[float] = Field(
        ...,
        description="查询嵌入向量，通常由嵌入模型提前计算得到，维度由使用的嵌入模型决定",
    )
    limit: int = Field(
        5,
        description="返回结果的最大数量限制，数值越大返回结果越多，但可能增加噪声和计算开销",
        examples=[3, 5, 10]
    )
    score_threshold: float = Field(
        0.5,
        description="相似度分数阈值，只有相似度高于该阈值的文档才会被返回。"
                    "值越大，结果越相关，但返回数量可能越少；值越小，返回数量越多，但可能包含不相关结果。",
        examples=[0.3, 0.5, 0.8]
    )
    collection_name: Optional[str] = Field(
        None,
        description="向量集合名称，用于隔离不同的文档知识库。"
                    "如果不指定，使用配置文件中定义的默认集合。",
        examples=["default_collection", "product_docs"]
    )


class LangChainDocumentRetrievalInput(BaseModel):
    """
    LangChain 原生文档检索工具入参定义

    基于 LangChain 生态系统，直接使用 LangChainDocumentRepository 进行检索。
    每个集合名称维护单例实例，提高重复查询性能。
    """
    query: str = Field(
        ...,
        description="用户查询文本，基于语义进行相似性匹配，支持自然语言查询",
        examples=[
            "Python异步编程最佳实践",
            "如何在LangChain中使用自定义工具",
            "Milvus混合检索使用说明"
        ]
    )
    collection_name: Optional[str] = Field(
        None,
        description="向量集合名称，用于隔离不同的文档知识库。"
                    "如果不指定，使用配置文件中定义的默认集合。"
                    "每个集合对应一个独立的LangChainDocumentRepository单例实例，避免重复创建。",
        examples=["default_collection", "user_manual", "knowledge_base"]
    )
