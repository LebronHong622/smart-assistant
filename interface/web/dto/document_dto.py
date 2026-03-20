"""
文档管理 API 的 DTO（数据传输对象）
用于 Interface 层与外部系统的数据交换
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ==================== 请求 DTO ====================

class UploadDocumentRequestDTO(BaseModel):
    """上传文档请求 DTO"""
    content: str = Field(..., description="文档内容", min_length=1)
    title: str = Field(..., description="文档标题", min_length=1)
    document_type: str = Field(default="txt", description="文档类型")
    source: str = Field(default="upload", description="文档来源")
    collection_name: Optional[str] = Field(default=None, description="集合名称")
    extra_fields: Optional[Dict[str, Any]] = Field(default=None, description="额外自定义字段")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "这是一个测试文档",
                "title": "测试文档",
                "document_type": "txt",
                "source": "upload",
                "collection_name": "my_collection",
                "extra_fields": {"author": "张三", "category": "技术"}
            }
        }


class RetrieveDocumentsRequestDTO(BaseModel):
    """检索文档请求 DTO"""
    query: str = Field(..., description="检索查询文本", min_length=1)
    limit: int = Field(default=5, description="返回结果数量", ge=1, le=100)
    score_threshold: float = Field(default=0.5, description="相似度阈值", ge=0.0, le=1.0)
    domain: Optional[str] = Field(default=None, description="业务领域（将通过配置映射到集合名称）")
    collection_name: Optional[str] = Field(default=None, description="直接指定集合名称")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "如何使用 Python",
                "limit": 5,
                "score_threshold": 0.6,
                "domain": "after_sales_policy",
                "collection_name": "my_collection"
            }
        }


class CreateCollectionRequestDTO(BaseModel):
    """创建集合请求 DTO"""
    collection_name: str = Field(..., description="集合名称", min_length=1)
    dimension: int = Field(default=1536, description="向量维度")
    metric_type: str = Field(default="COSINE", description="距离度量类型: COSINE/IP/L2")
    description: Optional[str] = Field(default=None, description="集合描述")

    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "product_info",
                "dimension": 1536,
                "metric_type": "COSINE",
                "description": "产品信息集合"
            }
        }


# ==================== 响应 DTO ====================

class UploadDocumentResponseDTO(BaseModel):
    """上传文档响应 DTO"""
    success: bool
    document_id: str
    title: str
    message: str


class RetrieveDocumentsResultDTO(BaseModel):
    """检索结果项 DTO"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float


class RetrieveDocumentsResponseDTO(BaseModel):
    """检索文档响应 DTO"""
    success: bool
    query: str
    result_count: int
    results: List[RetrieveDocumentsResultDTO]


class CollectionInfoResponseDTO(BaseModel):
    """集合信息响应 DTO"""
    success: bool
    collection_info: Dict[str, Any]


class DeleteDocumentResponseDTO(BaseModel):
    """删除文档响应 DTO"""
    success: bool
    document_id: str
    message: str


class ListCollectionsResponseDTO(BaseModel):
    """列出集合响应 DTO"""
    success: bool
    collections: List[str]
    count: int


class CreateCollectionResponseDTO(BaseModel):
    """创建集合响应 DTO"""
    success: bool
    collection_name: str
    message: str


class DeleteCollectionResponseDTO(BaseModel):
    """删除集合响应 DTO"""
    success: bool
    collection_name: str
    message: str
