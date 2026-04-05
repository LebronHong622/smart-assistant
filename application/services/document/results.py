"""
应用层结果对象 - 用于返回给 Interface 层，与 HTTP 响应解耦
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class DocumentUploadResult(BaseModel):
    """文档上传结果"""
    success: bool
    document_id: str
    title: str
    message: str


class RetrievedDocumentResult(BaseModel):
    """检索到的文档结果项"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float


class RetrieveDocumentsResult(BaseModel):
    """检索文档结果"""
    success: bool
    query: str
    result_count: int
    results: List[RetrievedDocumentResult]


class CollectionInfoResult(BaseModel):
    """集合信息结果"""
    success: bool
    collection_info: Dict[str, Any]


class DeleteDocumentResult(BaseModel):
    """删除文档结果"""
    success: bool
    document_id: str
    message: str


class ListCollectionsResult(BaseModel):
    """列出集合结果"""
    success: bool
    collections: List[str]
    count: int


class CreateCollectionResult(BaseModel):
    """创建集合结果"""
    success: bool
    collection_name: str
    message: str


class DeleteCollectionResult(BaseModel):
    """删除集合结果"""
    success: bool
    collection_name: str
    message: str
