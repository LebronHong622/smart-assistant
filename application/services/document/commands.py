"""
应用层命令对象 - 用于接收 Interface 层输入，封装用例参数
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class UploadDocumentCommand(BaseModel):
    """上传文档命令"""
    content: str = Field(..., description="文档内容", min_length=1)
    title: str = Field(..., description="文档标题", min_length=1)
    document_type: str = Field(default="txt", description="文档类型")
    source: str = Field(default="upload", description="文档来源")
    collection_name: Optional[str] = Field(default=None, description="集合名称")
    extra_fields: Optional[Dict[str, Any]] = Field(default=None, description="额外自定义字段")


class RetrieveDocumentsCommand(BaseModel):
    """检索文档命令"""
    query: str = Field(..., description="检索查询文本", min_length=1)
    limit: int = Field(default=5, description="返回结果数量", ge=1, le=100)
    score_threshold: float = Field(default=0.5, description="相似度阈值", ge=0.0, le=1.0)
    domain: Optional[str] = Field(default=None, description="业务领域")
    collection_name: Optional[str] = Field(default=None, description="直接指定集合名称")


class CreateCollectionCommand(BaseModel):
    """创建集合命令"""
    collection_name: str = Field(..., description="集合名称", min_length=1)
    dimension: int = Field(default=1536, description="向量维度")
    metric_type: str = Field(default="COSINE", description="距离度量类型")
    description: Optional[str] = Field(default=None, description="集合描述")
