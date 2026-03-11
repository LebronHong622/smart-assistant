"""
DTO 包初始化文件
导出所有 DTO 类
"""

from .document_dto import (
    # 请求 DTO
    UploadDocumentRequestDTO,
    RetrieveDocumentsRequestDTO,
    CreateCollectionRequestDTO,
    # 响应 DTO
    UploadDocumentResponseDTO,
    RetrieveDocumentsResponseDTO,
    RetrieveDocumentsResultDTO,
    CollectionInfoResponseDTO,
    DeleteDocumentResponseDTO,
    ListCollectionsResponseDTO,
    CreateCollectionResponseDTO,
    DeleteCollectionResponseDTO
)

from .agentic_rag_dto import (
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
    ActiveSessionsResponse
)

__all__ = [
    # 文档管理 DTO
    "UploadDocumentRequestDTO",
    "RetrieveDocumentsRequestDTO",
    "CreateCollectionRequestDTO",
    "UploadDocumentResponseDTO",
    "RetrieveDocumentsResponseDTO",
    "RetrieveDocumentsResultDTO",
    "CollectionInfoResponseDTO",
    "DeleteDocumentResponseDTO",
    "ListCollectionsResponseDTO",
    "CreateCollectionResponseDTO",
    "DeleteCollectionResponseDTO",
    # Agentic RAG DTO
    "ChatRequest",
    "ChatResponse",
    "SessionHistoryResponse",
    "ActiveSessionsResponse"
]
