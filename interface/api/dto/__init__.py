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

__all__ = [
    # 请求 DTO
    "UploadDocumentRequestDTO",
    "RetrieveDocumentsRequestDTO",
    "CreateCollectionRequestDTO",
    # 响应 DTO
    "UploadDocumentResponseDTO",
    "RetrieveDocumentsResponseDTO",
    "RetrieveDocumentsResultDTO",
    "CollectionInfoResponseDTO",
    "DeleteDocumentResponseDTO",
    "ListCollectionsResponseDTO",
    "CreateCollectionResponseDTO",
    "DeleteCollectionResponseDTO"
]
