"""
应用层：文档应用服务
协调 Interface 层与 Domain 层，负责 DTO → Entity 转换
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from domain.entity.document.document import Document
from domain.vo.document.document_metadata import DocumentMetadata, DocumentType, DocumentSource
from application.services.document.rag_processing_service import RAGProcessingService
from application.services.document.collection_management_service import CollectionManagementService
from application.services.document.document_retrieval_service import DocumentRetrievalAppService

from application.services.document.commands import (
    UploadDocumentCommand,
    RetrieveDocumentsCommand,
    CreateCollectionCommand,
)
from application.services.document.results import (
    DocumentUploadResult,
    RetrieveDocumentsResult,
    RetrievedDocumentResult,
    CollectionInfoResult,
    DeleteDocumentResult,
    ListCollectionsResult,
    CreateCollectionResult,
    DeleteCollectionResult,
)


class DocumentAppService:
    """
    文档应用服务
    负责 Interface DTO → Domain Entity 的转换和用例编排
    """

    def __init__(
        self,
        rag_service: RAGProcessingService,
        collection_service: CollectionManagementService,
        retrieval_service: DocumentRetrievalAppService,
    ):
        self._rag_service = rag_service
        self._collection_service = collection_service
        self._retrieval_service = retrieval_service

    # ========== 文档相关 ==========

    def upload_document(
        self,
        command: UploadDocumentCommand,
    ) -> DocumentUploadResult:
        """
        上传文档用例
        负责：DTO → Entity 转换 → 调用 Domain Service → 返回 Result
        """
        # 创建文档元数据（Application 层负责转换）
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = DocumentMetadata(
            title=command.title,
            document_type=DocumentType[command.document_type.upper()],
            source=DocumentSource[command.source.upper()],
            created_at=current_time,
            updated_at=current_time,
        )

        # 创建文档实体（Application 层负责组装）
        doc_data = {
            "content": command.content,
            "metadata": metadata.model_dump()
        }
        if command.extra_fields:
            doc_data.update(command.extra_fields)

        document = Document(**doc_data)

        # 调用 Domain Service 处理
        processed_document = self._rag_service.process_document(document)

        return DocumentUploadResult(
            success=True,
            document_id=str(processed_document.id),
            title=command.title,
            message="文档上传成功"
        )

    def retrieve_documents(
        self,
        command: RetrieveDocumentsCommand,
    ) -> RetrieveDocumentsResult:
        """
        检索文档用例
        负责：参数准备 → 调用 Domain Service → Entity → Result 转换
        """
        # 调用 Domain Service 检索
        documents = self._rag_service.retrieve_similar(
            query=command.query,
            limit=command.limit,
            score_threshold=command.score_threshold
        )

        # Entity → Result 转换（Application 层负责）
        result_items = [
            RetrievedDocumentResult(
                document_id=str(doc.id),
                content=doc.content,
                metadata=doc.metadata or {},
                similarity_score=doc.similarity_score if doc.similarity_score is not None else 0.0,
                distance=doc.distance if doc.distance is not None else 0.0
            )
            for doc in documents
        ]

        return RetrieveDocumentsResult(
            success=True,
            query=command.query,
            result_count=len(documents),
            results=result_items
        )

    # ========== 集合相关 ==========

    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfoResult:
        """获取集合信息"""
        info = self._get_collection_info_or_create_default(collection_name)
        return CollectionInfoResult(success=True, collection_info=info)

    def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> DeleteDocumentResult:
        """删除文档"""
        self._retrieval_service.remove_document_from_collection(document_id, collection_name=collection_name)
        return DeleteDocumentResult(
            success=True,
            document_id=document_id,
            message="文档删除成功"
        )

    def list_collections(self) -> ListCollectionsResult:
        """列出所有集合"""
        collections = self._collection_service.list_collections()
        collection_names = [col.name for col in collections]
        return ListCollectionsResult(
            success=True,
            collections=collection_names,
            count=len(collection_names)
        )

    def create_collection(self, command: CreateCollectionCommand) -> CreateCollectionResult:
        """创建集合"""
        collection = self._collection_service.create_collection(
            name=command.collection_name,
            description=command.description
        )
        return CreateCollectionResult(
            success=True,
            collection_name=collection.name,
            message="Collection 创建成功"
        )

    def delete_collection(self, collection_name: str) -> DeleteCollectionResult:
        """删除集合"""
        collection = self._collection_service.get_collection_by_name(collection_name)
        if not collection:
            raise ValueError(f"集合不存在: {collection_name}")

        self._collection_service.delete_collection(collection.id)
        return DeleteCollectionResult(
            success=True,
            collection_name=collection_name,
            message="Collection 删除成功"
        )

    def get_specific_collection_info(self, collection_name: str) -> CollectionInfoResult:
        """获取指定集合信息"""
        collection = self._collection_service.get_collection_by_name(collection_name)
        if not collection:
            raise ValueError(f"集合不存在: {collection_name}")

        info = self._collection_service.get_collection_info(collection.id)
        return CollectionInfoResult(success=True, collection_info=info)

    # ========== 内部方法 ==========

    def _get_collection_info_or_create_default(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取集合信息，如果不存在则创建默认集合"""
        if collection_name:
            collection = self._collection_service.get_collection_by_name(collection_name)
            if not collection:
                raise ValueError(f"集合不存在: {collection_name}")
            return self._collection_service.get_collection_info(collection.id)

        # 使用默认集合
        from config.settings import settings
        default_collection_name = settings.milvus.milvus_collection_name
        collection = self._collection_service.get_collection_by_name(default_collection_name)
        if not collection:
            collection = self._collection_service.create_collection(default_collection_name, "默认文档集合")
        return self._collection_service.get_collection_info(collection.id)
