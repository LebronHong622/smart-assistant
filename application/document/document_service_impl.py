"""
DocumentService 接口实现
"""
import json
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pymilvus import Collection
from domain.document.entity.document import Document
from domain.document.service.document_service import DocumentService
from domain.document.value_object.document_metadata import DocumentMetadata
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.model.embeddings_manager import EmbeddingsManager
from infrastructure.vector.vector_store import MilvusVectorStore
from infrastructure.log import app_logger


class DocumentServiceImpl(DocumentService):
    """
    文档管理服务实现
    """

    def __init__(self, document_repository: DocumentRepository = None):
        self.document_repository = document_repository or self._get_default_repository()
        self.embeddings_manager = EmbeddingsManager()
        self.vector_store = MilvusVectorStore()

    def _get_default_repository(self):
        """获取默认的文档仓库实现"""
        from infrastructure.vector.repository.document_repository_impl import MilvusDocumentRepository
        return MilvusDocumentRepository()

    def create_document(self, content: str, metadata: DocumentMetadata) -> Document:
        """创建文档"""
        app_logger.info(f"创建文档: {metadata.title}")

        # 生成嵌入向量
        embedding = self.embeddings_manager.generate_embedding(content)

        # 创建文档实体
        document = Document(
            content=content,
            metadata=metadata.model_dump(),
            embedding=embedding
        )

        # 保存文档
        return self.document_repository.save(document)

    def update_document(self, document_id: UUID, content: Optional[str] = None, metadata: Optional[DocumentMetadata] = None) -> Document:
        """更新文档"""
        app_logger.info(f"更新文档: {document_id}")

        # 获取现有文档
        document = self.document_repository.find_by_id(document_id)
        if not document:
            raise RuntimeError(f"文档不存在: {document_id}")

        # 更新内容
        if content is not None:
            document.content = content
            # 重新生成嵌入向量
            document.embedding = self.embeddings_manager.generate_embedding(content)

        # 更新元数据
        if metadata is not None:
            document.metadata = metadata.model_dump()

        # 保存更新
        # 注意：Milvus 不支持更新，需要先删除再插入
        self.document_repository.delete_by_id(document_id)
        return self.document_repository.save(document)

    def delete_document(self, document_id: UUID) -> None:
        """删除文档"""
        app_logger.info(f"删除文档: {document_id}")
        self.document_repository.delete_by_id(document_id)

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """获取文档"""
        return self.document_repository.find_by_id(document_id)

    def list_documents(self, limit: int = 10, offset: int = 0) -> List[Document]:
        """列表获取文档"""
        app_logger.debug(f"获取文档列表: limit={limit}, offset={offset}")

        collection = Collection(self.vector_store.collection_name)
        collection.load()

        try:
            # 查询文档
            results = collection.query(
                expr="",
                output_fields=["id", "content", "metadata"],
                offset=offset,
                limit=limit
            )

            documents = []
            for result in results:
                documents.append(Document(
                    id=UUID(result["id"]),
                    content=result["content"],
                    metadata=json.loads(result["metadata"]) if result.get("metadata") else {}
                ))

            return documents

        except Exception as e:
            app_logger.error(f"获取文档列表失败: {str(e)}")
            return []

    def search_documents_by_metadata(self, metadata: dict) -> List[Document]:
        """根据元数据搜索文档"""
        app_logger.debug(f"根据元数据搜索文档: {metadata}")

        collection = Collection(self.vector_store.collection_name)
        collection.load()

        try:
            # 构建查询表达式
            expr_conditions = []
            for key, value in metadata.items():
                # 元数据存储为 JSON 字符串，需要使用 JSON 匹配
                expr_conditions.append(f'JSON_CONTAINS(metadata, \'"{value}"\', "$.{key}")')

            expr = " AND ".join(expr_conditions) if expr_conditions else ""

            # 查询文档
            results = collection.query(
                expr=expr,
                output_fields=["id", "content", "metadata"],
                limit=1000
            )

            documents = []
            for result in results:
                documents.append(Document(
                    id=UUID(result["id"]),
                    content=result["content"],
                    metadata=json.loads(result["metadata"]) if result.get("metadata") else {}
                ))

            return documents

        except Exception as e:
            app_logger.error(f"根据元数据搜索文档失败: {str(e)}")
            return []

    def count_documents(self) -> int:
        """统计文档数量"""
        return self.document_repository.count()
