"""
DocumentRepository 接口的 Milvus 实现
"""
import json
from typing import List, Optional
from uuid import UUID
from pymilvus import Collection
from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.persistence.vector.vector_store import MilvusVectorStore
from infrastructure.core.log import app_logger


class MilvusDocumentRepository(DocumentRepository):
    """
    基于 Milvus 的文档仓库实现
    """

    def __init__(self, milvus_vector_store: MilvusVectorStore = None):
        self.vector_store = milvus_vector_store or MilvusVectorStore()
        self.collection_name = self.vector_store.collection_name
        # 确保集合存在
        self.vector_store.ensure_collection_exists()

    def _get_collection(self) -> Collection:
        """获取 Collection 对象"""
        return Collection(self.collection_name)

    def save(self, document: Document) -> Document:
        """保存文档"""
        app_logger.info(f"保存文档: {document.id}")

        # 转换为 Milvus 可插入的格式
        doc_dict = {
            "id": str(document.id),
            "content": document.content,
            "metadata": json.dumps(document.metadata),
            "embedding": document.embedding
        }

        # 插入文档
        self.vector_store.insert_documents([doc_dict])

        return document

    def save_all(self, documents: List[Document]) -> List[Document]:
        """保存多个文档"""
        app_logger.info(f"批量保存文档: {len(documents)} 个")

        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "id": str(doc.id),
                "content": doc.content,
                "metadata": json.dumps(doc.metadata),
                "embedding": doc.embedding
            })

        # 批量插入
        self.vector_store.insert_documents(doc_dicts)

        return documents

    def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """根据 ID 查找文档"""
        app_logger.debug(f"查找文档: {document_id}")

        collection = self._get_collection()
        collection.load()

        try:
            # 查询文档
            results = collection.query(
                expr=f'id == "{str(document_id)}"',
                output_fields=["id", "content", "metadata"]
            )

            if not results:
                return None

            result = results[0]
            return Document(
                id=UUID(result["id"]),
                content=result["content"],
                metadata=json.loads(result["metadata"]) if result.get("metadata") else {}
            )

        except Exception as e:
            app_logger.error(f"查找文档失败: {str(e)}")
            return None

    def find_all(self, limit: int = 1000, offset: int = 0) -> List[Document]:
        """查找所有文档"""
        app_logger.debug("查找所有文档")

        collection = self._get_collection()
        collection.load()

        try:
            # 查询文档（支持分页）
            results = collection.query(
                expr="",
                output_fields=["id", "content", "metadata"],
                limit=limit,
                offset=offset
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
            app_logger.error(f"查找所有文档失败: {str(e)}")
            return []

    def delete(self, document: Document) -> None:
        """删除文档"""
        self.delete_by_id(document.id)

    def delete_by_id(self, document_id: UUID) -> None:
        """根据 ID 删除文档"""
        app_logger.info(f"删除文档: {document_id}")

        collection = self._get_collection()

        try:
            collection.delete(f'id == "{str(document_id)}"')
            collection.flush()
            app_logger.info(f"文档删除成功: {document_id}")
        except Exception as e:
            app_logger.error(f"删除文档失败: {str(e)}")
            raise RuntimeError(f"删除文档失败: {str(e)}")

    def count(self) -> int:
        """统计文档数量"""
        collection = self._get_collection()
        return collection.num_entities
