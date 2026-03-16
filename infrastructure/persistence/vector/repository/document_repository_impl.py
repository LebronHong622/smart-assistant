"""
DocumentRepository 接口的 Milvus 实现
"""
import json
from typing import List, Optional, Union
from pymilvus import Collection
from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from infrastructure.persistence.vector.vector_store import MilvusVectorStore
from infrastructure.core.log import app_logger
from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter


class MilvusDocumentRepository(DocumentRepository):
    """
    基于 Milvus 的文档仓库实现
    
    注意：使用自增ID模式，插入时不提供 id，由 Milvus 自动分配
    """

    def __init__(
        self,
        milvus_vector_store: MilvusVectorStore = None,
        embedding_generator: Optional[EmbeddingGeneratorPort] = None
    ):
        self.vector_store = milvus_vector_store or MilvusVectorStore()
        self._embedding_generator = embedding_generator or DashScopeEmbeddingAdapter()
        self.collection_name = self.vector_store.collection_name
        # 确保集合存在
        self.vector_store.ensure_collection_exists()

    def set_embedding_function(self, embedding_generator: EmbeddingGeneratorPort) -> None:
        """设置嵌入生成器"""
        self._embedding_generator = embedding_generator

    def _get_collection(self) -> Collection:
        """获取 Collection 对象"""
        return Collection(self.collection_name)

    def save(self, document: Document, **kwargs) -> Document:
        """保存文档（使用自增ID，插入后ID由数据库分配）"""
        app_logger.info(f"保存文档，插入前ID: {document.id}")

        # 如果文档没有嵌入向量，生成嵌入
        if document.embedding is None:
            app_logger.debug("文档缺少嵌入向量，正在生成...")
            document.embedding = self._embedding_generator.embed_text(document.content)

        # 转换为 Milvus 可插入的格式（不包含 id，由 auto_id 分配）
        doc_dict = {
            "content": document.content,
            "metadata": json.dumps(document.metadata),
            "embedding": document.embedding
        }

        # 插入文档并获取分配的 ID
        collection = self._get_collection()
        result = collection.insert([doc_dict])
        collection.flush()
        
        # 更新文档 ID 为数据库分配的值
        if result.primary_keys:
            document.id = result.primary_keys[0]
            app_logger.info(f"文档保存成功，分配ID: {document.id}")

        return document

    def save_all(self, documents: List[Document], **kwargs) -> List[Document]:
        """保存多个文档（使用自增ID）"""
        app_logger.info(f"批量保存文档: {len(documents)} 个")

        # 为缺少嵌入的文档生成嵌入向量
        docs_needing_embedding = [doc for doc in documents if doc.embedding is None]
        if docs_needing_embedding:
            app_logger.debug(f"批量生成嵌入向量: {len(docs_needing_embedding)} 个文档需要生成")
            self._embedding_generator.embed_documents(docs_needing_embedding)

        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "content": doc.content,
                "metadata": json.dumps(doc.metadata),
                "embedding": doc.embedding
            })

        # 批量插入并获取分配的 IDs
        collection = self._get_collection()
        result = collection.insert(doc_dicts)
        collection.flush()
        
        # 更新每个文档的 ID
        if result.primary_keys:
            for i, doc in enumerate(documents):
                if i < len(result.primary_keys):
                    doc.id = result.primary_keys[i]
            app_logger.info(f"批量保存成功，分配了 {len(result.primary_keys)} 个ID")

        return documents

    def find_by_id(self, document_id: int) -> Optional[Document]:
        """根据 ID 查找文档"""
        app_logger.debug(f"查找文档: {document_id}")

        collection = self._get_collection()
        collection.load()

        try:
            # 查询文档
            results = collection.query(
                expr=f'id == {document_id}',
                output_fields=["id", "content", "metadata"]
            )

            if not results:
                return None

            result = results[0]
            return Document(
                id=int(result["id"]),
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
                    id=int(result["id"]),
                    content=result["content"],
                    metadata=json.loads(result["metadata"]) if result.get("metadata") else {}
                ))

            return documents

        except Exception as e:
            app_logger.error(f"查找所有文档失败: {str(e)}")
            return []

    def delete(self, document: Document) -> None:
        """删除文档"""
        if document.id is not None:
            self.delete_by_id(document.id)

    def delete_by_id(self, document_id: Union[int, str]) -> None:
        """根据 ID 删除文档"""
        doc_id = int(document_id) if isinstance(document_id, str) else document_id
        app_logger.info(f"删除文档: {doc_id}")

        collection = self._get_collection()

        try:
            collection.delete(f'id == {doc_id}')
            collection.flush()
            app_logger.info(f"文档删除成功: {doc_id}")
        except Exception as e:
            app_logger.error(f"删除文档失败: {str(e)}")
            raise RuntimeError(f"删除文档失败: {str(e)}")

    def count(self) -> int:
        """统计文档数量"""
        collection = self._get_collection()
        return collection.num_entities

    def delete_all(self, document_ids: List[Union[int, str]]) -> None:
        """批量删除文档"""
        app_logger.info(f"批量删除文档: {len(document_ids)} 个")

        collection = self._get_collection()

        try:
            # 转换为整数列表
            int_ids = [int(did) if isinstance(did, str) else did for did in document_ids]
            # 使用 Milvus 表达式批量删除
            expr = f"id in {int_ids}"
            collection.delete(expr)
            collection.flush()
            app_logger.info(f"批量删除文档成功: {len(document_ids)} 个")
        except Exception as e:
            app_logger.error(f"批量删除文档失败: {str(e)}")
            raise RuntimeError(f"批量删除文档失败: {str(e)}")

    def search_by_vector(
        self,
        embedding: List[float],
        limit: int = 5,
        **kwargs
    ) -> List[Document]:
        """通过向量搜索相似文档"""
        app_logger.debug(f"向量搜索文档: limit={limit}")

        # 委托给 vector_store 搜索
        search_results = self.vector_store.search_documents(embedding, limit, **kwargs)

        # 转换搜索结果为 Document 实体
        documents = []
        for result in search_results:
            documents.append(Document(
                id=int(result["id"]),
                content=result["content"],
                metadata=json.loads(result["metadata"]) if result.get("metadata") else {},
                embedding=embedding
            ))

        return documents

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """通过文本搜索相似文档"""
        app_logger.debug(f"文本搜索文档: query={query[:50]}..., limit={limit}")

        # 使用 DashScope 生成查询文本的嵌入向量
        from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
        embedding_adapter = DashScopeEmbeddingAdapter()
        query_embedding = embedding_adapter.embed_text(query)

        # 通过向量搜索
        return self.search_by_vector(query_embedding, limit, **kwargs)
