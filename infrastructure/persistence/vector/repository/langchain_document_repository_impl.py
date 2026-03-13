"""
DocumentRepository 接口的 LangChain 实现
支持任意 LangChain VectorStore
"""

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import VectorStore

from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.core.log import app_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class LangChainDocumentRepository(DocumentRepository):
    """
    基于 LangChain VectorStore 的文档仓库实现
    
    实现 DocumentRepository 接口，支持任意 LangChain 兼容的向量存储
    通过依赖注入或 VectorStoreFactory 创建 VectorStore
    """

    def __init__(
        self,
        collection_name: str,
        embedding_function: Optional["Embeddings"] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        初始化仓库
        
        Args:
            collection_name: 集合名称
            embedding_function: 嵌入函数（当 vector_store 未提供时使用）
            vector_store: 可选的 VectorStore 实例（依赖注入）
        """
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        
        if vector_store is not None:
            self._vector_store = vector_store
        elif embedding_function is not None:
            # 使用 VectorStoreFactory 创建
            from infrastructure.rag.embeddings import VectorStoreFactory
            self._vector_store = VectorStoreFactory.create_store(
                embedding=embedding_function,
                collection_name=collection_name,
            )
        else:
            raise ValueError("必须提供 vector_store 或 embedding_function 参数")
        
        app_logger.info(f"初始化 LangChainDocumentRepository: {collection_name}, vector_store={type(self._vector_store).__name__}")

    def set_embedding_function(self, embedding_function: "Embeddings") -> None:
        """
        设置嵌入函数
        
        注意：此方法会重新创建 VectorStore
        """
        self._embedding_function = embedding_function
        from infrastructure.rag.embeddings import VectorStoreFactory
        self._vector_store = VectorStoreFactory.create_store(
            embedding=embedding_function,
            collection_name=self._collection_name,
        )
        app_logger.info(f"更新嵌入函数并重建 VectorStore: {self._collection_name}")

    def save(self, document: Document) -> Document:
        """
        保存文档
        
        Args:
            document: 文档实体
            
        Returns:
            保存后的文档实体
        """
        app_logger.info(f"保存文档: {document.id}")
        
        lc_doc = LCDocument(
            page_content=document.content,
            metadata={
                "id": str(document.id),
                **(document.metadata or {}),
            },
        )
        
        # 使用 VectorStore 添加文档
        self._vector_store.add_documents([lc_doc])
        return document

    def save_all(self, documents: List[Document]) -> List[Document]:
        """
        批量保存文档
        
        Args:
            documents: 文档实体列表
            
        Returns:
            保存后的文档实体列表
        """
        app_logger.info(f"批量保存文档: {len(documents)} 个")
        
        lc_docs = []
        for doc in documents:
            lc_docs.append(LCDocument(
                page_content=doc.content,
                metadata={
                    "id": str(doc.id),
                    **(doc.metadata or {}),
                },
            ))
        
        # 使用 VectorStore 批量添加文档
        self._vector_store.add_documents(lc_docs)
        return documents

    def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """
        根据 ID 查找文档
        
        注意：LangChain VectorStore 标准接口不支持直接 ID 查询，
        此方法通过元数据过滤尝试实现，可能不被所有 VectorStore 支持
        
        Args:
            document_id: 文档 ID
            
        Returns:
            文档实体或 None
        """
        app_logger.debug(f"查找文档: {document_id}")
        
        try:
            # 尝试使用 similarity_search 并通过元数据过滤
            # 注意：并非所有 VectorStore 都支持元数据过滤
            results = self._vector_store.similarity_search(
                query="",
                k=1,
                filter={"id": str(document_id)},
            )
            
            if results:
                doc = results[0]
                doc_id = doc.metadata.get("id")
                return Document(
                    id=UUID(doc_id) if doc_id else document_id,
                    content=doc.page_content,
                    metadata={k: v for k, v in doc.metadata.items() if k != "id"},
                )
            return None
        except Exception as e:
            app_logger.warning(f"通过 ID 查找文档失败（可能不支持）: {e}")
            return None

    def find_all(self, limit: int = 1000, offset: int = 0) -> List[Document]:
        """
        查找所有文档（支持分页）
        
        注意：LangChain VectorStore 标准接口不支持列出所有文档，
        此方法返回空列表，子类可覆盖实现
        
        Args:
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            文档实体列表
        """
        app_logger.debug(f"查找所有文档: limit={limit}, offset={offset}")
        app_logger.warning("LangChain VectorStore 标准接口不支持 find_all，返回空列表")
        return []

    def delete(self, document: Document) -> None:
        """
        删除文档
        
        Args:
            document: 文档实体
        """
        self.delete_by_id(document.id)

    def delete_by_id(self, document_id: UUID) -> None:
        """
        根据 ID 删除文档
        
        注意：LangChain VectorStore 标准接口不支持直接删除，
        需要底层实现支持
        
        Args:
            document_id: 文档 ID
        """
        app_logger.info(f"删除文档: {document_id}")
        
        # 尝试通过底层实现删除
        try:
            # 检查是否有 delete 方法（部分 VectorStore 支持）
            if hasattr(self._vector_store, 'delete'):
                self._vector_store.delete([str(document_id)])
                app_logger.info(f"文档删除成功: {document_id}")
            else:
                app_logger.warning(f"当前 VectorStore 不支持删除操作: {type(self._vector_store).__name__}")
        except Exception as e:
            app_logger.error(f"删除文档失败: {e}")
            raise RuntimeError(f"删除文档失败: {e}")

    def delete_all(self, document_ids: List[str]) -> None:
        """
        批量删除文档
        
        注意：LangChain VectorStore 标准接口不支持直接删除，
        需要底层实现支持
        
        Args:
            document_ids: 文档 ID 列表
        """
        app_logger.info(f"批量删除文档: {len(document_ids)} 个")
        
        try:
            if hasattr(self._vector_store, 'delete'):
                self._vector_store.delete(document_ids)
                app_logger.info(f"批量删除文档成功: {len(document_ids)} 个")
            else:
                app_logger.warning(f"当前 VectorStore 不支持批量删除操作: {type(self._vector_store).__name__}")
        except Exception as e:
            app_logger.error(f"批量删除文档失败: {e}")
            raise RuntimeError(f"批量删除文档失败: {e}")

    def count(self) -> int:
        """
        统计文档数量
        
        注意：LangChain VectorStore 标准接口不支持计数，
        需要底层实现支持
        
        Returns:
            文档数量
        """
        try:
            # 尝试通过底层实现获取计数
            if hasattr(self._vector_store, '_collection'):
                # Milvus
                return self._vector_store._collection.num_entities
            elif hasattr(self._vector_store, '_collection_name') and hasattr(self._vector_store, '_client'):
                # Chroma 或其他
                client = self._vector_store._client
                return client.get_collection(self._vector_store._collection_name).count()
            else:
                app_logger.warning("当前 VectorStore 不支持计数操作")
                return 0
        except Exception as e:
            app_logger.error(f"统计文档数量失败: {e}")
            return 0

    def search_by_vector(
        self,
        embedding: List[float],
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Document]:
        """
        通过向量搜索文档
        
        Args:
            embedding: 查询向量
            limit: 返回数量
            filter_expr: 过滤表达式（可选）
            
        Returns:
            匹配的文档列表
        """
        # 使用 VectorStore 的 similarity_search_by_vector
        results = self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=limit,
            filter=filter_expr,
        )
        
        documents = []
        for result in results:
            doc_id = result.metadata.get("id")
            documents.append(Document(
                id=UUID(doc_id) if doc_id else None,
                content=result.page_content,
                metadata={k: v for k, v in result.metadata.items() if k != "id"},
            ))
        
        return documents

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        filter_expr: Optional[str] = None,
    ) -> List[Document]:
        """
        通过文本搜索文档
        
        Args:
            query: 查询文本
            limit: 返回数量
            score_threshold: 相似度阈值（部分 VectorStore 支持）
            filter_expr: 过滤表达式（可选）
            
        Returns:
            匹配的文档列表
        """
        # 使用 VectorStore 的 similarity_search
        results = self._vector_store.similarity_search(
            query=query,
            k=limit,
            filter=filter_expr,
        )
        
        documents = []
        for result in results:
            doc_id = result.metadata.get("id")
            documents.append(Document(
                id=UUID(doc_id) if doc_id else None,
                content=result.page_content,
                metadata={k: v for k, v in result.metadata.items() if k != "id"},
            ))
        
        return documents

    def get_vector_store(self) -> VectorStore:
        """获取底层 VectorStore 实例"""
        return self._vector_store

    def get_retriever(self, **kwargs) -> Any:
        """
        获取 LangChain 检索器
        
        Args:
            **kwargs: 检索器参数
            
        Returns:
            LangChain 检索器实例
        """
        return self._vector_store.as_retriever(**kwargs)
