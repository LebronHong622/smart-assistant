"""
DocumentRepository 接口的 LangChain 实现
支持任意 LangChain VectorStore
"""

import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import VectorStore

from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from infrastructure.core.log import app_logger

from langchain_core.embeddings import Embeddings


class LangChainDocumentRepository(DocumentRepository):
    """
    基于 LangChain VectorStore 的文档仓库实现
    
    实现 DocumentRepository 接口，支持任意 LangChain 兼容的向量存储
    通过依赖注入或 VectorStoreFactory 创建 VectorStore
    
    注意：使用自增ID模式，ID 在插入前为 None，插入后由数据库分配
    """

    def __init__(
        self,
        collection_name: str,
        embedding_function: Optional[Embeddings] = None,
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

    def set_embedding_function(self, embedding_function: Embeddings) -> None:
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

    def save(self, document: Document, **kwargs) -> Document:
        """
        保存文档

        Args:
            document: 文档实体（使用自增ID，插入前 id 为 None）
            **kwargs: 额外参数传递给底层 VectorStore

        Returns:
            保存后的文档实体（包含数据库分配的 ID）
        """
        app_logger.info(f"保存文档，插入前ID: {document.id}")

        # 构建元数据（不包含 id，由数据库自增）
        metadata = {
            **(document.metadata or {}),
        }

        lc_doc = LCDocument(
            page_content=document.content,
            metadata=metadata,
        )

        # 使用 VectorStore 添加文档，获取分配的 IDs
        ids = self._vector_store.add_documents([lc_doc], **kwargs)
        
        # 更新文档 ID 为数据库分配的值
        if ids and len(ids) > 0:
            try:
                document.id = int(ids[0])
                app_logger.info(f"文档保存成功，分配ID: {document.id}")
            except (ValueError, TypeError):
                # 如果无法转换为整数，保持字符串形式
                app_logger.warning(f"分配的ID无法转换为整数: {ids[0]}")

        return document

    def save_all(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        批量保存文档

        Args:
            documents: 文档实体列表（使用自增ID）
            **kwargs: 额外参数传递给底层 VectorStore

        Returns:
            保存后的文档实体列表（包含数据库分配的 ID）
        """
        app_logger.info(f"批量保存文档: {len(documents)} 个")

        lc_docs = []
        for doc in documents:
            lc_docs.append(LCDocument(
                page_content=doc.content,
                metadata={
                    **(doc.metadata or {}),
                },
            ))

        # 使用 VectorStore 批量添加文档
        ids = self._vector_store.add_documents(lc_docs, **kwargs)
        
        # 更新每个文档的 ID
        if ids:
            for i, doc in enumerate(documents):
                if i < len(ids):
                    try:
                        doc.id = int(ids[i])
                    except (ValueError, TypeError):
                        pass
            app_logger.info(f"批量保存成功，分配了 {len(ids)} 个ID")
        
        return documents

    def find_by_id(self, document_id: int) -> Optional[Document]:
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
            # 尝试通过底层实现获取文档
            if hasattr(self._vector_store, '_collection'):
                # Milvus 实现
                collection = self._vector_store._collection
                results = collection.query(
                    expr=f'id == {document_id}',
                    output_fields=["id", "content", "metadata"]
                )
                if results:
                    result = results[0]
                    return Document(
                        id=int(result["id"]),
                        content=result["content"],
                        metadata=json.loads(result.get("metadata", "{}")),
                    )
            
            # 通用方式：尝试使用 similarity_search 并通过元数据过滤
            results = self._vector_store.similarity_search(
                query="",
                k=1,
                filter={"id": str(document_id)},
            )
            
            if results:
                doc = results[0]
                doc_id = doc.metadata.get("id")
                return Document(
                    id=int(doc_id) if doc_id else document_id,
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
        
        try:
            # 尝试通过底层 Milvus 实现
            if hasattr(self._vector_store, '_collection'):
                collection = self._vector_store._collection
                collection.load()
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
                        metadata=json.loads(result.get("metadata", "{}")),
                    ))
                return documents
        except Exception as e:
            app_logger.warning(f"查找所有文档失败: {e}")
        
        app_logger.warning("LangChain VectorStore 标准接口不支持 find_all，返回空列表")
        return []

    def delete(self, document: Document) -> None:
        """
        删除文档
        
        Args:
            document: 文档实体
        """
        if document.id is not None:
            self.delete_by_id(document.id)

    def delete_by_id(self, document_id: Union[int, str]) -> None:
        """
        根据 ID 删除文档
        
        注意：LangChain VectorStore 标准接口不支持直接删除，
        需要底层实现支持
        
        Args:
            document_id: 文档 ID
        """
        doc_id = int(document_id) if isinstance(document_id, str) else document_id
        app_logger.info(f"删除文档: {doc_id}")
        
        # 尝试通过底层实现删除
        try:
            # 检查是否有 delete 方法（部分 VectorStore 支持）
            if hasattr(self._vector_store, 'delete'):
                self._vector_store.delete([str(doc_id)])
                app_logger.info(f"文档删除成功: {doc_id}")
            elif hasattr(self._vector_store, '_collection'):
                # Milvus 直接删除
                collection = self._vector_store._collection
                collection.delete(f'id == {doc_id}')
                collection.flush()
                app_logger.info(f"文档删除成功: {doc_id}")
            else:
                app_logger.warning(f"当前 VectorStore 不支持删除操作: {type(self._vector_store).__name__}")
        except Exception as e:
            app_logger.error(f"删除文档失败: {e}")
            raise RuntimeError(f"删除文档失败: {e}")

    def delete_all(self, document_ids: List[Union[int, str]]) -> None:
        """
        批量删除文档
        
        注意：LangChain VectorStore 标准接口不支持直接删除，
        需要底层实现支持
        
        Args:
            document_ids: 文档 ID 列表
        """
        app_logger.info(f"批量删除文档: {len(document_ids)} 个")
        
        try:
            # 转换为字符串列表
            str_ids = [str(did) for did in document_ids]
            
            if hasattr(self._vector_store, 'delete'):
                self._vector_store.delete(str_ids)
                app_logger.info(f"批量删除文档成功: {len(document_ids)} 个")
            elif hasattr(self._vector_store, '_collection'):
                # Milvus 批量删除
                collection = self._vector_store._collection
                int_ids = [int(did) if isinstance(did, str) else did for did in document_ids]
                collection.delete(f"id in {int_ids}")
                collection.flush()
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
        **kwargs
    ) -> List[Document]:
        """
        通过向量搜索文档

        Args:
            embedding: 查询向量
            limit: 返回数量
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表
        """
        # 使用 VectorStore 的 similarity_search_by_vector
        results = self._vector_store.similarity_search_by_vector(
            embedding=embedding,
            k=limit,
            **kwargs
        )

        documents = []
        for result in results:
            doc_id = result.metadata.get("id")
            documents.append(Document(
                id=int(doc_id) if doc_id is not None else None,
                content=result.page_content,
                metadata={k: v for k, v in result.metadata.items() if k != "id"},
            ))

        return documents

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        **kwargs
    ) -> List[Document]:
        """
        通过文本搜索文档

        Args:
            query: 查询文本
            limit: 返回数量
            score_threshold: 相似度阈值（部分 VectorStore 支持）
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表
        """
        # 使用 VectorStore 的 similarity_search
        results = self._vector_store.similarity_search(
            query=query,
            k=limit,
            **kwargs
        )

        documents = []
        for result in results:
            doc_id = result.metadata.get("id")
            documents.append(Document(
                id=int(doc_id) if doc_id is not None else None,
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
