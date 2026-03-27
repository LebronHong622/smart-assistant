"""
DocumentRepository 接口的 LangChain 实现
支持任意 LangChain VectorStore
"""

import json
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from langchain_core.documents import Document as LCDocument
from langchain_core.vectorstores import VectorStore

from domain.entity.document.document import Document
from domain.repository.document.document_repository import DocumentRepository
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
        collection_name: Optional[str] = None,
        embedding_function: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        primary_key_field: Optional[str] = None,
    ):
        """
        初始化仓库

        Args:
            collection_name: 集合名称（可选，也可通过 setter 设置）
            embedding_function: 可选的嵌入函数（不提供则从工厂自动创建）
            vector_store: 可选的 VectorStore 实例（依赖注入用于测试）
            primary_key_field: 主键字段名称（不提供则从配置读取）
        """
        self._collection_name = collection_name

        # 获取主键字段配置
        if primary_key_field is None:
            # 从全局配置读取
            from config.rag_settings import get_rag_settings
            settings = get_rag_settings()
            self._primary_key_field = settings.vector.milvus.langchain_config.primary_key_field
        else:
            self._primary_key_field = primary_key_field

        self._embedding_function = embedding_function
        self._vector_store = vector_store
        self._vector_store_injected = vector_store is not None  # 标记是否由用户注入

        # 如果已经提供 collection_name，立即初始化 vector_store
        if self._collection_name is not None:
            self._initialize_vector_store()

        app_logger.info(f"初始化 LangChainDocumentRepository: collection={self._collection_name}, primary_key={self._primary_key_field}, vector_store={type(self._vector_store).__name__ if self._vector_store else 'None'}")

    @property
    def collection_name(self) -> Optional[str]:
        """获取集合名称"""
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str) -> None:
        """设置集合名称并初始化 vector_store"""
        if self._collection_name is not None and self._collection_name != value:
            app_logger.warning(f"修改集合名称: {self._collection_name} -> {value}")

        self._collection_name = value
        self._initialize_vector_store()
        app_logger.info(f"设置集合名称并完成初始化: {value}")

    def _initialize_vector_store(self) -> None:
        """
        初始化 vector_store（延迟初始化）

        只有在 collection_name 设置后才会真正创建 vector_store
        如果 vector_store 已经通过依赖注入提供，则不重新创建
        如果是我们自己创建的，修改 collection_name 后需要重新创建
        """
        if self._collection_name is None:
            # 尚未设置 collection_name，等待延迟初始化
            return

        if self._vector_store_injected:
            # 已经通过依赖注入提供 vector_store，永远不需要重新创建
            return

        if self._embedding_function is not None:
            # 提供 embedding_function，使用 VectorStoreFactory 创建
            from infrastructure.rag.embeddings import VectorStoreFactory
            self._vector_store = VectorStoreFactory.create_store(
                embedding=self._embedding_function,
                collection_name=self._collection_name,
            )
        else:
            # 自动从工厂创建 embedding 和 vector_store
            from infrastructure.rag.embeddings import EmbeddingFactory, VectorStoreFactory
            embedding_generator = EmbeddingFactory.create_embedding()
            self._embedding_function = embedding_generator.to_langchain_embeddings()
            self._vector_store = VectorStoreFactory.create_store(
                embedding=self._embedding_function,
                collection_name=self._collection_name,
            )

    def set_embedding_function(self, embedding_function: Embeddings) -> None:
        """
        设置嵌入函数

        注意：此方法会重新创建 VectorStore
        """
        from infrastructure.rag.embeddings import VectorStoreFactory
        from infrastructure.rag.factory.langchain_factory import LangChainEmbeddingAdapter
        # 如果传入的是适配器，获取实际的 LangChain Embeddings
        if isinstance(embedding_function, LangChainEmbeddingAdapter):
            self._embedding_function = embedding_function.to_langchain_embeddings()
        else:
            self._embedding_function = embedding_function
        self._vector_store = VectorStoreFactory.create_store(
            embedding=self._embedding_function,
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

        Args:
            document_id: 文档 ID

        Returns:
            文档实体或 None

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("find_by_id not implemented yet")

    def find_all(self, limit: int = 1000, offset: int = 0) -> List[Document]:
        """
        查找所有文档（支持分页）

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            文档实体列表

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("find_all not implemented yet")

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

        try:
            # 对于 Milvus（有 col 属性），优先使用直接集合操作
            # 使用配置的主键字段进行删除
            if hasattr(self._vector_store, 'col'):
                collection = self._vector_store.col
                collection.delete(f'{self._primary_key_field} == {doc_id}')
                collection.flush()
                app_logger.info(f"文档删除成功: {doc_id}")
            elif hasattr(self._vector_store, 'delete'):
                self._vector_store.delete([str(doc_id)])
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
            # 对于 Milvus（有 col 属性），优先使用直接集合操作
            # 使用配置的主键字段进行删除
            if hasattr(self._vector_store, 'col'):
                collection = self._vector_store.col
                int_ids = [int(did) if isinstance(did, str) else did for did in document_ids]
                collection.delete(f"{self._primary_key_field} in {int_ids}")
                collection.flush()
                app_logger.info(f"批量删除文档成功: {len(document_ids)} 个")
            elif hasattr(self._vector_store, 'delete'):
                # 转换为字符串列表
                str_ids = [str(did) for did in document_ids]
                self._vector_store.delete(str_ids)
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
            if hasattr(self._vector_store, 'col'):
                # Milvus - LangChain Milvus 使用 'col' 属性
                return self._vector_store.col.num_entities
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
        search_type: str = "similarity",
        **kwargs
    ) -> List[Document]:
        """
        通过向量搜索文档

        Args:
            embedding: 查询向量
            limit: 返回数量
            search_type: 搜索类型
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("search_by_vector with search_type not implemented yet")

    def _convert_search_results(
        self,
        results: List[LCDocument],
        with_score: bool = False
    ) -> List[Document]:
        """
        将 VectorStore.search() 返回的 LCDocument 列表转换为 Domain Document 列表

        Args:
            results: search() 返回结果
            with_score: 是否需要计算分数，False 时直接返回 0 节省计算

        Returns:
            转换后的 Domain Document 列表
        """
        documents = []
        for result in results:
            # LangChain Milvus 返回的 metadata 中，ID 在 'pk' 字段
            doc_id = result.metadata.get("pk") or result.metadata.get("id")
            # 提取分数（LangChain 返回的 score 在 result.score 或 metadata）
            score = getattr(result, 'score', None)
            if score is None:
                score = result.metadata.get('score')

            doc = Document(
                id=int(doc_id) if doc_id is not None else None,
                content=result.page_content,
                metadata={k: v for k, v in result.metadata.items() if k not in ("pk", "id")},
            )

            if with_score and score is not None:
                # LangChain search() 返回的 score 实际上是 distance
                doc.distance = score
                from config.settings import settings
                metric_type = settings.milvus.milvus_metric_type
                if metric_type.upper() == "COSINE":
                    # COSINE 距离：distance = 1 - cosine_similarity
                    doc.similarity_score = 1.0 - doc.distance
                else:  # L2 或 IP
                    # L2 距离：similarity = 1 / (1 + distance)
                    doc.similarity_score = 1.0 / (1.0 + doc.distance)
            else:
                # 当不需要分数时直接返回 0，节省计算开销
                doc.distance = 0.0
                doc.similarity_score = 0.0

            documents.append(doc)

        return documents

    def _convert_score_results(
        self,
        results: List[tuple[LCDocument, float]],
    ) -> List[Document]:
        """
        将 similarity_search_with_relevance_scores 返回的 (doc, score) 元组列表转换为 Domain Document 列表

        Args:
            results: similarity_search_with_relevance_scores 返回结果，
                     每个元素是 (LCDocument, similarity_score) 元组

        Returns:
            转换后的 Domain Document 列表
        """
        documents = []
        from config.settings import settings
        metric_type = settings.milvus.milvus_metric_type

        for doc, similarity_score in results:
            # LangChain 返回的 metadata 中，ID 在 'pk' 字段
            doc_id = doc.metadata.get("pk") or doc.metadata.get("id")

            domain_doc = Document(
                id=int(doc_id) if doc_id is not None else None,
                content=doc.page_content,
                metadata={k: v for k, v in doc.metadata.items() if k not in ("pk", "id")},
                similarity_score=similarity_score,
            )

            # 根据 similarity_score 反向计算 distance
            if metric_type.upper() == "COSINE":
                # COSINE: distance = 1 - similarity_score
                domain_doc.distance = 1.0 - similarity_score
            else:  # L2 或 IP
                # similarity = 1 / (1 + distance) => distance = (1 / similarity) - 1
                if similarity_score > 0:
                    domain_doc.distance = (1.0 / similarity_score) - 1.0
                else:
                    domain_doc.distance = float('inf')

            documents.append(domain_doc)

        return documents

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        search_type: str = "similarity",
        with_score: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        通过文本搜索文档

        Args:
            query: 查询文本
            limit: 返回数量
            score_threshold: 相似度阈值（部分 VectorStore 支持）
            search_type: 搜索类型，默认为 similarity，可选值: similarity, mmr
            with_score: 是否直接从 LangChain 获取相似度分数，默认为 False 保持向后兼容，
                        当为 True 时使用 similarity_search_with_relevance_scores 获取分数
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表
        """
        if with_score:
            try:
                # 使用 similarity_search_with_relevance_scores 直接获取相似度分数
                results = self._vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=limit,
                    score_threshold=score_threshold,
                    **kwargs
                )
                return self._convert_score_results(results)
            except AssertionError as e:
                # 当启用混合搜索（多向量配置：稠密 + BM25 稀疏）时，
                # similarity_search_with_relevance_scores 会抛出异常:
                # "No supported normalization function for multi vectors. Could not determine relevance function."
                # 自动回退到使用 search 方法，该方法在混合搜索下能正常工作
                if "No supported normalization function for multi vectors" in str(e):
                    app_logger.warning(f"similarity_search_with_relevance_scores not supported with hybrid search (multi vectors), falling back to search method with score extraction from metadata")
                    # 使用 search 方法，结果中分数已经在 metadata 中
                    try:
                        results = self._vector_store.search(
                            query=query,
                            search_type=search_type,
                            k=limit,
                            score_threshold=score_threshold,
                            **kwargs
                        )
                    except AssertionError as e2:
                        # Milvus 多向量配置不支持 MMR 搜索，再次回退到 similarity 搜索
                        if "does not support multi-vector search" in str(e2) and search_type == "mmr":
                            app_logger.warning(f"MMR search not supported with multi-vector configuration, falling back to similarity search")
                            results = self._vector_store.search(
                                query=query,
                                search_type="similarity",
                                k=limit,
                                score_threshold=score_threshold,
                                **kwargs
                            )
                        else:
                            raise
                    # 使用 _convert_search_results 并设置 with_score=True 从 metadata 提取分数
                    return self._convert_search_results(results, with_score=True)
                else:
                    raise
        else:
            # 使用原有 search 方法，支持不同搜索类型
            try:
                results = self._vector_store.search(
                    query=query,
                    search_type=search_type,
                    k=limit,
                    score_threshold=score_threshold,
                    **kwargs
                )
            except AssertionError as e:
                # Milvus 多向量配置（BM25 + dense）不支持 MMR 搜索
                # 自动回退到 similarity 搜索
                if "does not support multi-vector search" in str(e) and search_type == "mmr":
                    app_logger.warning(f"MMR search not supported with multi-vector configuration, falling back to similarity search")
                    results = self._vector_store.search(
                        query=query,
                        search_type="similarity",
                        k=limit,
                        score_threshold=score_threshold,
                        **kwargs
                    )
                else:
                    raise

            return self._convert_search_results(results, with_score=False)

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

    # ========== 异步方法 ==========

    async def asave(self, document: Document, **kwargs) -> Document:
        """
        异步保存文档

        Args:
            document: 文档实体（使用自增ID，插入前 id 为 None）
            **kwargs: 额外参数传递给底层 VectorStore

        Returns:
            保存后的文档实体（包含数据库分配的 ID）
        """
        app_logger.info(f"异步保存文档，插入前ID: {document.id}")

        # 构建元数据（不包含 id，由数据库自增）
        metadata = {
            **(document.metadata or {}),
        }

        lc_doc = LCDocument(
            page_content=document.content,
            metadata=metadata,
        )

        # 使用 VectorStore 异步添加文档，获取分配的 IDs
        ids = await self._vector_store.aadd_documents([lc_doc], **kwargs)

        # 更新文档 ID 为数据库分配的值
        if ids and len(ids) > 0:
            try:
                document.id = int(ids[0])
                app_logger.info(f"文档异步保存成功，分配ID: {document.id}")
            except (ValueError, TypeError):
                # 如果无法转换为整数，保持字符串形式
                app_logger.warning(f"分配的ID无法转换为整数: {ids[0]}")

        return document

    async def asave_all(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        异步批量保存文档

        Args:
            documents: 文档实体列表（使用自增ID）
            **kwargs: 额外参数传递给底层 VectorStore

        Returns:
            保存后的文档实体列表（包含数据库分配的 ID）
        """
        app_logger.info(f"异步批量保存文档: {len(documents)} 个")

        lc_docs = []
        for doc in documents:
            lc_docs.append(LCDocument(
                page_content=doc.content,
                metadata={
                    **(doc.metadata or {}),
                },
            ))

        # 使用 VectorStore 异步批量添加文档
        ids = await self._vector_store.aadd_documents(lc_docs, **kwargs)

        # 更新每个文档的 ID
        if ids:
            for i, doc in enumerate(documents):
                if i < len(ids):
                    try:
                        doc.id = int(ids[i])
                    except (ValueError, TypeError):
                        pass
            app_logger.info(f"异步批量保存成功，分配了 {len(ids)} 个ID")

        return documents

    async def afind_by_id(self, document_id: int) -> Optional[Document]:
        """
        异步根据 ID 查找文档

        Args:
            document_id: 文档 ID

        Returns:
            文档实体或 None

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("afind_by_id not implemented yet")

    async def afind_all(self, limit: int = 1000, offset: int = 0) -> List[Document]:
        """
        异步查找所有文档（支持分页）

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            文档实体列表

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("afind_all not implemented yet")

    async def adelete(self, document: Document) -> None:
        """
        异步删除文档

        Args:
            document: 文档实体
        """
        if document.id is not None:
            await self.adelete_by_id(document.id)

    async def adelete_by_id(self, document_id: Union[int, str]) -> None:
        """
        异步根据 ID 删除文档

        Note:
            需要底层 VectorStore 支持异步删除

        Args:
            document_id: 文档 ID
        """
        doc_id = int(document_id) if isinstance(document_id, str) else document_id
        app_logger.info(f"异步删除文档: {doc_id}")

        # 调用底层异步删除
        try:
            if hasattr(self._vector_store, 'adelete'):
                await self._vector_store.adelete([str(doc_id)])
                app_logger.info(f"文档异步删除成功: {doc_id}")
            else:
                app_logger.warning(f"当前 VectorStore 不支持异步删除操作: {type(self._vector_store).__name__}")
        except Exception as e:
            app_logger.error(f"异步删除文档失败: {e}")
            raise RuntimeError(f"异步删除文档失败: {e}")

    async def adelete_all(self, document_ids: List[Union[int, str]]) -> None:
        """
        异步批量删除文档

        Note:
            需要底层 VectorStore 支持异步删除

        Args:
            document_ids: 文档 ID 列表
        """
        app_logger.info(f"异步批量删除文档: {len(document_ids)} 个")

        try:
            # 转换为字符串列表
            str_ids = [str(did) for did in document_ids]

            if hasattr(self._vector_store, 'adelete'):
                await self._vector_store.adelete(str_ids)
                app_logger.info(f"异步批量删除成功: {len(document_ids)} 个")
            else:
                app_logger.warning(f"当前 VectorStore 不支持异步批量删除操作: {type(self._vector_store).__name__}")
        except Exception as e:
            app_logger.error(f"异步批量删除失败: {e}")
            raise RuntimeError(f"异步批量删除失败: {e}")

    async def acount(self) -> int:
        """
        异步统计文档数量

        Returns:
            文档数量
        """
        # 计数不支持异步，复用同步逻辑
        return self.count()

    async def asearch_by_text(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
        search_type: str = "similarity",
        with_score: bool = False,
        **kwargs
    ) -> List[Document]:
        """
        异步通过文本搜索文档

        Args:
            query: 查询文本
            limit: 返回数量
            score_threshold: 相似度阈值（部分 VectorStore 支持）
            search_type: 搜索类型，默认为 similarity，可选值: similarity, mmr
            with_score: 是否直接从 LangChain 获取相似度分数，默认为 False 保持向后兼容，
                        当为 True 时使用 asimilarity_search_with_relevance_scores 获取分数
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表
        """
        if with_score:
            try:
                # 使用 asimilarity_search_with_relevance_scores 直接获取相似度分数
                results = await self._vector_store.asimilarity_search_with_relevance_scores(
                    query=query,
                    k=limit,
                    score_threshold=score_threshold,
                    **kwargs
                )
                return self._convert_score_results(results)
            except AssertionError as e:
                # 当启用混合搜索（多向量配置：稠密 + BM25 稀疏）时，
                # asimilarity_search_with_relevance_scores 会抛出异常:
                # "No supported normalization function for multi vectors. Could not determine relevance function."
                # 自动回退到使用 asearch 方法，该方法在混合搜索下能正常工作
                if "No supported normalization function for multi vectors" in str(e):
                    app_logger.warning(f"asimilarity_search_with_relevance_scores not supported with hybrid search (multi vectors), falling back to asearch method with score extraction from metadata")
                    # 使用 asearch 方法，结果中分数已经在 metadata 中
                    try:
                        results = await self._vector_store.asearch(
                            query=query,
                            search_type=search_type,
                            k=limit,
                            score_threshold=score_threshold,
                            **kwargs
                        )
                    except AssertionError as e2:
                        # Milvus 多向量配置不支持 MMR 搜索，再次回退到 similarity 搜索
                        if "does not support multi-vector search" in str(e2) and search_type == "mmr":
                            app_logger.warning(f"MMR search not supported with multi-vector configuration, falling back to similarity search")
                            results = await self._vector_store.asearch(
                                query=query,
                                search_type="similarity",
                                k=limit,
                                score_threshold=score_threshold,
                                **kwargs
                            )
                        else:
                            raise
                    # 使用 _convert_search_results 并设置 with_score=True 从 metadata 提取分数
                    return self._convert_search_results(results, with_score=True)
                else:
                    raise
        else:
            # 使用原有异步 search 方法，支持不同搜索类型
            try:
                results = await self._vector_store.asearch(
                    query=query,
                    search_type=search_type,
                    k=limit,
                    score_threshold=score_threshold,
                    **kwargs
                )
            except AssertionError as e:
                # Milvus 多向量配置（BM25 + dense）不支持 MMR 搜索
                # 自动回退到 similarity 搜索
                if "does not support multi-vector search" in str(e) and search_type == "mmr":
                    app_logger.warning(f"MMR search not supported with multi-vector configuration, falling back to similarity search")
                    results = await self._vector_store.asearch(
                        query=query,
                        search_type="similarity",
                        k=limit,
                        score_threshold=score_threshold,
                        **kwargs
                    )
                else:
                    raise

            return self._convert_search_results(results, with_score=False)

    async def asearch_by_vector(
        self,
        embedding: List[float],
        limit: int = 5,
        search_type: str = "similarity",
        **kwargs
    ) -> List[Document]:
        """
        异步通过向量搜索文档

        Args:
            embedding: 查询向量
            limit: 返回数量
            search_type: 搜索类型
            **kwargs: 额外参数传递给底层 VectorStore（例如 filter）

        Returns:
            匹配的文档列表

        Raises:
            NotImplementedError: 此功能暂未实现
        """
        raise NotImplementedError("asearch_by_vector with search_type not implemented yet")
