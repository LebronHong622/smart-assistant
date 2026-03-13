"""
Milvus LangChain 适配器
基于 langchain-milvus 实现 VectorStorePort 接口
"""

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_milvus import Milvus

from domain.shared.ports.vector_store_port import VectorStorePort
from config.settings import settings
from config.rag_settings import rag_settings
from infrastructure.core.log import app_logger


class MilvusLangchainAdapter(VectorStorePort):
    """
    基于 LangChain Milvus 的向量存储适配器
    
    实现 VectorStorePort 接口，提供与 LangChain 生态系统的集成
    支持自动创建 collection、批量插入、相似性搜索等功能
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        auto_create: bool = True,
    ):
        """
        初始化适配器
        
        Args:
            collection_name: 集合名称，默认使用配置中的名称
            embedding_function: 嵌入函数，用于生成向量
            auto_create: 是否自动创建集合
        """
        self._collection_name = collection_name or settings.milvus.milvus_collection_name
        self._embedding_function = embedding_function
        self._auto_create = auto_create
        self._vector_store: Optional[Milvus] = None
        self._connection_args = self._build_connection_args()
        
        app_logger.info(f"初始化 MilvusLangchainAdapter, collection: {self._collection_name}")

    def _build_connection_args(self) -> Dict[str, Any]:
        """构建连接参数"""
        uri = rag_settings.milvus.get_connection_uri()
        return {
            "uri": uri,
        }

    def _get_or_create_vector_store(self) -> Milvus:
        """获取或创建向量存储实例"""
        if self._vector_store is None:
            if self._embedding_function is None:
                raise ValueError("嵌入函数未设置，无法创建向量存储")
            
            langchain_config = rag_settings.milvus.langchain_config
            
            self._vector_store = Milvus(
                embedding_function=self._embedding_function,
                collection_name=self._collection_name,
                connection_args=self._connection_args,
                auto_id=langchain_config.auto_id,
                vector_field=langchain_config.vector_field,
                text_field=langchain_config.text_field,
            )
            app_logger.info(f"创建 LangChain Milvus 向量存储: {self._collection_name}")
        
        return self._vector_store

    def set_embedding_function(self, embedding_function: Any) -> None:
        """
        设置嵌入函数
        
        Args:
            embedding_function: 嵌入函数实例
        """
        self._embedding_function = embedding_function
        # 重置向量存储以使用新的嵌入函数
        self._vector_store = None

    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        插入文档嵌入向量
        
        Args:
            documents: 文档字典列表，每个字典应包含:
                      - content: 文档内容
                      - embedding: 嵌入向量（如果 embedding_function 未设置）
                      - metadata: 元数据（可选）
        """
        app_logger.info(f"插入 {len(documents)} 个文档到 Milvus: {self._collection_name}")
        
        # 转换为 LangChain Document 格式
        lc_documents = []
        embeddings = []
        
        for doc in documents:
            lc_doc = Document(
                page_content=doc.get("content", ""),
                metadata=doc.get("metadata", {})
            )
            lc_documents.append(lc_doc)
            
            if doc.get("embedding"):
                embeddings.append(doc["embedding"])
        
        # 获取向量存储
        vector_store = self._get_or_create_vector_store()
        
        # 如果提供了嵌入向量，使用 from_documents
        if embeddings and len(embeddings) == len(lc_documents):
            # 使用已有的嵌入向量
            vector_store.add_documents(lc_documents, embeddings=embeddings)
        else:
            # 使用嵌入函数生成向量
            vector_store.add_documents(lc_documents)
        
        app_logger.info(f"文档插入完成: {self._collection_name}")

    def search_documents(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            limit: 返回结果数量
            filter_expr: 过滤表达式（可选）
            
        Returns:
            相似文档列表，每个文档包含:
            - id: 文档 ID
            - content: 文档内容
            - metadata: 元数据
            - distance: 相似度距离
        """
        app_logger.info(f"搜索相似文档: collection={self._collection_name}, limit={limit}")
        
        vector_store = self._get_or_create_vector_store()
        
        # 使用 similarity_search_with_score_by_vector
        results = vector_store.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            k=limit,
            filter=filter_expr,
        )
        
        # 转换结果格式
        documents = []
        for doc, score in results:
            documents.append({
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance": score,
            })
        
        app_logger.debug(f"搜索完成，找到 {len(documents)} 个文档")
        return documents

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        from pymilvus import Collection, utility
        
        try:
            if self._collection_name not in utility.list_collections():
                return {
                    "exists": False,
                    "name": self._collection_name,
                }
            
            collection = Collection(self._collection_name)
            return {
                "exists": True,
                "name": collection.name,
                "description": collection.description,
                "num_entities": collection.num_entities,
            }
        except Exception as e:
            app_logger.error(f"获取集合信息失败: {e}")
            return {
                "exists": False,
                "name": self._collection_name,
                "error": str(e),
            }

    def get_collection_fields(self) -> List[str]:
        """
        获取集合字段列表
        
        Returns:
            字段名称列表
        """
        from pymilvus import Collection
        
        try:
            collection = Collection(self._collection_name)
            return [field.name for field in collection.schema.fields]
        except Exception as e:
            app_logger.error(f"获取集合字段失败: {e}")
            return []

    def ensure_collection_exists(self) -> None:
        """
        确保集合存在
        
        如果集合不存在，将自动创建
        """
        from pymilvus import utility
        
        if self._collection_name in utility.list_collections():
            app_logger.debug(f"集合已存在: {self._collection_name}")
            return
        
        if self._auto_create:
            # 通过创建向量存储来自动创建集合
            self._get_or_create_vector_store()
            app_logger.info(f"自动创建集合: {self._collection_name}")
        else:
            raise RuntimeError(f"集合不存在且未启用自动创建: {self._collection_name}")

    def delete_collection(self) -> None:
        """删除集合"""
        from pymilvus import utility
        from pymilvus import Collection
        
        if self._collection_name in utility.list_collections():
            collection = Collection(self._collection_name)
            collection.drop()
            app_logger.info(f"集合已删除: {self._collection_name}")
            self._vector_store = None

    def delete_documents(self, ids: List[str]) -> None:
        """
        根据ID删除文档
        
        Args:
            ids: 文档ID列表
        """
        from pymilvus import Collection
        
        collection = Collection(self._collection_name)
        expr = f'id in {ids}'
        collection.delete(expr)
        app_logger.info(f"删除文档: {len(ids)} 个")

    def search_by_text(
        self,
        query: str,
        limit: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        通过文本搜索相似文档
        
        Args:
            query: 查询文本
            limit: 返回结果数量
            filter_expr: 过滤表达式
            
        Returns:
            相似文档列表
        """
        if self._embedding_function is None:
            raise ValueError("嵌入函数未设置，无法进行文本搜索")
        
        vector_store = self._get_or_create_vector_store()
        
        results = vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            filter=filter_expr,
        )
        
        documents = []
        for doc, score in results:
            documents.append({
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "distance": score,
            })
        
        return documents

    def get_retriever(self, **kwargs) -> Any:
        """
        获取检索器
        
        Args:
            **kwargs: 检索器参数
            
        Returns:
            LangChain 检索器实例
        """
        vector_store = self._get_or_create_vector_store()
        return vector_store.as_retriever(**kwargs)
