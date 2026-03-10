"""
LangChain Milvus 适配器
将现有 VectorStorePort 实现包装为 LangChain 标准 VectorStore 接口
"""
from typing import List, Optional, Dict, Any, Iterable, Type
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, PrivateAttr
from domain.shared.ports.vector_store_port import VectorStorePort
from infrastructure.persistence.vector.factories.vector_store_factory import VectorStoreFactory
from infrastructure.external.model.embedding.adapters.langchain_embeddings_adapter import LangChainEmbeddingsAdapter
from config.settings import get_app_settings

settings = get_app_settings()

class LangChainMilvusAdapter(BaseModel, VectorStore):
    """
    LangChain Milvus 适配器
    兼容 LangChain 标准接口,内部使用项目现有 VectorStorePort 实现
    支持多 collection,每个 collection 可以有不同的 schema
    """
    collection_name: str = Field(default_factory=lambda: settings.milvus.milvus_collection_name)
    embedding_model: Embeddings = Field(default_factory=LangChainEmbeddingsAdapter)
    provider: str = Field(default="milvus")
    _vector_store: Optional[VectorStorePort] = PrivateAttr(default=None)
    _field_names: List[str] = PrivateAttr(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 传递 collection_name 给工厂
        self._vector_store = VectorStoreFactory.create(self.provider, collection_name=self.collection_name)
        # 确保集合存在
        self._vector_store.ensure_collection_exists()
        # 获取 collection 的 schema 字段
        self._field_names = self._vector_store.get_collection_fields()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        添加文本到向量存储
        :param texts: 文本列表
        :param metadatas: 元数据列表
        :param kwargs: 其他参数（如 ids）
        :return: 文档ID列表
        """
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized")

        documents = []
        ids = []
        text_list = list(texts)
        for i, text in enumerate(text_list):
            doc_id = kwargs.get("ids", [str(i)])[i] if "ids" in kwargs else str(i)
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            # 生成嵌入向量
            embedding = self.embedding_model.embed_query(text)

            # 动态构建 document，只包含 schema 中存在的字段
            document = {}
            if "id" in self._field_names:
                document["id"] = doc_id
            if "content" in self._field_names:
                document["content"] = text
            elif "text" in self._field_names:
                document["text"] = text
            if "embedding" in self._field_names:
                document["embedding"] = embedding
            if "metadata" in self._field_names:
                document["metadata"] = metadata

            # 从 metadata 中提取额外字段（扁平化存储到 schema 对应字段）
            for key, value in metadata.items():
                if key in self._field_names:
                    document[key] = value

            documents.append(document)
            ids.append(doc_id)

        # 插入到向量存储
        self._vector_store.insert_documents(documents)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        相似性搜索
        :param query: 查询文本
        :param k: 返回结果数量
        :param filter: 过滤条件（暂不支持）
        :param kwargs: 其他参数
        :return: 匹配的文档列表
        """
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized")

        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)
        # 搜索相似文档
        results = self._vector_store.search_documents(query_embedding, limit=k)
        # 转换为 LangChain Document 格式
        documents = []
        for result in results:
            # 动态获取 content 字段
            content = ""
            if "content" in result:
                content = result.get("content", "")
            elif "text" in result:
                content = result.get("text", "")
            # 获取 metadata（排除系统字段）
            metadata = {k: v for k, v in result.items()
                       if k not in ("id", "content", "text", "embedding", "score")}
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float]]:
        """
        带分数的相似性搜索
        :param query: 查询文本
        :param k: 返回结果数量
        :param filter: 过滤条件（暂不支持）
        :param kwargs: 其他参数
        :return: 匹配的文档和分数列表
        """
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized")

        # 生成查询向量
        query_embedding = self.embedding_model.embed_query(query)
        # 搜索相似文档
        results = self._vector_store.search_documents(query_embedding, limit=k)
        # 转换为 LangChain Document 格式和分数
        doc_score_pairs = []
        for result in results:
            # 动态获取 content 字段
            content = ""
            if "content" in result:
                content = result.get("content", "")
            elif "text" in result:
                content = result.get("text", "")
            # 获取 metadata（排除系统字段）
            metadata = {k: v for k, v in result.items()
                       if k not in ("id", "content", "text", "embedding", "score")}
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            score = result.get("score", 0.0)
            doc_score_pairs.append((doc, score))
        return doc_score_pairs

    @classmethod
    def from_texts(
        cls: Type["LangChainMilvusAdapter"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "LangChainMilvusAdapter":
        """
        从文本创建向量存储
        :param texts: 文本列表
        :param embedding: 嵌入模型
        :param metadatas: 元数据列表
        :param collection_name: collection 名称
        :param kwargs: 其他参数
        :return: LangChainMilvusAdapter 实例
        """
        vector_store = cls(embedding_model=embedding, collection_name=collection_name, **kwargs)
        vector_store.add_texts(texts, metadatas, **kwargs)
        return vector_store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        删除文档（暂不支持）
        :param ids: 文档ID列表
        :param kwargs: 其他参数
        :return: 删除是否成功
        """
        raise NotImplementedError("Delete operation is not supported yet")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        :return: 集合信息
        """
        if not self._vector_store:
            raise RuntimeError("Vector store not initialized")
        return self._vector_store.get_collection_info()
