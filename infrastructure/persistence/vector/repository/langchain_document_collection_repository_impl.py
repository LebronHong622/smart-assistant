"""
DocumentCollectionRepository 接口的 LangChain Milvus 实现
"""

import uuid
from typing import List, Optional
from uuid import UUID

from pymilvus import Collection, utility

from domain.entity.document.document_collection import DocumentCollection
from domain.repository.document.document_collection_repository import DocumentCollectionRepository
from infrastructure.persistence.vector.adapters.milvus_langchain_adapter import MilvusLangchainAdapter
from infrastructure.persistence.vector.milvus_collections.collection_manager import (
    MilvusCollectionCreator,
    CollectionSchemaConfig,
)
from infrastructure.persistence.vector.milvus_client import milvus_client
from infrastructure.core.log import app_logger
from config.settings import settings
from config.rag_settings import rag_settings


class LangChainDocumentCollectionRepository(DocumentCollectionRepository):
    """
    基于 LangChain Milvus 的文档集合仓库实现
    
    实现 DocumentCollectionRepository 接口，提供文档集合的管理操作
    每个业务领域对应一个 Milvus collection
    """

    def __init__(self):
        """初始化仓库"""
        self._collection_creator = MilvusCollectionCreator()
        app_logger.info("初始化 LangChainDocumentCollectionRepository")

    def _name_to_uuid(self, name: str) -> UUID:
        """
        将集合名称转换为 UUID
        
        使用 UUID5 基于命名空间生成确定性 UUID
        
        Args:
            name: 集合名称
            
        Returns:
            UUID
        """
        return uuid.uuid5(uuid.NAMESPACE_DNS, name)

    def _create_collection_schema(self, collection: DocumentCollection) -> CollectionSchemaConfig:
        """
        创建集合 Schema 配置
        
        Args:
            collection: 文档集合实体
            
        Returns:
            Schema 配置
        """
        dimension = rag_settings.milvus.default_dimension
        
        return CollectionSchemaConfig(
            collection_name=collection.name,
            description=collection.description or "",
            fields=[
                {
                    "name": "id",
                    "data_type": "VARCHAR",
                    "max_length": 64,
                    "is_primary": True,
                    "auto_id": rag_settings.milvus.langchain_config.auto_id,
                },
                {
                    "name": rag_settings.milvus.langchain_config.text_field,
                    "data_type": "VARCHAR",
                    "max_length": 8192,  # 支持更长的文档内容
                },
                {
                    "name": rag_settings.milvus.langchain_config.metadata_field,
                    "data_type": "VARCHAR",
                    "max_length": 2048,  # 支持更丰富的元数据
                },
                {
                    "name": rag_settings.milvus.langchain_config.vector_field,
                    "data_type": "FLOAT_VECTOR",
                    "dim": dimension,
                },
            ],
            index={
                "field_name": rag_settings.milvus.langchain_config.vector_field,
                "index_type": rag_settings.milvus.index_type,
                "metric_type": rag_settings.milvus.metric_type,
                "params": {"nlist": rag_settings.milvus.n_list},
            },
        )

    def save(self, collection: DocumentCollection) -> DocumentCollection:
        """
        保存文档集合
        
        创建 Milvus collection，如果已存在则跳过
        
        Args:
            collection: 文档集合实体
            
        Returns:
            保存后的文档集合实体
        """
        app_logger.info(f"保存文档集合: {collection.name}")

        # 检查集合是否已存在
        if collection.name in utility.list_collections():
            app_logger.warning(f"集合已存在: {collection.name}，将使用现有集合")
            return collection

        # 创建 Schema 配置
        schema_config = self._create_collection_schema(collection)

        # 创建集合
        try:
            self._collection_creator.create_collection(schema_config, overwrite=False)
            # 注册 Schema 到 MilvusClient
            milvus_client.register_schema(schema_config)
            app_logger.info(f"文档集合创建成功: {collection.name}")
        except Exception as e:
            app_logger.error(f"创建文档集合失败: {e}")
            raise RuntimeError(f"创建文档集合失败: {e}")

        return collection

    def find_by_id(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """
        根据 ID 查找文档集合
        
        Args:
            collection_id: 集合 ID
            
        Returns:
            文档集合实体或 None
        """
        all_collections = self.find_all()
        for collection in all_collections:
            if collection.id == collection_id:
                return collection
        return None

    def find_by_name(self, name: str) -> Optional[DocumentCollection]:
        """
        根据名称查找文档集合
        
        Args:
            name: 集合名称
            
        Returns:
            文档集合实体或 None
        """
        app_logger.debug(f"查找文档集合: {name}")

        if name not in utility.list_collections():
            return None

        try:
            collection = Collection(name)
            return DocumentCollection(
                id=self._name_to_uuid(name),
                name=name,
                description=collection.description,
                documents=[],
            )
        except Exception as e:
            app_logger.error(f"查找文档集合失败: {e}")
            return None

    def find_all(self) -> List[DocumentCollection]:
        """
        查找所有文档集合
        
        Returns:
            文档集合实体列表
        """
        app_logger.debug("查找所有文档集合")

        collection_names = utility.list_collections()
        collections = []

        for name in collection_names:
            try:
                collection = Collection(name)
                collections.append(DocumentCollection(
                    id=self._name_to_uuid(name),
                    name=name,
                    description=collection.description,
                    documents=[],
                ))
            except Exception as e:
                app_logger.warning(f"获取集合信息失败 [{name}]: {e}")

        return collections

    def delete(self, collection: DocumentCollection) -> None:
        """
        删除文档集合
        
        Args:
            collection: 文档集合实体
        """
        self.delete_by_id(collection.id)

    def delete_by_id(self, collection_id: UUID) -> None:
        """
        根据 ID 删除文档集合
        
        Args:
            collection_id: 集合 ID
        """
        collection = self.find_by_id(collection_id)
        if not collection:
            app_logger.warning(f"文档集合不存在: {collection_id}")
            return

        app_logger.info(f"删除文档集合: {collection.name}")
        try:
            self._collection_creator.delete_collection(collection.name)
            app_logger.info(f"文档集合删除成功: {collection.name}")
        except Exception as e:
            app_logger.error(f"删除文档集合失败: {e}")
            raise RuntimeError(f"删除文档集合失败: {e}")

    def count(self) -> int:
        """
        统计文档集合数量
        
        Returns:
            集合数量
        """
        return len(utility.list_collections())

    def get_document_repository(
        self,
        collection_name: str,
        embedding_function: Optional[any] = None,
    ) -> "LangChainDocumentRepository":
        """
        获取指定集合的文档仓库
        
        Args:
            collection_name: 集合名称
            embedding_function: 嵌入函数
            
        Returns:
            LangChainDocumentRepository 实例
        """
        from infrastructure.persistence.vector.repository.langchain_document_repository_impl import (
            LangChainDocumentRepository,
        )
        
        return LangChainDocumentRepository(
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

    def collection_exists(self, name: str) -> bool:
        """
        检查集合是否存在
        
        Args:
            name: 集合名称
            
        Returns:
            是否存在
        """
        return name in utility.list_collections()

    def get_collection_stats(self, name: str) -> dict:
        """
        获取集合统计信息
        
        Args:
            name: 集合名称
            
        Returns:
            统计信息字典
        """
        if name not in utility.list_collections():
            return {"exists": False}

        try:
            collection = Collection(name)
            return {
                "exists": True,
                "name": name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "schema": str(collection.schema),
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
