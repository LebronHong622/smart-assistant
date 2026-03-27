"""
DocumentCollectionRepository 接口的 Milvus 实现
"""
import uuid
from typing import List, Optional
from uuid import UUID
from pymilvus import utility, Collection
from domain.entity.document.document_collection import DocumentCollection
from domain.repository.document.document_collection_repository import DocumentCollectionRepository
from infrastructure.persistence.vector.milvus_client import milvus_client
from infrastructure.persistence.vector.vector_store import MilvusVectorStore
from infrastructure.persistence.vector.milvus_collections.collection_manager import MilvusCollectionCreator, CollectionSchemaConfig
from infrastructure.core.log import app_logger
from config.settings import settings


class MilvusDocumentCollectionRepository(DocumentCollectionRepository):
    """
    基于 Milvus 的文档集合仓库实现
    """

    def __init__(self):
        self.collection_creator = MilvusCollectionCreator()

    def _name_to_uuid(self, name: str) -> UUID:
        """将集合名称转换为 UUID"""
        return uuid.uuid5(uuid.NAMESPACE_DNS, name)

    def save(self, collection: DocumentCollection) -> DocumentCollection:
        """保存文档集合"""
        app_logger.info(f"保存文档集合: {collection.name}")

        # 检查集合是否已存在
        if collection.name in utility.list_collections():
            app_logger.warning(f"集合已存在: {collection.name}，将使用现有集合")
            return collection

        # 创建默认 Schema
        default_schema = CollectionSchemaConfig(
            collection_name=collection.name,
            description=collection.description or "",
            fields=[
                {
                    "name": "id",
                    "data_type": "VARCHAR",
                    "max_length": 64,
                    "is_primary": True
                },
                {
                    "name": "content",
                    "data_type": "VARCHAR",
                    "max_length": 4096
                },
                {
                    "name": "metadata",
                    "data_type": "VARCHAR",
                    "max_length": 1024
                },
                {
                    "name": "embedding",
                    "data_type": "FLOAT_VECTOR",
                    "dim": settings.milvus.milvus_dimension
                }
            ],
            index={
                "field_name": "embedding",
                "index_type": settings.milvus.milvus_index_type,
                "metric_type": settings.milvus.milvus_metric_type,
                "params": {"nlist": settings.milvus.milvus_n_list}
            }
        )

        # 创建集合
        self.collection_creator.create_collection(default_schema, overwrite=False)

        # 注册 Schema 到 MilvusClient
        milvus_client.register_schema(default_schema)

        return collection

    def find_by_id(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """根据 ID 查找文档集合"""
        # Milvus 集合没有 ID，遍历所有集合查找匹配的 UUID
        all_collections = self.find_all()
        for collection in all_collections:
            if collection.id == collection_id:
                return collection
        return None

    def find_by_name(self, name: str) -> Optional[DocumentCollection]:
        """根据名称查找文档集合"""
        app_logger.debug(f"查找文档集合: {name}")

        if name not in utility.list_collections():
            return None

        # 获取集合信息
        collection = Collection(name)
        info = {
            "name": collection.name,
            "description": collection.description,
            "num_entities": collection.num_entities
        }

        return DocumentCollection(
            id=self._name_to_uuid(name),
            name=name,
            description=info.get("description", ""),
            documents=[]
        )

    def find_all(self) -> List[DocumentCollection]:
        """查找所有文档集合"""
        app_logger.debug("查找所有文档集合")

        collection_names = utility.list_collections()
        collections = []

        for name in collection_names:
            collection = Collection(name)
            collections.append(DocumentCollection(
                id=self._name_to_uuid(name),
                name=name,
                description=collection.description,
                documents=[]
            ))

        return collections

    def delete(self, collection: DocumentCollection) -> None:
        """删除文档集合"""
        self.delete_by_id(collection.id)

    def delete_by_id(self, collection_id: UUID) -> None:
        """根据 ID 删除文档集合"""
        collection = self.find_by_id(collection_id)
        if not collection:
            app_logger.warning(f"文档集合不存在: {collection_id}")
            return

        app_logger.info(f"删除文档集合: {collection.name}")
        self.collection_creator.delete_collection(collection.name)

    def count(self) -> int:
        """统计文档集合数量"""
        return len(utility.list_collections())
