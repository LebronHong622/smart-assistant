"""
CollectionService 接口实现
"""
from typing import List, Optional
from uuid import UUID
from domain.document.entity.document_collection import DocumentCollection
from domain.document.service.collection_service import CollectionService
from domain.document.repository.document_collection_repository import DocumentCollectionRepository
from domain.shared.ports.vector_store_port import VectorStorePort
from domain.shared.ports.logger_port import LoggerPort


class CollectionServiceImpl(CollectionService):
    """
    集合管理服务实现
    """

    def __init__(
        self,
        collection_repository: DocumentCollectionRepository,
        vector_store: VectorStorePort,
        logger: LoggerPort
    ):
        self.collection_repository = collection_repository
        self.vector_store = vector_store
        self.logger = logger

    def create_collection(self, name: str, description: Optional[str] = None) -> DocumentCollection:
        """创建文档集合"""
        self.logger.info(f"创建文档集合: {name}")

        # 检查集合是否已存在
        existing_collection = self.collection_repository.find_by_name(name)
        if existing_collection:
            raise RuntimeError(f"文档集合已存在: {name}")

        # 创建集合实体
        collection = DocumentCollection(
            name=name,
            description=description or ""
        )

        # 保存集合
        return self.collection_repository.save(collection)

    def delete_collection(self, collection_id: UUID) -> None:
        """删除文档集合"""
        self.logger.info(f"删除文档集合: {collection_id}")
        self.collection_repository.delete_by_id(collection_id)

    def get_collection(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """获取文档集合"""
        return self.collection_repository.find_by_id(collection_id)

    def get_collection_by_name(self, name: str) -> Optional[DocumentCollection]:
        """根据名称获取文档集合"""
        return self.collection_repository.find_by_name(name)

    def list_collections(self) -> List[DocumentCollection]:
        """获取所有文档集合列表"""
        return self.collection_repository.find_all()

    def count_collections(self) -> int:
        """统计文档集合数量"""
        return self.collection_repository.count()

    def get_collection_info(self, collection_id: UUID) -> dict:
        """获取集合详细信息"""
        collection = self.get_collection(collection_id)
        if not collection:
            raise RuntimeError(f"文档集合不存在: {collection_id}")

        # 获取集合的详细信息
        collection_info = self.vector_store.get_collection_info()

        return {
            "id": str(collection.id),
            "name": collection.name,
            "description": collection.description,
            "created_at": collection.created_at,
            "updated_at": collection.updated_at,
            "exists": collection_info.get("exists", False),
            "num_entities": collection_info.get("num_entities", 0),
            "schema": str(collection_info.get("schema", ""))
        }
