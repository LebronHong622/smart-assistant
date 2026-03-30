"""
集合管理应用服务
负责文档集合管理流程的协调和编排
"""

from typing import List, Optional
from uuid import UUID
from domain.entity.document.document_collection import DocumentCollection
from domain.service.document.collection_service import CollectionService
from domain.repository.document.document_collection_repository import DocumentCollectionRepository
from domain.shared.ports.vector_store_port import VectorStorePort
from domain.shared.ports.logger_port import LoggerPort


class CollectionManagementService(CollectionService):
    """
    集合管理应用服务
    协调集合管理流程，处理基础设施和领域服务之间的交互
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
        saved_collection = self.collection_repository.save(collection)
        self.logger.info(f"文档集合创建成功: ID={saved_collection.id}")
        return saved_collection

    def delete_collection(self, collection_id: UUID) -> None:
        """删除文档集合"""
        self.logger.info(f"删除文档集合: {collection_id}")
        self.collection_repository.delete_by_id(collection_id)
        self.logger.info(f"文档集合删除成功: {collection_id}")

    def get_collection(self, collection_id: UUID) -> Optional[DocumentCollection]:
        """获取文档集合"""
        return self.collection_repository.find_by_id(collection_id)

    def get_collection_by_name(self, name: str) -> Optional[DocumentCollection]:
        """根据名称获取文档集合"""
        return self.collection_repository.find_by_name(name)

    def list_collections(self) -> List[DocumentCollection]:
        """获取所有文档集合列表"""
        collections = self.collection_repository.find_all()
        self.logger.debug(f"获取到 {len(collections)} 个文档集合")
        return collections

    def count_collections(self) -> int:
        """统计文档集合数量"""
        count = self.collection_repository.count()
        self.logger.debug(f"文档集合总数: {count}")
        return count

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

    def get_active_collections(self) -> List[DocumentCollection]:
        """
        获取活跃的文档集合（包含文档的集合）

        Returns:
            活跃集合列表
        """
        all_collections = self.list_collections()
        active_collections = []

        for collection in all_collections:
            try:
                # 检查集合中是否有文档
                collection_info = self.get_collection_info(collection.id)
                if collection_info.get("num_entities", 0) > 0:
                    active_collections.append(collection)
            except Exception as e:
                self.logger.warning(f"检查集合活跃状态失败: {collection.name}, 错误: {str(e)}")

        return active_collections

    def search_collections_by_name(self, name_partial: str) -> List[DocumentCollection]:
        """
        根据名称部分匹配搜索集合

        Args:
            name_partial: 名称部分匹配字符串

        Returns:
            匹配的集合列表
        """
        all_collections = self.list_collections()
        matched_collections = [
            collection for collection in all_collections
            if name_partial.lower() in collection.name.lower()
        ]
        return matched_collections
