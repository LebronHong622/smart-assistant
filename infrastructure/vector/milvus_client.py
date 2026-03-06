"""
Milvus 向量数据库客户端封装
使用单例模式确保整个应用使用单一 Milvus 连接实例
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from infrastructure.config.settings import settings


class MilvusClient:
    """Milvus 向量数据库客户端"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._connect()
            self._initialized = True

    def _connect(self):
        """连接到 Milvus 服务器"""
        try:
            # 检查是否已连接
            if "default" in connections.list_connections():
                return

            # 使用 URI 连接（支持单机和集群）
            connections.connect(
                alias="default",
                uri=settings.milvus.milvus_uri,
            )

        except Exception as e:
            raise RuntimeError(f"连接 Milvus 服务器失败: {str(e)}")

    def ensure_collection_exists(self, collection_name: str = None) -> Collection:
        """确保集合存在，不存在则创建"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        try:
            # 检查集合是否存在
            if collection_name in Collection.list_collections():
                return Collection(collection_name)

            # 创建集合
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.milvus.milvus_dimension)
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Document embeddings collection"
            )

            collection = Collection(
                name=collection_name,
                schema=schema
            )

            # 创建索引
            index_params = {
                "metric_type": settings.milvus.milvus_metric_type,
                "index_type": settings.milvus.milvus_index_type,
                "params": {"nlist": settings.milvus.milvus_n_list}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )

            return collection

        except Exception as e:
            raise RuntimeError(f"确保 Milvus 集合存在失败: {str(e)}")

    def insert_embeddings(self, documents: list[dict], collection_name: str = None):
        """插入文档嵌入向量"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        collection = self.ensure_collection_exists(collection_name)

        try:
            entities = []
            for doc in documents:
                entities.append([
                    doc["id"],
                    doc["content"],
                    doc["metadata"],
                    doc["embedding"]
                ])

            collection.insert(entities)
            collection.flush()

        except Exception as e:
            raise RuntimeError(f"插入文档嵌入向量失败: {str(e)}")

    def search_embeddings(self, query_embedding: list[float], limit: int = 5, collection_name: str = None) -> list[dict]:
        """搜索相似向量"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        collection = self.ensure_collection_exists(collection_name)
        collection.load()

        try:
            search_params = {
                "metric_type": settings.milvus.milvus_metric_type,
                "params": {"nprobe": 10}
            }

            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=["content", "metadata"]
            )

            # 处理搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    search_results.append({
                        "id": hit.id,
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "distance": hit.distance
                    })

            return search_results

        except Exception as e:
            raise RuntimeError(f"搜索相似向量失败: {str(e)}")

    def get_collection_info(self, collection_name: str = None) -> dict:
        """获取集合信息"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        try:
            if collection_name not in Collection.list_collections():
                return {"exists": False, "description": "Collection does not exist"}

            collection = Collection(collection_name)
            info = {
                "exists": True,
                "name": collection.name,
                "description": collection.description,
                "schema": collection.schema,
                "num_entities": collection.num_entities
            }

            return info

        except Exception as e:
            raise RuntimeError(f"获取集合信息失败: {str(e)}")

    def delete_collection(self, collection_name: str = None):
        """删除集合"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        try:
            if collection_name in Collection.list_collections():
                collection = Collection(collection_name)
                collection.drop()

        except Exception as e:
            raise RuntimeError(f"删除集合失败: {str(e)}")

    def disconnect(self):
        """断开 Milvus 连接"""
        try:
            if "default" in connections.list_connections():
                connections.disconnect("default")
        except Exception as e:
            raise RuntimeError(f"断开 Milvus 连接失败: {str(e)}")


# 全局 Milvus 客户端实例（单例）
milvus_client = MilvusClient()