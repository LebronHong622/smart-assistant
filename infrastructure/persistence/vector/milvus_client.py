"""
Milvus 向量数据库客户端封装
使用单例模式确保整个应用使用单一 Milvus 连接实例
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config.settings import settings
from infrastructure.core.log import app_logger
from infrastructure.persistence.vector.milvus_collections.collection_manager import CollectionSchemaConfig, MilvusCollectionCreator


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
            self._schema_registry = {}
            self._collection_creator = MilvusCollectionCreator()
            self._initialized = True

    def register_schema(self, schema_config: CollectionSchemaConfig) -> None:
        """注册 Collection Schema"""
        self._schema_registry[schema_config.collection_name] = schema_config
        app_logger.info(f"已注册 Collection Schema: {schema_config.collection_name}")

    def list_collections(self) -> list[str]:
        """列出所有 Collection"""
        return utility.list_collections()

    def _connect(self):
        """连接到 Milvus 服务器"""
        app_logger.info("正在连接到 Milvus 服务器")
        try:
            # 检查是否已连接
            if "default" in connections.list_connections():
                app_logger.debug("Milvus 连接已存在，跳过连接过程")
                return

            # 使用 URI 连接（支持单机和集群）
            app_logger.debug(f"使用 URI 连接 Milvus: {settings.milvus.milvus_uri}")
            connections.connect(
                alias="default",
                uri=settings.milvus.milvus_uri,
            )

            app_logger.info("Milvus 连接成功")

        except Exception as e:
            app_logger.error(f"连接 Milvus 服务器失败: {str(e)}")
            raise RuntimeError(f"连接 Milvus 服务器失败: {str(e)}")

    def ensure_collection_exists(self, collection_name: str = None) -> Collection:
        """确保集合存在，不存在则创建"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        app_logger.info(f"正在检查 Milvus 集合是否存在: {collection_name}")

        try:
            # 检查集合是否存在
            if collection_name in utility.list_collections():
                app_logger.debug(f"Milvus 集合已存在: {collection_name}")
                return Collection(collection_name)

            app_logger.info(f"正在创建 Milvus 集合: {collection_name}")

            # 检查是否有注册的 Schema
            if collection_name in self._schema_registry:
                app_logger.debug(f"使用已注册的 Schema 创建集合: {collection_name}")
                schema_config = self._schema_registry[collection_name]
                return self._collection_creator.create_collection(schema_config)
            else:
                # 使用默认 Schema
                app_logger.debug(f"使用默认 Schema 创建集合: {collection_name}")
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

                app_logger.info(f"Milvus 集合创建成功: {collection_name}")

                return collection

        except Exception as e:
            app_logger.error(f"确保 Milvus 集合存在失败: {str(e)}")
            raise RuntimeError(f"确保 Milvus 集合存在失败: {str(e)}")

    def insert_embeddings(self, documents: list[dict], collection_name: str = None):
        """插入文档嵌入向量"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        collection = self.ensure_collection_exists(collection_name)

        app_logger.info(f"正在插入 {len(documents)} 个文档嵌入向量到 Milvus 集合: {collection_name}")

        try:
            # 获取集合字段列表
            field_names = [field.name for field in collection.schema.fields]
            app_logger.info(f"Collection 字段列表: {field_names}")
            app_logger.info(f"文档字段: {list(documents[0].keys())}")

            # 检查是否有 auto_id 字段，如果有则跳过
            has_auto_id = False
            for field in collection.schema.fields:
                if field.is_primary and field.auto_id:
                    has_auto_id = True
                    break

            # 动态构建实体列表，按照 Schema 字段顺序排列
            entities = []
            for field in field_names:
                # 跳过 auto_id 字段
                if has_auto_id and field == "id":
                    app_logger.info(f"跳过 auto_id 字段: {field}")
                    continue

                field_data = []
                for doc in documents:
                    value = doc.get(field)
                    # 处理稀疏向量字段：如果为 None，提供空稀疏向量
                    if value is None and field == "sparse_embedding":
                        # 提供空的稀疏向量字典
                        value = {}
                    field_data.append(value)
                entities.append(field_data)

            app_logger.info(f"构建的实体数: {len(entities)}")
            collection.insert(entities)
            collection.flush()

            app_logger.info(f"文档嵌入向量插入成功，集合: {collection_name}")

        except Exception as e:
            app_logger.error(f"插入文档嵌入向量失败: {str(e)}")
            raise RuntimeError(f"插入文档嵌入向量失败: {str(e)}")

    def search_embeddings(self, query_embedding: list[float], limit: int = 5, collection_name: str = None, anns_field: str = "embedding") -> list[dict]:
        """搜索相似向量"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        collection = self.ensure_collection_exists(collection_name)
        collection.load()

        app_logger.info(f"正在 Milvus 集合中搜索相似向量: {collection_name}, 限制返回 {limit} 个结果")

        try:
            # 获取所有非向量字段作为输出字段
            output_fields = [field.name for field in collection.schema.fields
                           if field.dtype not in [DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR]]
            app_logger.info(f"输出字段: {output_fields}")

            search_params = {
                "metric_type": settings.milvus.milvus_metric_type,
                "params": {"nprobe": 10}
            }

            results = collection.search(
                data=[query_embedding],
                anns_field=anns_field,
                param=search_params,
                limit=limit,
                expr=None,
                output_fields=output_fields
            )

            # 处理搜索结果，动态包含所有输出字段
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": str(hit.id),  # 将整数 ID 转换为字符串
                        "distance": hit.distance
                    }
                    # 添加所有输出字段
                    for field in output_fields:
                        if field != "id":
                            result[field] = hit.entity.get(field)
                    search_results.append(result)

            app_logger.debug(f"搜索完成，找到 {len(search_results)} 个相似文档")

            return search_results

        except Exception as e:
            app_logger.error(f"搜索相似向量失败: {str(e)}")
            raise RuntimeError(f"搜索相似向量失败: {str(e)}")

    def get_collection_info(self, collection_name: str = None) -> dict:
        """获取集合信息"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        app_logger.info(f"正在获取 Milvus 集合信息: {collection_name}")

        try:
            if collection_name not in Collection.list_collections():
                app_logger.debug(f"Milvus 集合不存在: {collection_name}")
                return {"exists": False, "description": "Collection does not exist"}

            collection = Collection(collection_name)
            info = {
                "exists": True,
                "name": collection.name,
                "description": collection.description,
                "schema": collection.schema,
                "num_entities": collection.num_entities
            }

            app_logger.debug(f"Milvus 集合信息: {info}")

            return info

        except Exception as e:
            app_logger.error(f"获取集合信息失败: {str(e)}")
            raise RuntimeError(f"获取集合信息失败: {str(e)}")

    def delete_collection(self, collection_name: str = None):
        """删除集合"""
        if collection_name is None:
            collection_name = settings.milvus.milvus_collection_name

        app_logger.info(f"正在删除 Milvus 集合: {collection_name}")

        try:
            if collection_name in Collection.list_collections():
                collection = Collection(collection_name)
                collection.drop()
                app_logger.info(f"Milvus 集合删除成功: {collection_name}")
            else:
                app_logger.warning(f"Milvus 集合不存在，无需删除: {collection_name}")

        except Exception as e:
            app_logger.error(f"删除集合失败: {str(e)}")
            raise RuntimeError(f"删除集合失败: {str(e)}")

    def disconnect(self):
        """断开 Milvus 连接"""
        app_logger.info("正在断开 Milvus 连接")

        try:
            if "default" in connections.list_connections():
                connections.disconnect("default")
                app_logger.info("Milvus 连接断开成功")
            else:
                app_logger.warning("Milvus 连接已不存在")

        except Exception as e:
            app_logger.error(f"断开 Milvus 连接失败: {str(e)}")
            raise RuntimeError(f"断开 Milvus 连接失败: {str(e)}")


# 全局 Milvus 客户端实例（单例）
milvus_client = MilvusClient()