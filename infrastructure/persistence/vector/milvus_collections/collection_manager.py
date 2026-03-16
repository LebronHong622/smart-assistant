"""
通过 JSON 配置文件创建 Milvus Collection 的脚本
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, Function, FunctionType
from config.settings import settings
from infrastructure.core.log import app_logger


class FieldDefinition(BaseModel):
    """字段定义"""
    name: str = Field(..., description="字段名称")
    data_type: str = Field(..., description="数据类型: VARCHAR, INT64, FLOAT, DOUBLE, FLOAT_VECTOR, SPARSE_FLOAT_VECTOR, BOOL")
    is_primary: bool = Field(False, description="是否为主键")
    auto_id: bool = Field(False, description="是否自动生成ID")
    max_length: Optional[int] = Field(None, description="VARCHAR 类型的最大长度")
    dim: Optional[int] = Field(None, description="向量维度")
    enable_analyzer: Optional[bool] = Field(None, description="是否启用分词器，BM25 Function 需要输入字段启用分词")
    description: Optional[str] = Field(None, description="字段描述")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """验证数据类型"""
        valid_types = {"VARCHAR", "INT64", "FLOAT", "DOUBLE", "FLOAT_VECTOR", "SPARSE_FLOAT_VECTOR", "BOOL", "JSON"}
        if v.upper() not in valid_types:
            raise ValueError(f"无效的数据类型: {v}, 支持的类型: {valid_types}")
        return v.upper()

    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: Optional[int], info) -> Optional[int]:
        """VARCHAR 字段必须指定 max_length"""
        if info and info.data.get("data_type") == "VARCHAR" and v is None:
            raise ValueError("VARCHAR 类型字段必须指定 max_length")
        return v

    @field_validator("dim")
    @classmethod
    def validate_dim(cls, v: Optional[int], info) -> Optional[int]:
        """向量字段必须指定 dim"""
        if info and info.data.get("data_type") == "FLOAT_VECTOR" and v is None:
            raise ValueError("FLOAT_VECTOR 类型字段必须指定 dim")
        return v


class IndexParams(BaseModel):
    """索引参数"""
    field_name: str = Field(..., description="索引字段名称")
    index_type: str = Field(..., description="索引类型: IVF_FLAT, IVF_SQ8, HNSW, FLAT, etc.")
    metric_type: str = Field(..., description="度量类型: L2, IP, COSINE, BM25, etc.")
    params: dict = Field(default_factory=dict, description="索引参数")


class BM25FunctionConfig(BaseModel):
    """BM25 函数配置"""
    enable: bool = Field(True, description="是否启用 Milvus 内置 BM25 函数，默认为 True")
    input_field: str = Field("content", description="BM25 函数输入字段（原始文本字段），默认为 content")
    output_field: str = Field("sparse_embedding", description="BM25 函数输出字段（稀疏向量字段），必须是 SPARSE_FLOAT_VECTOR 类型，默认为 sparse_embedding")
    k1: float = Field(1.2, description="BM25 参数 k1，默认 1.2")
    b: float = Field(0.75, description="BM25 参数 b，默认 0.75")


class CollectionSchemaConfig(BaseModel):
    """Collection Schema 配置"""
    collection_name: str = Field(..., description="Collection 名称")
    description: str = Field("", description="Collection 描述")
    fields: list[FieldDefinition] = Field(..., description="字段列表")
    index: Optional[IndexParams] = Field(None, description="索引配置")
    sparse_index: Optional[IndexParams] = Field(None, description="稀疏向量索引配置")
    enable_dynamic_field: bool = Field(True, description="是否启用动态字段")
    bm25_function: Optional[BM25FunctionConfig] = Field(None, description="BM25 函数配置，启用内置 BM25 时需要配置")


class MilvusCollectionCreator:
    """Milvus Collection 创建器"""

    # 数据类型映射
    DATA_TYPE_MAPPING = {
        "VARCHAR": DataType.VARCHAR,
        "INT64": DataType.INT64,
        "FLOAT": DataType.FLOAT,
        "DOUBLE": DataType.DOUBLE,
        "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
        "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
        "BOOL": DataType.BOOL,
        "JSON": DataType.JSON,
    }

    def __init__(self, uri: Optional[str] = None):
        """初始化创建器
        
        Args:
            uri: Milvus 连接 URI,默认使用配置文件中的值
        """
        self.uri = uri or settings.milvus.milvus_uri
        self._connect()

    def _connect(self):
        """连接到 Milvus"""
        app_logger.info(f"正在连接到 Milvus: {self.uri}")
        try:
            # 检查是否已连接
            if "default" in connections.list_connections():
                app_logger.debug("Milvus 连接已存在")
                return

            connections.connect(alias="default", uri=self.uri)
            app_logger.info("Milvus 连接成功")
        except Exception as e:
            app_logger.error(f"连接 Milvus 失败: {e}")
            raise RuntimeError(f"连接 Milvus 失败: {e}")

    def load_schema_from_file(self, file_path: str) -> CollectionSchemaConfig:
        """从 JSON 文件加载 Schema 配置"""
        app_logger.info(f"正在加载 Schema 配置文件: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CollectionSchemaConfig(**data)
        except FileNotFoundError:
            app_logger.error(f"文件不存在: {file_path}")
            raise
        except json.JSONDecodeError as e:
            app_logger.error(f"JSON 解析失败: {e}")
            raise
        except Exception as e:
            app_logger.error(f"加载配置文件失败: {e}")
            raise

    def load_schema_from_dict(self, data: dict) -> CollectionSchemaConfig:
        """从字典加载 Schema 配置"""
        app_logger.info("正在从字典加载 Schema 配置")
        return CollectionSchemaConfig(**data)

    def validate_schema(self, config: CollectionSchemaConfig):
        """验证 Schema 配置"""
        app_logger.info("正在验证 Schema 配置")

        # 自动推断 BM25 Function 配置
        if config.sparse_index is not None and config.bm25_function is None:
            if config.sparse_index.metric_type == "BM25":
                # 检查是否存在 content(VARCHAR) 和 sparse_embedding(SPARSE_FLOAT_VECTOR)
                has_content = any(f.name == "content" and f.data_type == "VARCHAR" for f in config.fields)
                has_sparse = any(f.name == "sparse_embedding" and f.data_type == "SPARSE_FLOAT_VECTOR" for f in config.fields)
                if has_content and has_sparse:
                    # 从 sparse_index.params 读取参数
                    k1 = 1.2
                    b = 0.75
                    if "bm25_k1" in config.sparse_index.params:
                        k1 = float(config.sparse_index.params["bm25_k1"])
                    if "bm25_b" in config.sparse_index.params:
                        b = float(config.sparse_index.params["bm25_b"])
                    config.bm25_function = BM25FunctionConfig(
                        enable=True,
                        input_field="content",
                        output_field="sparse_embedding",
                        k1=k1,
                        b=b
                    )
                    app_logger.info(f"自动启用 BM25 Function: input=content, output=sparse_embedding, k1={k1}, b={b}")

        # 检查主键
        primary_keys = [f for f in config.fields if f.is_primary]
        if len(primary_keys) == 0:
            raise ValueError("必须指定一个主键字段")
        if len(primary_keys) > 1:
            raise ValueError("只能有一个主键字段")

        # 检查向量字段
        vector_fields = [f for f in config.fields if f.data_type == "FLOAT_VECTOR"]
        if len(vector_fields) == 0:
            raise ValueError("至少需要一个 FLOAT_VECTOR 字段")

        # 如果配置了索引,检查索引字段是否存在
        if config.index:
            index_field_names = [f.name for f in config.fields]
            if config.index.field_name not in index_field_names:
                raise ValueError(f"索引字段 '{config.index.field_name}' 不存在")

        # 如果配置了稀疏索引,检查索引字段是否存在
        if config.sparse_index:
            index_field_names = [f.name for f in config.fields]
            if config.sparse_index.field_name not in index_field_names:
                raise ValueError(f"稀疏索引字段 '{config.sparse_index.field_name}' 不存在")

        # 验证 BM25 Function 配置
        if config.bm25_function and config.bm25_function.enable:
            # 必须有 sparse_index
            if not config.sparse_index:
                raise ValueError("启用 BM25 Function 必须配置 sparse_index")

            # 检查 input_field
            input_field_exists = any(f.name == config.bm25_function.input_field for f in config.fields)
            if not input_field_exists:
                raise ValueError(f"BM25 Function input_field '{config.bm25_function.input_field}' 不存在")

            # 检查 input_field 类型必须是 VARCHAR
            input_field = next(f for f in config.fields if f.name == config.bm25_function.input_field)
            if input_field.data_type != "VARCHAR":
                raise ValueError(f"BM25 Function input_field '{config.bm25_function.input_field}' 必须是 VARCHAR 类型")

            # Milvus 要求 BM25 input field must have enable_analyzer=True
            if input_field.enable_analyzer is None:
                # Automatically set enable_analyzer=True
                input_field.enable_analyzer = True
                app_logger.info(f"已自动为 BM25 Function 输入字段 '{config.bm25_function.input_field}' 设置 enable_analyzer=True")
            elif not input_field.enable_analyzer:
                app_logger.warning(f"BM25 Function 输入字段 '{config.bm25_function.input_field}' 的 enable_analyzer 应为 True，当前为 False")

            # 检查 output_field
            output_field_exists = any(f.name == config.bm25_function.output_field for f in config.fields)
            if not output_field_exists:
                raise ValueError(f"BM25 Function output_field '{config.bm25_function.output_field}' 不存在")

            # 检查 output_field 类型必须是 SPARSE_FLOAT_VECTOR
            output_field = next(f for f in config.fields if f.name == config.bm25_function.output_field)
            if output_field.data_type != "SPARSE_FLOAT_VECTOR":
                raise ValueError(f"BM25 Function output_field '{config.bm25_function.output_field}' 必须是 SPARSE_FLOAT_VECTOR 类型")

            # 检查 sparse_index 的 metric_type 必须是 BM25
            if config.sparse_index.metric_type != "BM25":
                app_logger.warning(f"Milvus 要求 BM25 Function 的稀疏索引 metric_type 必须是 BM25，当前是 {config.sparse_index.metric_type}")

        app_logger.info("Schema 验证通过")

    def _create_field_schema(self, field_def: FieldDefinition) -> FieldSchema:
        """创建字段 Schema"""
        data_type = self.DATA_TYPE_MAPPING[field_def.data_type]

        # 构建字段参数
        params = {"name": field_def.name, "dtype": data_type}

        if field_def.is_primary:
            params["is_primary"] = True
        if field_def.auto_id:
            params["auto_id"] = True
        if field_def.max_length is not None:
            params["max_length"] = field_def.max_length
        if field_def.dim is not None:
            params["dim"] = field_def.dim
        if field_def.enable_analyzer is not None:
            params["enable_analyzer"] = field_def.enable_analyzer
        if field_def.description:
            params["description"] = field_def.description

        return FieldSchema(**params)

    def create_collection(self, config: CollectionSchemaConfig, overwrite: bool = False) -> Collection:
        """创建 Collection
        
        Args:
            config: Collection Schema 配置
            overwrite: 如果 Collection 已存在是否覆盖
            
        Returns:
            创建的 Collection 对象
        """
        app_logger.info(f"正在创建 Collection: {config.collection_name}")

        # 验证配置
        self.validate_schema(config)

        # 检查 Collection 是否已存在
        if config.collection_name in utility.list_collections():
            if not overwrite:
                raise RuntimeError(f"Collection '{config.collection_name}' 已存在")
            else:
                app_logger.warning(f"Collection '{config.collection_name}' 已存在,准备删除")
                collection = Collection(config.collection_name)
                collection.drop()
                app_logger.info(f"已删除旧 Collection: {config.collection_name}")

        # 创建字段 Schema
        fields = [self._create_field_schema(f) for f in config.fields]

        # 准备 functions
        functions = []
        if config.bm25_function and config.bm25_function.enable:
            # 从 sparse_index.params 获取 bm25 参数（如果存在）覆盖配置
            k1 = config.bm25_function.k1
            b = config.bm25_function.b
            if config.sparse_index and "bm25_k1" in config.sparse_index.params:
                k1 = float(config.sparse_index.params["bm25_k1"])
            if config.sparse_index and "bm25_b" in config.sparse_index.params:
                b = float(config.sparse_index.params["bm25_b"])

            # Milvus BM25 function doesn't accept params directly - parameters are stored in the index
            bm25_function = Function(
                name="bm25_function",
                function_type=FunctionType.BM25,
                input_field_names=config.bm25_function.input_field,
                output_field_names=config.bm25_function.output_field,
                params={}
            )
            functions.append(bm25_function)
            app_logger.info(f"BM25 Function 已添加: input={config.bm25_function.input_field}, output={config.bm25_function.output_field}")

        # 创建 Collection Schema
        schema = CollectionSchema(
            fields=fields,
            description=config.description,
            enable_dynamic_field=config.enable_dynamic_field,
            functions=functions if functions else None
        )

        # 创建 Collection
        collection = Collection(name=config.collection_name, schema=schema)
        app_logger.info(f"Collection '{config.collection_name}' 创建成功")

        # 创建索引
        if config.index:
            index_params = {
                "metric_type": config.index.metric_type,
                "index_type": config.index.index_type,
                "params": config.index.params
            }
            collection.create_index(
                field_name=config.index.field_name,
                index_params=index_params
            )
            app_logger.info(f"索引创建成功: {config.index.field_name}")

        # 创建稀疏向量索引
        if config.sparse_index:
            index_params = {
                "metric_type": config.sparse_index.metric_type,
                "index_type": config.sparse_index.index_type,
                "params": config.sparse_index.params
            }
            collection.create_index(
                field_name=config.sparse_index.field_name,
                index_params=index_params
            )
            app_logger.info(f"稀疏向量索引创建成功: {config.sparse_index.field_name}")

        return collection

    def list_collections(self) -> list[str]:
        """列出所有 Collection"""
        return utility.list_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        """获取 Collection 信息"""
        if collection_name not in utility.list_collections():
            return {"exists": False}

        collection = Collection(collection_name)
        return {
            "exists": True,
            "name": collection.name,
            "description": collection.description,
            "num_entities": collection.num_entities
        }

    def delete_collection(self, collection_name: str) -> bool:
        """删除 Collection

        Args:
            collection_name: Collection 名称

        Returns:
            删除成功返回 True，否则返回 False
        """
        app_logger.info(f"正在删除 Collection: {collection_name}")

        if collection_name not in utility.list_collections():
            app_logger.warning(f"Collection '{collection_name}' 不存在")
            return False

        try:
            collection = Collection(collection_name)
            collection.drop()
            app_logger.info(f"Collection '{collection_name}' 删除成功")
            return True
        except Exception as e:
            app_logger.error(f"删除 Collection '{collection_name}' 失败: {e}")
            raise RuntimeError(f"删除 Collection 失败: {e}")

    def update_collection(self, config: CollectionSchemaConfig, migrate_data: bool = True) -> Collection:
        """更新 Collection

        注意: Milvus 不支持直接修改已存在 Collection 的 Schema，更新操作会:
        1. 创建新的 Collection
        2. （可选）迁移旧 Collection 的数据到新 Collection
        3. 删除旧 Collection
        4. 将新 Collection 重命名为旧名称

        Args:
            config: 新的 Collection Schema 配置
            migrate_data: 是否迁移旧数据

        Returns:
            更新后的 Collection 对象
        """
        collection_name = config.collection_name
        app_logger.info(f"正在更新 Collection: {collection_name}")

        # 检查 Collection 是否存在
        if collection_name not in utility.list_collections():
            raise RuntimeError(f"Collection '{collection_name}' 不存在，无法更新")

        old_collection = Collection(collection_name)
        # 不要加载旧 collection，避免索引问题

        # 获取旧 collection 的字段列表
        old_field_names = [f.name for f in old_collection.schema.fields]
        new_field_names = [f.name for f in config.fields]

        # 临时名称
        temp_name = f"{collection_name}_temp_{int(time.time())}"

        try:
            # 1. 创建临时 Collection
            temp_config = config.model_copy(update={"collection_name": temp_name})
            temp_collection = self.create_collection(temp_config, overwrite=False)
            app_logger.info(f"临时 Collection '{temp_name}' 创建成功")

            # 2. 迁移数据
            if migrate_data and old_collection.num_entities > 0:
                app_logger.info(f"正在迁移数据: {old_collection.num_entities} 条记录")

                # 获取旧 collection 和新 collection 的公共字段
                common_fields = [f for f in old_field_names if f in new_field_names]
                app_logger.info(f"迁移字段: {common_fields}")

                # 分页查询所有数据
                batch_size = 1000
                offset = 0

                while offset < old_collection.num_entities:
                    # 查询数据
                    query_iterator = old_collection.query(
                        expr="",
                        output_fields=common_fields,
                        offset=offset,
                        limit=batch_size
                    )

                    if not query_iterator:
                        break

                    # 转换数据格式 - 只迁移公共字段
                    insert_data = []
                    for field in temp_collection.schema.fields:
                        if field.name in common_fields:
                            field_data = [item[field.name] for item in query_iterator]
                            insert_data.append(field_data)
                        else:
                            # 新字段填充默认值
                            num_records = len(query_iterator)
                            if field.dtype == DataType.SPARSE_FLOAT_VECTOR:
                                # 稀疏向量填充空字典
                                field_data = [{} for _ in range(num_records)]
                            elif field.dtype == DataType.FLOAT_VECTOR:
                                # 稠密向量填充零向量
                                dim = field.dim if hasattr(field, 'dim') else 768
                                field_data = [[0.0] * dim for _ in range(num_records)]
                            elif field.dtype in [DataType.INT64]:
                                field_data = [0 for _ in range(num_records)]
                            elif field.dtype in [DataType.FLOAT, DataType.DOUBLE]:
                                field_data = [0.0 for _ in range(num_records)]
                            else:
                                field_data = ["" for _ in range(num_records)]
                            insert_data.append(field_data)

                    # 插入到临时 Collection
                    temp_collection.insert(insert_data)
                    offset += batch_size

                    app_logger.debug(f"已迁移 {offset}/{old_collection.num_entities} 条记录")

                temp_collection.flush()
                app_logger.info(f"数据迁移完成，共迁移 {temp_collection.num_entities} 条记录")

            # 3. 删除旧 Collection
            old_collection.drop()
            app_logger.info(f"旧 Collection '{collection_name}' 已删除")

            # 4. 将临时 Collection 重命名为原名称
            utility.rename_collection(temp_name, collection_name)
            app_logger.info(f"临时 Collection 已重命名为 '{collection_name}'")

            # 获取更新后的 Collection
            updated_collection = Collection(collection_name)

            # 加载 Collection
            updated_collection.load()
            app_logger.info(f"Collection '{collection_name}' 更新成功")

            return updated_collection

        except Exception as e:
            # 清理临时资源
            if temp_name in utility.list_collections():
                temp_collection = Collection(temp_name)
                temp_collection.drop()
                app_logger.warning(f"清理临时 Collection '{temp_name}'")

            app_logger.error(f"更新 Collection '{collection_name}' 失败: {e}")
            raise RuntimeError(f"更新 Collection 失败: {e}")

    def disconnect(self):
        """断开连接"""
        if "default" in connections.list_connections():
            connections.disconnect("default")
            app_logger.info("已断开 Milvus 连接")


def create_from_json_file(json_file_path: str, overwrite: bool = False):
    """从 JSON 文件创建 Collection"""
    creator = MilvusCollectionCreator()

    try:
        # 加载配置
        config = creator.load_schema_from_file(json_file_path)

        # 创建 Collection
        collection = creator.create_collection(config, overwrite=overwrite)

        print(f"\n✅ Collection 创建成功!")
        print(f"   名称: {collection.name}")
        print(f"   描述: {collection.description}")
        print(f"   实体数: {collection.num_entities}")

        return collection

    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def create_from_json_content(json_content: str, overwrite: bool = False):
    """从 JSON 内容创建 Collection"""
    creator = MilvusCollectionCreator()

    try:
        # 解析 JSON
        data = json.loads(json_content)
        config = creator.load_schema_from_dict(data)

        # 创建 Collection
        collection = creator.create_collection(config, overwrite=overwrite)

        print(f"\n✅ Collection 创建成功!")
        print(f"   名称: {collection.name}")
        print(f"   描述: {collection.description}")
        print(f"   实体数: {collection.num_entities}")

        return collection

    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def list_all_collections():
    """列出所有 Collection"""
    creator = MilvusCollectionCreator()

    try:
        collections = creator.list_collections()
        print(f"\n📋 现有 Collections ({len(collections)}):")
        for name in collections:
            info = creator.get_collection_info(name)
            print(f"   - {name}: {info.get('num_entities', 0)} 个实体")
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def show_collection_info(collection_name: str):
    """显示 Collection 详细信息"""
    creator = MilvusCollectionCreator()

    try:
        info = creator.get_collection_info(collection_name)
        if not info.get("exists"):
            print(f"\n❌ Collection '{collection_name}' 不存在")
            sys.exit(1)

        collection = Collection(collection_name)
        print(f"\n📊 Collection 信息:")
        print(f"   名称: {collection.name}")
        print(f"   描述: {collection.description}")
        print(f"   实体数: {collection.num_entities}")
        print(f"\n   字段:")
        for field in collection.schema.fields:
            print(f"   - {field.name}: {field.dtype}")
    except Exception as e:
        print(f"\n❌ 查询失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def delete_collection(collection_name: str, force: bool = False):
    """删除 Collection"""
    creator = MilvusCollectionCreator()

    try:
        # 确认删除
        if not force:
            confirm = input(f"\n⚠️  确定要删除 Collection '{collection_name}' 吗？此操作不可恢复！(y/N): ")
            if confirm.lower() != "y":
                print("\n❌ 已取消删除")
                sys.exit(0)

        success = creator.delete_collection(collection_name)
        if success:
            print(f"\n✅ Collection '{collection_name}' 删除成功")
        else:
            print(f"\n❌ Collection '{collection_name}' 不存在")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 删除失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def update_collection(json_file_path: str, migrate_data: bool = True):
    """更新 Collection"""
    creator = MilvusCollectionCreator()

    try:
        # 加载配置
        config = creator.load_schema_from_file(json_file_path)

        # 确认更新
        confirm = input(f"\n⚠️  确定要更新 Collection '{config.collection_name}' 吗？此操作会覆盖现有 Schema！(y/N): ")
        if confirm.lower() != "y":
            print("\n❌ 已取消更新")
            sys.exit(0)

        # 更新 Collection
        collection = creator.update_collection(config, migrate_data=migrate_data)

        print(f"\n✅ Collection 更新成功!")
        print(f"   名称: {collection.name}")
        print(f"   描述: {collection.description}")
        print(f"   实体数: {collection.num_entities}")

        return collection
    except Exception as e:
        print(f"\n❌ 更新失败: {e}")
        sys.exit(1)
    finally:
        creator.disconnect()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus Collection 管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 创建 Collection 命令
    create_parser = subparsers.add_parser("create", help="创建 Collection")
    create_parser.add_argument("json_file", help="JSON 配置文件路径")
    create_parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 Collection")

    # 从 JSON 内容创建命令
    create_content_parser = subparsers.add_parser("create-from-content", help="从 JSON 内容创建 Collection")
    create_content_parser.add_argument("json_content", help="JSON 配置内容")
    create_content_parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 Collection")

    # 列出所有 Collections
    list_parser = subparsers.add_parser("list", help="列出所有 Collections")

    # 显示 Collection 信息
    info_parser = subparsers.add_parser("info", help="显示 Collection 信息")
    info_parser.add_argument("collection_name", help="Collection 名称")

    # 删除 Collection
    delete_parser = subparsers.add_parser("delete", help="删除 Collection")
    delete_parser.add_argument("collection_name", help="Collection 名称")
    delete_parser.add_argument("--force", action="store_true", help="强制删除，无需确认")

    # 更新 Collection
    update_parser = subparsers.add_parser("update", help="更新 Collection")
    update_parser.add_argument("json_file", help="JSON 配置文件路径")
    update_parser.add_argument("--no-migrate", action="store_true", help="不迁移旧数据")

    args = parser.parse_args()

    if args.command == "create":
        create_from_json_file(args.json_file, args.overwrite)
    elif args.command == "create-from-content":
        create_from_json_content(args.json_content, args.overwrite)
    elif args.command == "list":
        list_all_collections()
    elif args.command == "info":
        show_collection_info(args.collection_name)
    elif args.command == "delete":
        delete_collection(args.collection_name, args.force)
    elif args.command == "update":
        update_collection(args.json_file, migrate_data=not args.no_migrate)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
