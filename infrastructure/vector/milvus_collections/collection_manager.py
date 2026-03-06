"""
通过 JSON 配置文件创建 Milvus Collection 的脚本
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from infrastructure.config.settings import settings
from infrastructure.log import app_logger


class FieldDefinition(BaseModel):
    """字段定义"""
    name: str = Field(..., description="字段名称")
    data_type: str = Field(..., description="数据类型: VARCHAR, INT64, FLOAT, DOUBLE, FLOAT_VECTOR, BOOL")
    is_primary: bool = Field(False, description="是否为主键")
    auto_id: bool = Field(False, description="是否自动生成ID")
    max_length: Optional[int] = Field(None, description="VARCHAR 类型的最大长度")
    dim: Optional[int] = Field(None, description="向量维度")
    description: Optional[str] = Field(None, description="字段描述")

    @field_validator("data_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        """验证数据类型"""
        valid_types = {"VARCHAR", "INT64", "FLOAT", "DOUBLE", "FLOAT_VECTOR", "BOOL"}
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
    metric_type: str = Field(..., description="度量类型: L2, IP, COSINE, etc.")
    params: dict = Field(default_factory=dict, description="索引参数")


class CollectionSchemaConfig(BaseModel):
    """Collection Schema 配置"""
    collection_name: str = Field(..., description="Collection 名称")
    description: str = Field("", description="Collection 描述")
    fields: list[FieldDefinition] = Field(..., description="字段列表")
    index: Optional[IndexParams] = Field(None, description="索引配置")


class MilvusCollectionCreator:
    """Milvus Collection 创建器"""

    # 数据类型映射
    DATA_TYPE_MAPPING = {
        "VARCHAR": DataType.VARCHAR,
        "INT64": DataType.INT64,
        "FLOAT": DataType.FLOAT,
        "DOUBLE": DataType.DOUBLE,
        "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
        "BOOL": DataType.BOOL,
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

        # 创建 Collection Schema
        schema = CollectionSchema(
            fields=fields,
            description=config.description
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

    args = parser.parse_args()

    if args.command == "create":
        create_from_json_file(args.json_file, args.overwrite)
    elif args.command == "create-from-content":
        create_from_json_content(args.json_content, args.overwrite)
    elif args.command == "list":
        list_all_collections()
    elif args.command == "info":
        show_collection_info(args.collection_name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
