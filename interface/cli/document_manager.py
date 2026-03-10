"""
文档管理命令行工具
用于管理 Milvus 向量数据库中的文档
"""

import argparse
import sys
import json
from datetime import datetime
from application.services.document_retrieval_service_impl import MilvusDocumentRetrievalService
from infrastructure.core.log import app_logger


def main():
    app_logger.info("文档管理命令行工具启动")
    parser = argparse.ArgumentParser(
        description="文档管理命令行工具 - 管理 Milvus 向量数据库中的文档"
    )

    subparsers = parser.add_subparsers(title="子命令", dest="subcommand")

    # 上传文档子命令
    upload_parser = subparsers.add_parser("upload", help="上传文档到向量数据库")
    upload_parser.add_argument("-t", "--title", required=True, help="文档标题")
    upload_parser.add_argument("-c", "--content", required=True, help="文档内容")
    upload_parser.add_argument("-p", "--type", default="txt", help="文档类型 (txt/pdf/word/excel/csv)")
    upload_parser.add_argument("-s", "--source", default="upload", help="文档来源 (upload/web/database)")
    upload_parser.add_argument("--collection", help="指定 Collection 名称，默认使用配置中的默认 Collection")
    upload_parser.add_argument("--extra", help="JSON 格式的自定义额外字段，例如 '{\"price\": 99.9, \"category\": \"book\"}'")

    # 检索文档子命令
    retrieve_parser = subparsers.add_parser("retrieve", help="检索相似文档")
    retrieve_parser.add_argument("-q", "--query", required=True, help="查询文本")
    retrieve_parser.add_argument("-l", "--limit", type=int, default=5, help="返回结果数量限制")
    retrieve_parser.add_argument("-s", "--score", type=float, default=0.5, help="相似度分数阈值")
    retrieve_parser.add_argument("--collection", help="指定 Collection 名称，默认使用配置中的默认 Collection")

    # 获取集合信息子命令
    info_parser = subparsers.add_parser("info", help="获取集合信息")
    info_parser.add_argument("--collection", help="指定 Collection 名称，默认使用配置中的默认 Collection")

    # 删除文档子命令
    delete_parser = subparsers.add_parser("delete", help="删除文档")
    delete_parser.add_argument("-i", "--id", required=True, help="文档ID")
    delete_parser.add_argument("--collection", help="指定 Collection 名称，默认使用配置中的默认 Collection")

    # Collection 管理子命令组
    collection_parser = subparsers.add_parser("collection", help="Collection 管理")
    collection_subparsers = collection_parser.add_subparsers(title="Collection 子命令", dest="collection_subcommand")

    # 列出所有 Collection
    list_parser = collection_subparsers.add_parser("list", help="列出所有 Collection")

    # 创建 Collection
    create_parser = collection_subparsers.add_parser("create", help="创建新的 Collection")
    create_parser.add_argument("--schema", required=True, help="JSON 配置文件路径，包含 Collection Schema 定义")
    create_parser.add_argument("--overwrite", action="store_true", help="如果 Collection 已存在则覆盖")

    # 获取 Collection 信息
    collection_info_parser = collection_subparsers.add_parser("info", help="获取指定 Collection 的详细信息")
    collection_info_parser.add_argument("collection_name", help="Collection 名称")

    # 删除 Collection
    delete_collection_parser = collection_subparsers.add_parser("delete", help="删除 Collection")
    delete_collection_parser.add_argument("collection_name", help="Collection 名称")
    delete_collection_parser.add_argument("--force", action="store_true", help="强制删除，无需确认")

    args = parser.parse_args()

    try:
        retrieval_service = MilvusDocumentRetrievalService()

        if args.subcommand == "upload":
            app_logger.info(f"📄 正在上传文档: {args.title}")
            print(f"📄 正在上传文档: {args.title}")
            from domain.document.entity.document import Document
            from domain.document.value_object.document_metadata import DocumentMetadata, DocumentType, DocumentSource
            from uuid import uuid4

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 创建文档元数据
            metadata = DocumentMetadata(
                title=args.title,
                document_type=DocumentType[args.type.upper()],
                source=DocumentSource[args.source.upper()],
                created_at=current_time,
                updated_at=current_time
            )

            # 创建文档实体，包含自定义字段
            doc_data = {
                "id": uuid4(),
                "content": args.content,
                "metadata": metadata.model_dump()
            }

            # 解析额外字段
            if args.extra:
                try:
                    extra_fields = json.loads(args.extra)
                    doc_data.update(extra_fields)
                except json.JSONDecodeError as e:
                    print(f"❌ 额外字段 JSON 格式错误: {str(e)}")
                    return 1

            document = Document(**doc_data)

            # 添加到检索集合
            retrieval_service.add_document_to_collection(document, collection_name=args.collection)
            app_logger.info(f"✅ 文档上传成功，ID: {document.id}")
            print(f"✅ 文档上传成功，ID: {document.id}")

        elif args.subcommand == "retrieve":
            app_logger.info(f"🔍 正在检索文档: {args.query}")
            print(f"🔍 正在检索文档: {args.query}")
            results = retrieval_service.retrieve_similar_documents(
                query=args.query,
                limit=args.limit,
                score_threshold=args.score,
                collection_name=args.collection
            )

            app_logger.info(f"📊 找到 {len(results)} 个相关文档")
            print(f"📊 找到 {len(results)} 个相关文档:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📄 文档ID: {result.document_id}")
                print(f"   📝 内容预览: {result.content[:100]}...")
                print(f"   📊 相似度: {result.similarity_score:.4f}")
                if result.metadata:
                    print(f"   🏷️  标题: {result.metadata.get('title', '未命名')}")

        elif args.subcommand == "info":
            app_logger.info("📊 获取集合信息")
            print("📊 集合信息:")
            from application.services.collection_service_impl import CollectionServiceImpl
            collection_service = CollectionServiceImpl()

            if args.collection:
                collection = collection_service.get_collection_by_name(args.collection)
                if not collection:
                    print(f"❌ 集合不存在: {args.collection}")
                    return 1
                info = collection_service.get_collection_info(collection.id)
            else:
                # 使用默认集合
                from config.settings import settings
                default_collection_name = settings.milvus.milvus_collection_name
                collection = collection_service.get_collection_by_name(default_collection_name)
                if not collection:
                    # 创建默认集合
                    collection = collection_service.create_collection(default_collection_name, "默认文档集合")
                info = collection_service.get_collection_info(collection.id)

            print(f"名称: {info['name']}")
            print(f"描述: {info['description']}")
            print(f"文档数量: {info['num_entities']}")
            print(f"创建时间: {info['created_at']}")
            print(f"更新时间: {info['updated_at']}")

        elif args.subcommand == "delete":
            app_logger.info(f"🗑️  正在删除文档: {args.id}")
            print(f"🗑️  正在删除文档: {args.id}")
            retrieval_service.remove_document_from_collection(args.id, collection_name=args.collection)
            app_logger.info("✅ 文档删除成功")
            print("✅ 文档删除成功")

        elif args.subcommand == "collection":
            from application.services.collection_service_impl import CollectionServiceImpl
            collection_service = CollectionServiceImpl()

            if args.collection_subcommand == "list":
                collections = collection_service.list_collections()
                print(f"\n📋 现有 Collections ({len(collections)}):")
                for col in collections:
                    print(f"   - {col.name}: {col.description} (ID: {col.id})")

            elif args.collection_subcommand == "create":
                # 简化创建，使用默认 schema
                print("⚠️  注意：当前版本使用默认 Schema 创建集合")
                collection_name = input("请输入集合名称: ")
                description = input("请输入集合描述（可选）: ")
                collection = collection_service.create_collection(collection_name, description)
                print(f"\n✅ Collection 创建成功!")
                print(f"   名称: {collection.name}")
                print(f"   描述: {collection.description}")
                print(f"   ID: {collection.id}")

            elif args.collection_subcommand == "info":
                collection = collection_service.get_collection_by_name(args.collection_name)
                if not collection:
                    print(f"\n❌ Collection '{args.collection_name}' 不存在")
                    return 1

                info = collection_service.get_collection_info(collection.id)
                print(f"\n📊 Collection 信息:")
                print(f"   名称: {info['name']}")
                print(f"   描述: {info['description']}")
                print(f"   实体数: {info['num_entities']}")
                print(f"   创建时间: {info['created_at']}")
                print(f"   更新时间: {info['updated_at']}")

            elif args.collection_subcommand == "delete":
                collection = collection_service.get_collection_by_name(args.collection_name)
                if not collection:
                    print(f"\n❌ Collection '{args.collection_name}' 不存在")
                    return 1

                if not args.force:
                    confirm = input(f"\n⚠️  确定要删除 Collection '{args.collection_name}' 吗？此操作不可恢复！(y/N): ")
                    if confirm.lower() != "y":
                        print("\n❌ 已取消删除")
                        return 0

                collection_service.delete_collection(collection.id)
                print(f"\n✅ Collection '{args.collection_name}' 删除成功")

            else:
                collection_parser.print_help()

        else:
            parser.print_help()

    except Exception as e:
        app_logger.error(f"❌ 错误: {str(e)}")
        print(f"❌ 错误: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())