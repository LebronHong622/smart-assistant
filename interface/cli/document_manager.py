"""
文档管理命令行工具
用于管理 Milvus 向量数据库中的文档
"""

import argparse
import sys
import json
from datetime import datetime
from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
from application.services.document.rag_processing_service_impl import RAGProcessingServiceFactoryImpl
from application.services.document.rag_processing_service import RAGProcessingService
from infrastructure.persistence.vector.repository.langchain_document_repository_impl import LangChainDocumentRepository
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

    # 摄入文件子命令
    ingest_file_parser = subparsers.add_parser("ingest-file", help="从文件路径摄入文档到向量数据库")
    ingest_file_parser.add_argument("-p", "--path", required=True, help="文件路径")
    ingest_file_parser.add_argument("-t", "--type", help="加载器类型 (pdf/txt/md/csv等)，不指定则自动根据扩展名推断")
    ingest_file_parser.add_argument("-d", "--domain", default="default", help="业务领域，默认 default")
    ingest_file_parser.add_argument("-c", "--collection", help="指定 Collection 名称，默认使用领域默认集合")

    # 摄入目录子命令
    ingest_dir_parser = subparsers.add_parser("ingest-dir", help="从目录批量摄入文档到向量数据库")
    ingest_dir_parser.add_argument("-p", "--path", required=True, help="目录路径")
    ingest_dir_parser.add_argument("-g", "--pattern", default="**/*", help="glob 文件匹配模式，默认 **/* 匹配所有文件")
    ingest_dir_parser.add_argument("-t", "--type", help="加载器类型，不指定则自动根据扩展名推断")
    ingest_dir_parser.add_argument("-d", "--domain", default="default", help="业务领域，默认 default")
    ingest_dir_parser.add_argument("-c", "--collection", help="指定 Collection 名称，默认使用领域默认集合")

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
        if args.subcommand == "upload":
            # Lazy import and create retrieval_service when needed
            from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
            from infrastructure.persistence.vector.milvus_client import MilvusClient
            from infrastructure.model.embeddings_manager import EmbeddingsManager
            from config.settings import settings

            milvus_client = MilvusClient()
            embeddings_manager = EmbeddingsManager()
            retrieval_service = MilvusDocumentRetrievalService(
                vector_store=milvus_client,
                embedding_generator=embeddings_manager,
                logger=app_logger,
                default_collection=settings.milvus.milvus_collection_name
            )
            app_logger.info(f"📄 正在上传文档: {args.title}")
            print(f"📄 正在上传文档: {args.title}")
            from domain.entity.document.document import Document
            from domain.vo.document.document_metadata import DocumentMetadata, DocumentType, DocumentSource

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 创建文档元数据
            metadata = DocumentMetadata(
                title=args.title,
                document_type=DocumentType[args.type.upper()],
                source=DocumentSource[args.source.upper()],
                created_at=current_time,
                updated_at=current_time
            )

            # 创建文档实体（使用自增ID，不预设ID）
            doc_data = {
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
            # Lazy import and create retrieval_service when needed
            from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
            from infrastructure.persistence.vector.milvus_client import MilvusClient
            from infrastructure.model.embeddings_manager import EmbeddingsManager
            from config.settings import settings

            milvus_client = MilvusClient()
            embeddings_manager = EmbeddingsManager()
            retrieval_service = MilvusDocumentRetrievalService(
                vector_store=milvus_client,
                embedding_generator=embeddings_manager,
                logger=app_logger,
                default_collection=settings.milvus.milvus_collection_name
            )

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
            # Lazy import and create retrieval_service when needed
            from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
            from infrastructure.persistence.vector.milvus_client import MilvusClient
            from infrastructure.model.embeddings_manager import EmbeddingsManager
            from config.settings import settings

            milvus_client = MilvusClient()
            embeddings_manager = EmbeddingsManager()
            retrieval_service = MilvusDocumentRetrievalService(
                vector_store=milvus_client,
                embedding_generator=embeddings_manager,
                logger=app_logger,
                default_collection=settings.milvus.milvus_collection_name
            )
            app_logger.info("📊 获取集合信息")
            print("📊 集合信息:")
            from application.services.document.collection_service_impl import CollectionServiceImpl
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
            # Lazy import and create retrieval_service when needed
            from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
            from infrastructure.persistence.vector.milvus_client import MilvusClient
            from infrastructure.model.embeddings_manager import EmbeddingsManager
            from config.settings import settings

            milvus_client = MilvusClient()
            embeddings_manager = EmbeddingsManager()
            retrieval_service = MilvusDocumentRetrievalService(
                vector_store=milvus_client,
                embedding_generator=embeddings_manager,
                logger=app_logger,
                default_collection=settings.milvus.milvus_collection_name
            )

            app_logger.info(f"🗑️  正在删除文档: {args.id}")
            print(f"🗑️  正在删除文档: {args.id}")
            retrieval_service.remove_document_from_collection(args.id, collection_name=args.collection)
            app_logger.info("✅ 文档删除成功")
            print("✅ 文档删除成功")

        elif args.subcommand == "ingest-file":
            app_logger.info(f"📄 正在摄入文件: {args.path}")
            print(f"📄 正在摄入文件: {args.path}")

            # 创建文档仓储
            doc_repo = LangChainDocumentRepository()
            if args.collection:
                doc_repo.collection_name = args.collection

            # 创建RAG处理服务
            factory = RAGProcessingServiceFactoryImpl()
            rag_service: RAGProcessingService = factory.create_service(
                domain=args.domain,
                document_repository=doc_repo
            )

            # 处理文件（process_file 内部已经调用 add_documents 完成插入）
            doc_ids = rag_service.process_file(args.path, args.type)

            app_logger.info(f"✅ 文件摄入完成，生成 {len(doc_ids)} 个文档块")
            print(f"✅ 文件摄入完成")
            print(f"   源文件: {args.path}")
            print(f"   生成文档块数量: {len(doc_ids)}")
            if len(doc_ids) > 0:
                print(f"   第一个文档ID: {doc_ids[0]}")

        elif args.subcommand == "ingest-dir":
            app_logger.info(f"📂 正在批量摄入目录: {args.path}, 模式: {args.pattern}")
            print(f"📂 正在批量摄入目录: {args.path}")
            print(f"   匹配模式: {args.pattern}")

            # 创建文档仓储
            doc_repo = LangChainDocumentRepository()
            if args.collection:
                doc_repo.collection_name = args.collection

            # 创建RAG处理服务
            factory = RAGProcessingServiceFactoryImpl()
            rag_service: RAGProcessingService = factory.create_service(
                domain=args.domain,
                document_repository=doc_repo
            )

            # 处理目录（process_directory 内部已经调用 add_documents 完成插入）
            doc_ids = rag_service.process_directory(
                directory_path=args.path,
                loader_type=args.type,
                glob_pattern=args.pattern
            )

            app_logger.info(f"✅ 目录摄入完成，共生成 {len(doc_ids)} 个文档块")
            print(f"✅ 目录摄入完成")
            print(f"   目录: {args.path}")
            print(f"   匹配模式: {args.pattern}")
            print(f"   生成文档块数量: {len(doc_ids)}")

        elif args.subcommand == "collection":
            from application.services.document.collection_service_impl import CollectionServiceImpl
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