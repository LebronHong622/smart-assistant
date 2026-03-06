"""
文档管理命令行工具
用于管理 Milvus 向量数据库中的文档
"""

import argparse
import sys
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService


def main():
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

    # 检索文档子命令
    retrieve_parser = subparsers.add_parser("retrieve", help="检索相似文档")
    retrieve_parser.add_argument("-q", "--query", required=True, help="查询文本")
    retrieve_parser.add_argument("-l", "--limit", type=int, default=5, help="返回结果数量限制")
    retrieve_parser.add_argument("-s", "--score", type=float, default=0.5, help="相似度分数阈值")

    # 获取集合信息子命令
    info_parser = subparsers.add_parser("info", help="获取集合信息")

    # 删除文档子命令
    delete_parser = subparsers.add_parser("delete", help="删除文档")
    delete_parser.add_argument("-i", "--id", required=True, help="文档ID")

    args = parser.parse_args()

    try:
        retrieval_service = MilvusDocumentRetrievalService()

        if args.subcommand == "upload":
            print(f"📄 正在上传文档: {args.title}")
            from domain.document.entity.document import Document
            from domain.document.value_object.document_metadata import DocumentMetadata, DocumentType, DocumentSource
            from uuid import uuid4

            # 创建文档元数据
            metadata = DocumentMetadata(
                title=args.title,
                document_type=DocumentType[args.type.upper()],
                source=DocumentSource[args.source.upper()],
                created_at="2024-01-01 00:00:00",
                updated_at="2024-01-01 00:00:00"
            )

            # 创建文档实体
            document = Document(
                id=uuid4(),
                content=args.content,
                metadata=metadata.__dict__
            )

            # 添加到检索集合
            retrieval_service.add_document_to_collection(document)
            print(f"✅ 文档上传成功，ID: {document.id}")

        elif args.subcommand == "retrieve":
            print(f"🔍 正在检索文档: {args.query}")
            results = retrieval_service.retrieve_similar_documents(
                query=args.query,
                limit=args.limit,
                score_threshold=args.score
            )

            print(f"📊 找到 {len(results)} 个相关文档:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📄 文档ID: {result.document_id}")
                print(f"   📝 内容预览: {result.content[:100]}...")
                print(f"   📊 相似度: {result.similarity_score:.4f}")
                if result.metadata:
                    print(f"   🏷️  标题: {result.metadata.get('title', '未命名')}")

        elif args.subcommand == "info":
            print("📊 集合信息:")
            from infrastructure.vector.vector_store import MilvusVectorStore
            vector_store = MilvusVectorStore()
            info = vector_store.get_collection_info()

            print(f"名称: {info['name']}")
            print(f"描述: {info['description']}")
            print(f"文档数量: {info['num_entities']}")
            print(f"向量维度: {info['schema']['fields'][3]['params']['dim']}")

        elif args.subcommand == "delete":
            print(f"🗑️  正在删除文档: {args.id}")
            retrieval_service.remove_document_from_collection(args.id)
            print("✅ 文档删除成功")

        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())