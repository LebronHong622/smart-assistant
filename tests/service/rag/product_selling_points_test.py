#!/usr/bin/env python
"""
测试脚本：向 doc_product_selling_points 集合插入测试数据并验证查询
功能：
1. 加载环境配置
2. 插入商品卖点测试数据
3. 语义检索测试
4. 获取单个文档测试
5. 删除功能测试
6. 验证删除结果
"""
import os
import sys
import time
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载环境变量
load_dotenv()

from domain.entity.document.document import Document
from infrastructure.persistence.vector.repository.langchain_document_repository_impl import (
    LangChainDocumentRepository,
)
from infrastructure.core.log import app_logger
from application.services.document.rag_processing_service_impl import (
    RAGProcessingServiceImpl,
)


def create_test_data() -> list[Document]:
    """创建商品卖点测试数据，符合 product_selling_points 集合 schema"""
    products = [
        {
            "selling_point_id": "SP001",
            "product_id": "P001",
            "category": "蓝牙耳机",
            "is_highlight": True,
            "priority": 1,
            "status": "active",
            "title": "无线蓝牙耳机 Pro",
            "source": "test_data",
            "content": """商品名称：无线蓝牙耳机 Pro
品牌：声悦
主要卖点：
1. 主动降噪深度达40dB，有效隔绝外界噪音
2. 单次续航8小时，配合充电盒总续航32小时
3. 支持透明模式，无需摘下即可对话
4. 蓝牙5.3连接，稳定低延迟
5. IPX4防水防汗，适合运动使用
适用场景：通勤、办公、运动、旅行
价格：299元""",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        },
        {
            "selling_point_id": "SP002",
            "product_id": "P002",
            "category": "智能穿戴",
            "is_highlight": True,
            "priority": 2,
            "status": "active",
            "title": "智能健康手表 Fit",
            "source": "test_data",
            "content": """商品名称：智能健康手表 Fit
品牌：智联
主要卖点：
1. 24小时实时心率监测，异常提醒
2. 支持100+运动模式，自动识别运动类型
3. 血氧检测，睡眠质量分析
4. 50米防水，可游泳佩戴
5. 单次续航7天，待机30天
6. 支持NFC支付和消息提醒
功能：健康监测、运动记录、消息通知
价格：399元""",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        },
        {
            "selling_point_id": "SP003",
            "product_id": "P003",
            "category": "移动电源",
            "is_highlight": True,
            "priority": 3,
            "status": "active",
            "title": "20000mAh 便携充电宝",
            "source": "test_data",
            "content": """商品名称：20000mAh 轻薄便携充电宝
品牌：电宝
主要卖点：
1. 20000mAh大容量，可为手机充电4-5次
2. 支持22.5W双向快充
3. 三输入三输出，同时给三台设备充电
4. 机身厚度仅15mm，重量280g，便携轻盈
5. 内置智能保护芯片，防过充过放
6. 支持PD3.0 QC3.0快充协议
适用场景：出行、旅游、应急充电
价格：129元""",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
        }
    ]

    documents = []
    for product in products:
        doc = Document(
            content=product["content"],
            metadata={
                "selling_point_id": product["selling_point_id"],
                "product_id": product["product_id"],
                "category": product["category"],
                "is_highlight": product["is_highlight"],
                "priority": product["priority"],
                "status": product["status"],
                "title": product["title"],
                "source": product["source"],
                "created_at": product["created_at"],
                "updated_at": product["updated_at"],
            }
        )
        documents.append(doc)

    return documents


def main():
    """主测试流程"""
    app_logger.info("=== 开始商品卖点集合 RAG 测试 (LangChainDocumentRepository)")

    # 1. 初始化组件
    collection_name = "doc_product_selling_points"
    domain = "product_selling_points"

    app_logger.info(f"初始化组件: collection={collection_name}, domain={domain}")

    # 创建 LangChainDocumentRepository 实例
    repo = LangChainDocumentRepository(
        collection_name=collection_name
    )
    # 创建 RAGProcessingServiceImpl 实例
    service = RAGProcessingServiceImpl(
        domain=domain,
        document_repository=repo
    )

    # 输出当前文档数量
    try:
        count = repo.count()
        app_logger.info(f"当前集合文档数量: {count}")
    except Exception as e:
        app_logger.warning(f"获取文档数量失败: {e}")

    # 2. 插入测试数据
    app_logger.info("插入商品卖点测试数据...")
    test_docs = create_test_data()
    inserted_ids = service.add_documents(test_docs)
    app_logger.info(f"插入完成，插入的文档ID: {inserted_ids}")
    print(f"\n✅ 插入 {len(inserted_ids)} 个测试文档")
    for idx, doc_id in enumerate(inserted_ids):
        title = test_docs[idx].metadata.get("title", "未知")
        print(f"   [{idx+1}] ID: {doc_id} - {title}")

    # 3. 测试不同查询
    print("\n=== 语义检索测试")
    test_queries = [
        "续航长的充电宝",
        "能监测心率的智能手表",
        "降噪蓝牙耳机",
        "支持快充的移动电源",
        "运动防水手表"
    ]

    all_inserted_ids = inserted_ids.copy()

    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        results = service.retrieve_similar(query, limit=2, score_threshold=0.5)
        if results:
            for i, doc in enumerate(results):
                title = doc.metadata.get("title", "未知")
                score = doc.metadata.get("similarity_score", "N/A")
                if isinstance(score, float):
                    print(f"  {i+1}. {title} (相似度: {score:.4f})")
                else:
                    print(f"  {i+1}. {title}")
                content_preview = doc.content[:50].replace("\n", " ") + "..."
                print(f"      {content_preview}")
        else:
            print("  ❌ 未找到相关结果")

    # 4. 测试get_document - 注意：LangChainDocumentRepository 的 find_by_id 暂未实现
    # 这一步预期会抛出 NotImplementedError，我们记录这个情况
    print(f"\n=== 获取单个文档测试")
    print(f"⚠️  LangChainDocumentRepository 暂未实现 find_by_id 方法，跳过此测试")

    # 5. 删除功能测试
    print(f"\n=== 删除功能测试")
    delete_result = service.delete_documents(all_inserted_ids)
    if delete_result:
        print(f"✅ 删除成功，删除了 {len(all_inserted_ids)} 个文档")
    else:
        print(f"❌ 删除失败")

    # 6. 验证删除 - 通过检索验证
    print(f"\n=== 最终检索验证（删除后）")
    final_results = service.retrieve_similar("蓝牙耳机", limit=3)
    if len(final_results) == 0:
        print(f"✅ 删除后检索无结果，验证通过")
    else:
        print(f"⚠️ 删除后仍有 {len(final_results)} 条结果，可能是其他测试数据")
        for i, doc in enumerate(final_results):
            title = doc.metadata.get("title", "未知")
            print(f"  {i+1}. {title}")

    app_logger.info("=== 商品卖点集合 RAG 测试完成")


if __name__ == "__main__":
    main()
