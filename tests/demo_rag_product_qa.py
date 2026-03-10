"""
LangChain RAG 商品问答 Demo
基于 product_info.json schema 实现完整的 RAG 流程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from datetime import datetime

from infrastructure.persistence.vector.adapters.langchain_milvus_adapter import LangChainMilvusAdapter
from infrastructure.external.model.embedding.adapters.langchain_embeddings_adapter import LangChainEmbeddingsAdapter
from infrastructure.external.model.model_factory import ModelFactory


# 示例商品数据（基于 product_info.json schema）
SAMPLE_PRODUCTS = [
    {
        "product_id": "P001",
        "name": "荣耀Magic6 Pro",
        "category": "智能手机",
        "brand": "荣耀",
        "price": 5999.00,
        "tags": "5G,旗舰,拍照,商务",
        "specifications": "屏幕:6.8英寸OLED|处理器:骁龙8 Gen3|内存:12GB|存储:256GB|电池:5600mAh",
        "key_attributes": "鹰眼相机,青海湖电池,鸿燕通信",
        "sales_count": 15000,
        "rating": 4.8,
        "status": "上架",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": "荣耀Magic6 Pro是一款旗舰级智能手机，搭载骁龙8 Gen3处理器，配备6.8英寸OLED曲面屏。最大亮点是鹰眼相机系统，支持全焦段拍摄，夜景表现出色。青海湖电池技术带来5600mAh大容量，续航强劲。支持鸿燕卫星通信，适合商务人士使用。"
    },
    {
        "product_id": "P002",
        "name": "荣耀MagicBook 14",
        "category": "笔记本电脑",
        "brand": "荣耀",
        "price": 4999.00,
        "tags": "轻薄,商务,办公,高性价比",
        "specifications": "屏幕:14英寸2.5K|处理器:i5-13500H|内存:16GB|存储:512GB|重量:1.52kg",
        "key_attributes": "轻薄设计,长续航,多屏协同",
        "sales_count": 8000,
        "rating": 4.6,
        "status": "上架",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": "荣耀MagicBook 14是一款轻薄商务笔记本，采用14英寸2.5K高刷屏，搭载第13代英特尔酷睿i5处理器。机身重量仅1.52kg，支持多屏协同功能，可与荣耀手机无缝连接。75Wh大电池提供长达15小时续航，非常适合移动办公。"
    },
    {
        "product_id": "P003",
        "name": "荣耀手表4",
        "category": "智能手表",
        "brand": "荣耀",
        "price": 999.00,
        "tags": "运动,健康,长续航,时尚",
        "specifications": "屏幕:1.75英寸AMOLED|电池:451mAh|防水:5ATM|重量:29g",
        "key_attributes": "eSIM独立通话,血氧检测,运动模式",
        "sales_count": 25000,
        "rating": 4.5,
        "status": "上架",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": "荣耀手表4是一款支持eSIM独立通话的智能手表，配备1.75英寸AMOLED高清屏幕。支持100+运动模式和全天候健康监测，包括心率、血氧、睡眠监测等。典型使用续航达10天，支持无线充电，适合运动爱好者和关注健康的用户。"
    },
    {
        "product_id": "P004",
        "name": "荣耀平板9",
        "category": "平板电脑",
        "brand": "荣耀",
        "price": 1999.00,
        "tags": "大屏,影音,学习,护眼",
        "specifications": "屏幕:12.1英寸2.5K|处理器:骁龙6 Gen1|内存:8GB|存储:128GB|电池:8300mAh",
        "key_attributes": "纸感护眼屏,多窗口,魔法键盘",
        "sales_count": 12000,
        "rating": 4.7,
        "status": "上架",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": "荣耀平板9配备12.1英寸2.5K纸感护眼屏，获得莱茵护眼认证，长时间使用不易疲劳。支持多窗口分屏功能，可同时处理多个任务。搭配魔法键盘和手写笔，适合学生学习和轻办公场景。8300mAh大电池提供持久续航。"
    },
    {
        "product_id": "P005",
        "name": "荣耀Earbuds 3 Pro",
        "category": "真无线耳机",
        "brand": "荣耀",
        "price": 699.00,
        "tags": "降噪,音质,舒适,运动",
        "specifications": "驱动单元:11mm|降噪深度:46dB|续航:24小时|防水:IP54",
        "key_attributes": "自适应降噪,空间音频,智能佩戴检测",
        "sales_count": 30000,
        "rating": 4.4,
        "status": "上架",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "content": "荣耀Earbuds 3 Pro是一款旗舰级真无线耳机，支持46dB自适应主动降噪，可根据环境自动调整降噪强度。11mm大动圈单元提供出色音质，支持空间音频技术。佩戴检测功能可自动暂停播放，IP54防水适合运动使用。配合充电盒续航达24小时。"
    }
]

# RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """你是一个专业的荣耀商品推荐助手。请根据以下上下文信息回答用户问题。
如果上下文中没有相关信息，请诚实地说"根据现有信息，我无法回答这个问题"，不要编造答案。

上下文信息：
{context}

用户问题：{question}

请提供专业、准确的回答："""


def init_vector_store(collection_name: str = "product_intro") -> LangChainMilvusAdapter:
    """初始化向量存储"""
    embeddings = LangChainEmbeddingsAdapter()
    vector_store = LangChainMilvusAdapter(
        collection_name=collection_name,
        embedding_model=embeddings
    )
    print(f"✓ 向量存储初始化完成，集合名称: {collection_name}")
    return vector_store


def insert_sample_data(vector_store: LangChainMilvusAdapter):
    """插入示例商品数据"""
    print("\n=== 开始插入示例数据 ===")
    
    texts = []
    metadatas = []
    
    for product in SAMPLE_PRODUCTS:
        texts.append(product["content"])
        metadatas.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
            "brand": product["brand"],
            "price": product["price"],
            "tags": product["tags"],
            "specifications": product["specifications"],
            "key_attributes": product["key_attributes"],
            "sales_count": product["sales_count"],
            "rating": product["rating"],
            "status": product["status"],
            "created_at": product["created_at"],
            "updated_at": product["updated_at"]
        })
    
    ids = vector_store.add_texts(texts, metadatas=metadatas)
    print(f"✓ 成功插入 {len(ids)} 条商品数据")
    return ids


def create_rag_chain(vector_store: LangChainMilvusAdapter):
    """创建 RAG 问答链"""
    # 获取 LLM
    llm = ModelFactory.get_llm(adapter_name="default", model_name="deepseek-chat")
    
    # 创建 Prompt 模板
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    # 创建 RetrievalQA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    
    print("✓ RAG 问答链创建完成")
    return qa_chain


def test_search(vector_store: LangChainMilvusAdapter, query: str, k: int = 3):
    """测试向量检索"""
    print(f"\n=== 检索测试: '{query}' ===")
    results = vector_store.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[结果 {i}] 相似度分数: {score:.4f}")
        print(f"商品名称: {doc.metadata.get('name', 'N/A')}")
        print(f"类别: {doc.metadata.get('category', 'N/A')}")
        print(f"价格: ¥{doc.metadata.get('price', 'N/A')}")
        print(f"内容摘要: {doc.page_content[:100]}...")


def test_qa(qa_chain, question: str):
    """测试 RAG 问答"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print("="*60)
    
    result = qa_chain.invoke({"query": question})
    
    print(f"\n回答: {result['result']}")
    print(f"\n来源文档数量: {len(result['source_documents'])}")
    
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n  [{i}] {doc.metadata.get('name', 'Unknown')}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("LangChain RAG 商品问答 Demo")
    print("="*60)
    
    # 1. 初始化向量存储
    vector_store = init_vector_store("product_intro")
    
    # 2. 插入示例数据
    insert_sample_data(vector_store)
    
    # 3. 测试向量检索
    print("\n" + "="*60)
    print("向量检索测试")
    print("="*60)
    test_search(vector_store, "拍照效果好的手机", k=2)
    test_search(vector_store, "适合办公的笔记本", k=2)
    
    # 4. 创建 RAG 问答链
    print("\n" + "="*60)
    print("创建 RAG 问答链")
    print("="*60)
    qa_chain = create_rag_chain(vector_store)
    
    # 5. 测试 RAG 问答
    print("\n" + "="*60)
    print("RAG 问答测试")
    print("="*60)
    
    test_questions = [
        "推荐一款拍照效果好的手机",
        "有什么适合办公使用的设备？",
        "预算2000元左右有什么好推荐？",
        "运动时适合用什么耳机？",
        "有没有长续航的智能手表？"
    ]
    
    for question in test_questions:
        test_qa(qa_chain, question)
    
    print("\n" + "="*60)
    print("Demo 执行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
