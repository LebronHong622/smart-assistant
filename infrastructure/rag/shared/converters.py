"""
文档转换工具
LangChain Document 与领域 Document 之间的相互转换
"""
from typing import List
from langchain_core.documents import Document as LCDocument
from domain.entity.document.document import Document


def convert_lc_to_domain(lc_docs: List[LCDocument]) -> List[Document]:
    """将 LangChain Document 转换为领域 Document"""
    return [
        Document(content=doc.page_content, metadata=doc.metadata)
        for doc in lc_docs
    ]


def convert_domain_to_lc(docs: List[Document]) -> List[LCDocument]:
    """将领域 Document 转换为 LangChain Document"""
    return [
        LCDocument(page_content=doc.content, metadata=doc.metadata or {})
        for doc in docs
    ]
