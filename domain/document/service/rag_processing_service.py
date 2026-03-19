"""
领域层：RAG处理服务抽象接口
完全不依赖任何外部框架，定义核心业务接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository


class RAGProcessingService(ABC):
    """
    RAG处理服务抽象基类
    定义完整的RAG处理流程核心接口
    """

    @abstractmethod
    def process_document(self, document: Document) -> Document:
        """
        处理单个文档：包括文本清洗、分块、生成嵌入向量
        :param document: 原始文档实体
        :return: 处理完成的文档实体（包含嵌入向量）
        """
        pass

    @abstractmethod
    def batch_process_documents(self, documents: List[Document]) -> List[Document]:
        """
        批量处理多个文档
        :param documents: 原始文档实体列表
        :return: 处理完成的文档实体列表
        """
        pass

    @abstractmethod
    def retrieve_similar(self, query: str, limit: int = 5, score_threshold: float = 0.7) -> List[Document]:
        """
        根据查询文本检索相似文档
        :param query: 查询文本
        :param limit: 返回结果数量上限
        :param score_threshold: 相似度阈值，高于该值的结果才会返回
        :return: 相似文档列表
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        将处理后的文档添加到向量存储中
        :param documents: 处理完成的文档实体列表
        :return: 新增文档ID列表
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        根据文档ID删除向量存储中的文档
        :param document_ids: 文档ID列表
        :return: 是否删除成功
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        根据文档ID获取文档信息
        :param document_id: 文档ID
        :return: 文档实体，如果不存在返回None
        """
        pass

    @abstractmethod
    def process_file(self, file_path: str, loader_type: Optional[str] = None, **kwargs) -> List[Document]:
        """处理单个文件：加载文件、文本清洗、分块、生成嵌入向量
        :param file_path: 文件路径
        :param loader_type: 加载器类型 (pdf/txt/md/csv等)，不指定则使用默认配置
        :param kwargs: 传递给加载器的额外参数
        :return: 处理完成的文档块列表
        """
        pass

    @abstractmethod
    def process_directory(self, directory_path: str, loader_type: Optional[str] = None,
                      glob_pattern: str = "**/*", **kwargs) -> List[Document]:
        """处理目录中的所有文件
        :param directory_path: 目录路径
        :param loader_type: 加载器类型，不指定则使用默认配置
        :param glob_pattern: 文件匹配模式，默认为 **/* 匹配所有文件
        :param kwargs: 传递给加载器的额外参数
        :return: 处理完成的文档块列表
        """
        pass


class RAGProcessingServiceFactory(ABC):
    """
    RAG处理服务工厂抽象接口
    用于创建不同业务领域的RAG服务实例
    """

    @abstractmethod
    def create_service(self, domain: str = "default", document_repository: Optional[DocumentRepository] = None, **kwargs) -> RAGProcessingService:
        """
        创建指定领域的RAG处理服务实例
        :param domain: 业务领域标识，用于区分不同的RAG实例
        :param document_repository: 文档仓储实例
        :return: RAG处理服务实例
        """
        pass
