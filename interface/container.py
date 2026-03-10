"""
依赖注入容器
负责实例化所有基础设施实现并注入到应用层和领域层
"""

from functools import lru_cache
from typing import Optional

# 导入端口
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.embedding_port import EmbeddingGeneratorPort
from domain.shared.ports.vector_store_port import VectorStorePort
from domain.shared.ports.memory_port import MemoryPort
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.model_port import ModelPort

# 导入适配器
from infrastructure.core.log.adapters.logger_adapter import LoggerAdapter
from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
from infrastructure.external.model.llm.adapters.llm_adapter import LLMAdapter
from infrastructure.core.memory.adapters.memory_adapter import MemoryAdapter
from infrastructure.external.tool.adapters.tool_adapter import ToolAdapter

# 导入仓储
from domain.document.repository.document_repository import DocumentRepository
from domain.document.repository.document_collection_repository import DocumentCollectionRepository

# 导入服务
from domain.qa.service.qa_service import QAService
from application.services.document_service_impl import DocumentServiceImpl
from application.services.document_retrieval_service_impl import MilvusDocumentRetrievalService
from application.services.collection_service_impl import CollectionServiceImpl

# 导入配置
from config.settings import settings


class Container:
    """
    依赖注入容器
    使用单例模式管理所有依赖
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # ========== 基础设施层适配器 ==========
    
    @lru_cache
    def get_logger(self) -> LoggerPort:
        """获取日志适配器"""
        return LoggerAdapter()
    
    @lru_cache
    def get_embedding_generator(self) -> EmbeddingGeneratorPort:
        """获取嵌入向量生成适配器"""
        return DashScopeEmbeddingAdapter()
    
    @lru_cache
    def get_vector_store(self, collection_name: str = None) -> VectorStorePort:
        """获取向量存储适配器（延迟导入）"""
        from infrastructure.persistence.vector.adapters.milvus_adapter import MilvusAdapter
        return MilvusAdapter(collection_name=collection_name or settings.milvus.milvus_collection_name)
    
    @lru_cache
    def get_memory_provider(self) -> MemoryPort:
        """获取内存管理适配器"""
        return MemoryAdapter()
    
    @lru_cache
    def get_tool_provider(self) -> ToolPort:
        """获取工具适配器"""
        return ToolAdapter()
    
    @lru_cache
    def get_model_provider(self) -> ModelPort:
        """获取模型适配器"""
        return LLMAdapter()
    
    # ========== 仓储层 ==========
    
    @lru_cache
    def get_document_repository(self) -> DocumentRepository:
        """获取文档仓储"""
        storage_type = settings.document_storage.document_storage_type
        
        if storage_type == "milvus":
            from infrastructure.persistence.vector.repository.document_repository_impl import MilvusDocumentRepository
            return MilvusDocumentRepository()
        else:  # 默认使用本地存储
            from infrastructure.persistence.vector.repository.local_document_repository import LocalDocumentRepository
            return LocalDocumentRepository()
    
    @lru_cache
    def get_collection_repository(self) -> DocumentCollectionRepository:
        """获取文档集合仓储"""
        from infrastructure.persistence.vector.repository.document_collection_repository_impl import MilvusDocumentCollectionRepository
        return MilvusDocumentCollectionRepository()
    
    # ========== 领域服务 ==========
    
    @lru_cache
    def get_qa_service(self) -> QAService:
        """获取问答服务"""
        return QAService(
            logger=self.get_logger(),
            tool_provider=self.get_tool_provider(),
            model_provider=self.get_model_provider(),
            memory_provider=self.get_memory_provider()
        )
    
    # ========== 应用服务 ==========
    
    @lru_cache
    def get_document_service(self) -> DocumentServiceImpl:
        """获取文档管理服务"""
        return DocumentServiceImpl(
            document_repository=self.get_document_repository(),
            logger=self.get_logger()
        )
    
    @lru_cache
    def get_document_retrieval_service(self, collection_name: str = None) -> MilvusDocumentRetrievalService:
        """获取文档检索服务"""
        return MilvusDocumentRetrievalService(
            vector_store=self.get_vector_store(collection_name),
            embedding_generator=self.get_embedding_generator(),
            logger=self.get_logger(),
            default_collection=collection_name
        )
    
    @lru_cache
    def get_collection_service(self) -> CollectionServiceImpl:
        """获取集合管理服务"""
        return CollectionServiceImpl(
            collection_repository=self.get_collection_repository(),
            vector_store=self.get_vector_store(),
            logger=self.get_logger()
        )


# 全局容器实例
container = Container()
