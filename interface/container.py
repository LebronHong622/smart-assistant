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
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.ports.prompt_port import PromptPort
from application.services.rag.agentic_rag_workflow import AgenticRAGWorkflow

# 导入适配器
from infrastructure.core.log.adapters.logger_adapter import LoggerAdapter
from infrastructure.external.model.embedding.adapters.dashscope_embedding_adapter import DashScopeEmbeddingAdapter
from infrastructure.external.model.llm.adapters.llm_adapter import LLMAdapter
from infrastructure.external.model.llm.adapters.langchain_chat_adapter import LangChainChatAdapter
from infrastructure.core.memory.adapters.session_memory_adapter import SessionMemoryAdapter
from infrastructure.external.tool.adapters.langchain_frame_adapter import LangChainFrameAdapter
from infrastructure.external.tool.loaders.yaml_config_loader import YamlConfigLoader
from infrastructure.external.prompt.adapters.langchain_prompt_adapter import LangChainPromptAdapter
from infrastructure.external.prompt.loaders.yaml_loader import YamlTemplateLoader
from infrastructure.external.model.routing import ModelRouter


# 导入仓储
from domain.document.repository.document_repository import DocumentRepository
from domain.document.repository.document_collection_repository import DocumentCollectionRepository

# 导入服务
from domain.service.conversation.conversation_service import ConversationService
from domain.service.document.rag_processing_service import RAGProcessingService
from application.services.conversation.langchain_conversation_service_impl import LangchainConversationServiceImpl
from application.services.document.document_service_impl import DocumentServiceImpl
from application.services.document.document_retrieval_service_impl import MilvusDocumentRetrievalService
from application.services.document.collection_service_impl import CollectionServiceImpl
from application.services.rag.langchain_agentic_rag_service_impl import LangchainAgenticRagServiceImpl
from application.agent.agentic_rag_agent import AgenticRagAgent

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
        return SessionMemoryAdapter()
    
    @lru_cache
    def get_tool_provider(self) -> ToolPort:
        """获取工具端口实现，依赖注入YAML配置加载器"""
        # 创建配置加载器
        config_loader = YamlConfigLoader()
        # 注入到LangChain框架适配器
        return LangChainFrameAdapter(config_loader=config_loader)
    
    @lru_cache
    def get_model_router(self) -> ModelRouterPort:
        """获取模型适配器"""
        return ModelRouter()

    @lru_cache
    def get_model_provider(self) -> ModelPort:
        """获取模型适配器"""
        return LLMAdapter()

    @lru_cache
    def get_prompt_provider(self) -> PromptPort:
        """获取提示词管理适配器"""
        yaml_loader = YamlTemplateLoader()
        return LangChainPromptAdapter(yaml_loader)
    
    # ========== 仓储层 ==========
    
    @lru_cache
    def get_document_repository(self) -> DocumentRepository:
        """获取文档仓储"""
        from infrastructure.persistence.vector.repository.langchain_document_repository_impl import LangChainDocumentRepository
        return LangChainDocumentRepository()
    
    @lru_cache
    def get_collection_repository(self) -> DocumentCollectionRepository:
        """获取文档集合仓储"""
        from infrastructure.persistence.vector.repository.document_collection_repository_impl import MilvusDocumentCollectionRepository
        return MilvusDocumentCollectionRepository()
    
    # ========== 领域服务 ==========
    
    @lru_cache
    def get_conversation_service(self) -> ConversationService:
        """获取对话服务"""
        return LangchainConversationServiceImpl(
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

    @lru_cache
    def get_agentic_rag_workflow(self) -> AgenticRAGWorkflow:
        """获取Agentic RAG工作流"""
        from application.services.rag.rag_processing_service_impl import RAGProcessingServiceFactoryImpl

        return LangchainAgenticRagServiceImpl(
            logger=self.get_logger(),
            tool_port=self.get_tool_provider(),
            prompt_port=self.get_prompt_provider(),
            model_router_port=self.get_model_router(),
            rag_processing_service_factory=RAGProcessingServiceFactoryImpl(),
            document_repository_factory=self.get_document_repository
        )

    def get_agentic_rag_agent(self, session_id: Optional[str] = None) -> AgenticRagAgent:
        """获取Agentic RAG代理实例（非缓存，每个会话独立实例）"""
        # 使用已缓存的单例memory_provider
        memory_port = self.get_memory_provider()
        return AgenticRagAgent(
            rag_workflow=self.get_agentic_rag_workflow(),
            memory_port=memory_port,
            logger=self.get_logger(),
            session_id=session_id
        )

    @lru_cache
    def get_rag_processing_service(self, domain: str = "default") -> RAGProcessingService:
        """获取RAG处理服务实例"""
        from application.services.rag.rag_processing_service_impl import RAGProcessingServiceFactoryImpl
        factory = RAGProcessingServiceFactoryImpl()
        return factory.create_service(
            domain=domain,
            document_repository=self.get_document_repository()
        )


# 全局容器实例
container = Container()
