"""
单元测试：RAG工作流节点工厂
使用 mock 隔离所有外部依赖，不依赖实际LLM和数据库
"""
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import AIMessage

from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.ports.model_capability_port import BaseModel
from domain.entity.document.document import Document
from domain.service.document.rag_processing_service import RAGProcessingServiceFactory
from domain.repository.document.document_repository import DocumentRepository
from domain.shared.model_enums import ModelType, RoutingStrategy
from application.services.rag.workflow.nodes import (
    create_intent_classification_node,
    create_product_retrieve_node,
    create_after_sales_retrieve_node,
    create_promotion_retrieve_node,
    create_general_generate_node as create_generate_node
)
from application.services.rag.workflow.state import AgentState


class TestIntentClassificationNode:
    """测试意图分类节点"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_prompt_port = Mock(spec=PromptPort)
        self.mock_model_router = Mock(spec=ModelRouterPort)
        self.mock_logger = Mock(spec=LoggerPort)

        # Mock LLM
        self.mock_llm = Mock(spec=BaseModel)
        self.mock_model_router.get_model.return_value = self.mock_llm

    def test_create_intent_classification_node_returns_callable(self):
        """测试工厂函数返回可调用节点"""
        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )
        assert callable(node)

    def test_calls_get_model_with_correct_parameters(self):
        """测试使用正确参数获取LLM"""
        # 设置mock返回
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "商品导购"
        self.mock_llm.invoke.return_value = mock_ai_message

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "这款商品有什么特点？",
            "session_id": "test_session",
            "chat_history": []
        }

        node(state)

        # 验证调用参数
        self.mock_model_router.get_model.assert_called_once_with(
            ModelType.CHAT,
            strategy=RoutingStrategy.DEFAULT
        )

    def test_intent_mapping_product_selling_points(self):
        """测试中文意图映射：商品导购"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "商品导购"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "这款商品有什么特点？",
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert result["intent"] == "product_selling_points"
        assert result["needs_retrieval"] is True
        assert result["rewrite_count"] == 0

    def test_intent_mapping_after_sales_policy(self):
        """测试中文意图映射：售后规则"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "售后规则"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "退货政策是什么？",
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert result["intent"] == "after_sales_policy"
        assert result["needs_retrieval"] is True

    def test_intent_mapping_promotion_rules(self):
        """测试中文意图映射：促销规则"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "促销规则"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "现在有什么优惠活动？",
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert result["intent"] == "promotion_rules"
        assert result["needs_retrieval"] is True

    def test_intent_mapping_normal(self):
        """测试中文意图映射：normal"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "normal"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "你好，请介绍一下自己",
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert result["intent"] == "normal"
        assert result["needs_retrieval"] is False

    def test_unknown_intent_falls_back_to_normal(self):
        """测试未知意图默认fallback到normal"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "unknown_category"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_intent_classification_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "这是什么分类？",
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert result["intent"] == "normal"
        assert result["needs_retrieval"] is False


class TestProductRetrieveNode:
    """测试商品导购检索节点"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()
        self.mock_logger = Mock(spec=LoggerPort)

        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

        self.mock_rag_service = Mock()
        self.mock_rag_factory.create_service.return_value = self.mock_rag_service

    def test_create_retrieve_node_creates_service_with_correct_domain(self):
        """测试工厂使用正确domain创建服务"""
        create_product_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        self.mock_rag_factory.create_service.assert_called_once_with(
            domain="product_selling_points",
            document_repository=self.mock_doc_repo
        )

    def test_retrieve_documents_returns_correct_format(self):
        """测试检索返回正确格式的数据"""
        # 准备测试数据
        doc1 = Document(content="商品特点1", metadata={"category": "electronics"})
        doc1.id = 1
        doc1.similarity_score = 0.85

        doc2 = Document(content="商品特点2", metadata={"category": "electronics"})
        doc2.id = 2
        doc2.similarity_score = 0.75

        self.mock_rag_service.retrieve_similar.return_value = [doc1, doc2]

        node = create_product_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        state: AgentState = {
            "query": "这款商品有什么特点？",
            "rewritten_query": None,
            "session_id": "test_session",
            "chat_history": [],
            "intent": "product_selling_points"
        }

        result = node(state)

        # 验证调用参数
        self.mock_rag_service.retrieve_similar.assert_called_once_with(
            query="这款商品有什么特点？",
            limit=5,
            score_threshold=0.7
        )

        # 验证返回格式
        assert "retrieved_documents" in result
        assert "relevant_documents" in result
        documents = result["retrieved_documents"]
        assert len(documents) == 2
        assert documents[0]["id"] == "1"
        assert documents[0]["content"] == "商品特点1"
        assert documents[0]["domain"] == "product_selling_points"
        assert documents[0]["similarity_score"] == 0.85

    def test_retrieve_documents_uses_rewritten_query_when_available(self):
        """测试当存在rewritten_query时使用重写后的查询"""
        self.mock_rag_service.retrieve_similar.return_value = []

        node = create_product_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        state: AgentState = {
            "query": "原查询",
            "rewritten_query": "重写后的查询",
            "session_id": "test_session",
            "chat_history": []
        }

        node(state)

        self.mock_rag_service.retrieve_similar.assert_called_once()
        call_args = self.mock_rag_service.retrieve_similar.call_args
        assert call_args[1]["query"] == "重写后的查询"

    def test_retrieve_empty_results_returns_empty_list(self):
        """测试空检索结果返回空列表"""
        self.mock_rag_service.retrieve_similar.return_value = []

        node = create_product_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        state: AgentState = {
            "query": "无结果查询",
            "rewritten_query": None,
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        assert len(result["retrieved_documents"]) == 0


class TestAfterSalesRetrieveNode:
    """测试售后规则检索节点"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()
        self.mock_logger = Mock(spec=LoggerPort)

        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

        self.mock_rag_service = Mock()
        self.mock_rag_factory.create_service.return_value = self.mock_rag_service

    def test_create_retrieve_node_creates_service_with_correct_domain(self):
        """测试工厂使用正确domain创建服务"""
        create_after_sales_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        self.mock_rag_factory.create_service.assert_called_once_with(
            domain="after_sales_policy",
            document_repository=self.mock_doc_repo
        )

    def test_retrieve_documents_returns_correct_domain(self):
        """测试检索返回正确domain标记"""
        doc = Document(content="退货七天无理由", metadata={})
        doc.id = 1
        doc.similarity_score = 0.9
        self.mock_rag_service.retrieve_similar.return_value = [doc]

        node = create_after_sales_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        state: AgentState = {
            "query": "怎么退货？",
            "rewritten_query": None,
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        documents = result["retrieved_documents"]
        assert len(documents) == 1
        assert documents[0]["domain"] == "after_sales_policy"


class TestPromotionRetrieveNode:
    """测试促销规则检索节点"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()
        self.mock_logger = Mock(spec=LoggerPort)

        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

        self.mock_rag_service = Mock()
        self.mock_rag_factory.create_service.return_value = self.mock_rag_service

    def test_create_retrieve_node_creates_service_with_correct_domain(self):
        """测试工厂使用正确domain创建服务"""
        create_promotion_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        self.mock_rag_factory.create_service.assert_called_once_with(
            domain="promotion_rules",
            document_repository=self.mock_doc_repo
        )

    def test_retrieve_documents_returns_correct_domain(self):
        """测试检索返回正确domain标记"""
        doc = Document(content="满减活动", metadata={})
        doc.id = 1
        doc.similarity_score = 0.8
        self.mock_rag_service.retrieve_similar.return_value = [doc]

        node = create_promotion_retrieve_node(
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            self.mock_logger,
            default_limit=5
        )

        state: AgentState = {
            "query": "有什么优惠？",
            "rewritten_query": None,
            "session_id": "test_session",
            "chat_history": []
        }

        result = node(state)

        documents = result["retrieved_documents"]
        assert len(documents) == 1
        assert documents[0]["domain"] == "promotion_rules"


class TestGenerateNode:
    """测试回答生成节点"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_prompt_port = Mock(spec=PromptPort)
        self.mock_model_router = Mock(spec=ModelRouterPort)
        self.mock_logger = Mock(spec=LoggerPort)

        self.mock_llm = Mock(spec=BaseModel)
        self.mock_model_router.get_model.return_value = self.mock_llm

    def test_create_generate_node_returns_callable(self):
        """测试工厂函数返回可调用节点"""
        node = create_generate_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )
        assert callable(node)

    def test_calls_get_model_with_correct_parameters(self):
        """测试使用正确参数获取LLM"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "这是生成的回答"
        self.mock_llm.invoke.return_value = mock_ai_message

        node = create_generate_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "这是问题",
            "session_id": "test_session",
            "chat_history": [],
            "relevant_documents": []
        }

        node(state)

        self.mock_model_router.get_model.assert_called_once_with(
            ModelType.CHAT,
            strategy=RoutingStrategy.DEFAULT
        )

    def test_build_context_empty_documents(self):
        """测试空文档上下文构建"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "回答"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_generate_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "问题",
            "session_id": "test_session",
            "chat_history": [],
            "relevant_documents": []
        }

        node(state)

        # 验证prompt调用时传入了正确的context
        call_args = self.mock_prompt_port.get_prompt.call_args
        assert call_args[1]["context"] == "无相关文档"

    def test_build_context_multiple_documents(self):
        """测试多文档上下文拼接"""
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = "回答"
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_generate_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "问题",
            "session_id": "test_session",
            "chat_history": [],
            "relevant_documents": [
                {"content": "文档1内容"},
                {"content": "文档2内容"}
            ]
        }

        node(state)

        call_args = self.mock_prompt_port.get_prompt.call_args
        assert call_args[1]["context"] == "文档1内容\n\n文档2内容"

    def test_returns_answer_in_result(self):
        """测试返回值包含answer字段"""
        expected_answer = "这是基于检索内容生成的回答"
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.content = expected_answer
        self.mock_llm.invoke.return_value = mock_ai_message
        self.mock_prompt_port.get_prompt.return_value = Mock()

        node = create_generate_node(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger
        )

        state: AgentState = {
            "query": "问题",
            "session_id": "test_session",
            "chat_history": [],
            "relevant_documents": [{"content": "相关内容"}]
        }

        result = node(state)

        assert "answer" in result
        assert result["answer"] == expected_answer
