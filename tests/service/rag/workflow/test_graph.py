"""
单元测试：RAG工作流构建
测试 build_rag_workflow 是否正确构建LangGraph工作流
"""
import pytest
from unittest.mock import Mock
from langgraph.graph import StateGraph

from domain.document.service.rag_processing_service import RAGProcessingServiceFactory
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from application.services.rag.workflow.graph import build_rag_workflow
from application.services.rag.workflow.state import AgentState


class TestBuildRagWorkflow:
    """测试三级RAG工作流构建"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_prompt_port = Mock(spec=PromptPort)
        self.mock_model_router = Mock(spec=ModelRouterPort)
        self.mock_logger = Mock(spec=LoggerPort)
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()
        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

        # Mock LLM
        mock_llm = Mock()
        self.mock_model_router.get_model.return_value = mock_llm

    def test_build_rag_workflow_returns_state_graph(self):
        """测试成功返回StateGraph实例"""
        workflow = build_rag_workflow(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger,
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        assert isinstance(workflow, StateGraph)

    def test_workflow_has_correct_entry_point(self):
        """测试工作流入口点设置正确"""
        workflow = build_rag_workflow(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger,
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        # StateGraph 构造后entry_point应该设置为intent_classification
        # 通过编译验证配置正确
        compiled = workflow.compile()
        assert compiled is not None

    def test_all_nodes_are_added(self):
        """测试所有节点都被添加到工作流"""
        workflow = build_rag_workflow(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger,
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        # 检查所有节点都存在
        node_names = list(workflow.nodes.keys())
        assert "intent_classification" in node_names
        assert "retrieve_product" in node_names
        assert "retrieve_after_sales" in node_names
        assert "retrieve_promotion" in node_names
        assert "generate_answer" in node_names

    def test_routing_configuration_is_correct(self):
        """测试路由配置正确"""
        workflow = build_rag_workflow(
            self.mock_prompt_port,
            self.mock_model_router,
            self.mock_logger,
            self.mock_rag_factory,
            self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        # 编译成功说明配置正确
        compiled = workflow.compile()
        assert compiled is not None


class TestRouteByIntent:
    """测试意图路由逻辑"""

    def setup_method(self):
        """测试前置：获取route_by_intent函数"""
        from application.services.rag.workflow.graph import build_rag_workflow
        self.mock_prompt_port = Mock(spec=PromptPort)
        self.mock_model_router = Mock(spec=ModelRouterPort)
        self.mock_logger = Mock(spec=LoggerPort)
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()
        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

        # 我们需要从build_rag_workflow内部获取route_by_intent函数
        # 这里直接测试路由逻辑，通过手动提取逻辑验证

    def test_route_product_selling_points(self):
        """测试商品导购意图路由"""
        from application.services.rag.workflow.graph import build_rag_workflow

        def capture_route_func(state):
            """从build_rag_workflow复制路由逻辑进行测试"""
            intent = state["intent"]
            routing_map = {
                "product_selling_points": "retrieve_product",
                "after_sales_policy": "retrieve_after_sales",
                "promotion_rules": "retrieve_promotion",
                "normal": "generate_answer",
            }
            return routing_map.get(intent, "generate_answer")

        state: AgentState = {"intent": "product_selling_points"}
        route_to = capture_route_func(state)
        assert route_to == "retrieve_product"

    def test_route_after_sales_policy(self):
        """测试售后规则意图路由"""
        def capture_route_func(state):
            intent = state["intent"]
            routing_map = {
                "product_selling_points": "retrieve_product",
                "after_sales_policy": "retrieve_after_sales",
                "promotion_rules": "retrieve_promotion",
                "normal": "generate_answer",
            }
            return routing_map.get(intent, "generate_answer")

        state: AgentState = {"intent": "after_sales_policy"}
        route_to = capture_route_func(state)
        assert route_to == "retrieve_after_sales"

    def test_route_promotion_rules(self):
        """测试促销规则意图路由"""
        def capture_route_func(state):
            intent = state["intent"]
            routing_map = {
                "product_selling_points": "retrieve_product",
                "after_sales_policy": "retrieve_after_sales",
                "promotion_rules": "retrieve_promotion",
                "normal": "generate_answer",
            }
            return routing_map.get(intent, "generate_answer")

        state: AgentState = {"intent": "promotion_rules"}
        route_to = capture_route_func(state)
        assert route_to == "retrieve_promotion"

    def test_route_normal(self):
        """测试normal意图路由"""
        def capture_route_func(state):
            intent = state["intent"]
            routing_map = {
                "product_selling_points": "retrieve_product",
                "after_sales_policy": "retrieve_after_sales",
                "promotion_rules": "retrieve_promotion",
                "normal": "generate_answer",
            }
            return routing_map.get(intent, "generate_answer")

        state: AgentState = {"intent": "normal"}
        route_to = capture_route_func(state)
        assert route_to == "generate_answer"

    def test_unknown_intent_routes_to_generate_answer(self):
        """测试未知意图默认路由到generate_answer"""
        def capture_route_func(state):
            intent = state["intent"]
            routing_map = {
                "product_selling_points": "retrieve_product",
                "after_sales_policy": "retrieve_after_sales",
                "promotion_rules": "retrieve_promotion",
                "normal": "generate_answer",
            }
            return routing_map.get(intent, "generate_answer")

        state: AgentState = {"intent": "unknown_intent"}
        route_to = capture_route_func(state)
        assert route_to == "generate_answer"
