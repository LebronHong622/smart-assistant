"""
单元测试：LangchainAgenticRagServiceImpl
只测试初始化和 execute_workflow 方法
"""
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import AIMessage

from domain.qa.service.agentic_rag_service import AgenticRagService
from domain.document.service.rag_processing_service import RAGProcessingServiceFactory
from domain.document.repository.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.prompt_port import PromptPort
from domain.shared.ports.model_router_port import ModelRouterPort
from domain.shared.ports.model_capability_port import BaseModel
from domain.shared.model_enums import ModelType
from domain.qa.value_object.rag_state import RagState
from application.services.rag.langchain_agentic_rag_service_impl import LangchainAgenticRagServiceImpl


class TestLangchainAgenticRagServiceImpl:
    """测试 Agentic RAG 主服务"""

    def setup_method(self):
        """测试前置：创建所有依赖mock"""
        self.mock_logger = Mock(spec=LoggerPort)
        self.mock_tool_port = Mock(spec=ToolPort)
        self.mock_prompt_port = Mock(spec=PromptPort)
        self.mock_model_router = Mock(spec=ModelRouterPort)
        self.mock_rag_factory = Mock(spec=RAGProcessingServiceFactory)
        self.mock_doc_repo_factory = Mock()

        # Mock LLM
        self.mock_llm = Mock(spec=BaseModel)
        self.mock_model_router.get_model.return_value = self.mock_llm

        # Mock tools
        self.mock_tool_port.get_tools.return_value = []

        # Mock document repo factory
        self.mock_doc_repo = Mock(spec=DocumentRepository)
        self.mock_doc_repo_factory.return_value = self.mock_doc_repo

    def test_initialization_success(self):
        """测试服务成功初始化，工作流成功编译"""
        service = LangchainAgenticRagServiceImpl(
            logger=self.mock_logger,
            tool_port=self.mock_tool_port,
            prompt_port=self.mock_prompt_port,
            model_router_port=self.mock_model_router,
            rag_processing_service_factory=self.mock_rag_factory,
            document_repository_factory=self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        assert service is not None
        assert hasattr(service, 'app')
        assert service.app is not None
        self.mock_logger.info.assert_called()

    def test_initialization_passes_model_router_to_workflow(self):
        """测试初始化将model_router传递给工作流构建"""
        # 由于workflow在__init__内部构建，我们只需验证初始化成功
        # 这已经证明参数传递正确
        service = LangchainAgenticRagServiceImpl(
            logger=self.mock_logger,
            tool_port=self.mock_tool_port,
            prompt_port=self.mock_prompt_port,
            model_router_port=self.mock_model_router,
            rag_processing_service_factory=self.mock_rag_factory,
            document_repository_factory=self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        assert service is not None
        assert self.mock_model_router.get_model.called
        # 首次调用是获取self.llm
        call_args = self.mock_model_router.get_model.call_args_list[0]
        assert call_args[0][0] == ModelType.CHAT

    def test_execute_workflow_returns_rag_state_with_normal_intent(self):
        """测试execute_workflow正常执行（normal意图路径）"""
        # Mock prompts for intent classification
        self.mock_prompt_port.get_prompt.side_effect = [
            # First call: intent classification
            Mock(),  # prompt_value
            # Second call: answer generation
            Mock()   # prompt_value
        ]

        # Mock LLM responses
        mock_intent_message = Mock(spec=AIMessage)
        mock_intent_message.content = "normal"

        mock_answer_message = Mock(spec=AIMessage)
        mock_answer_message.content = "这是回答"

        self.mock_llm.invoke.side_effect = [
            mock_intent_message,  # intent classification
            mock_answer_message   # answer generation
        ]

        service = LangchainAgenticRagServiceImpl(
            logger=self.mock_logger,
            tool_port=self.mock_tool_port,
            prompt_port=self.mock_prompt_port,
            model_router_port=self.mock_model_router,
            rag_processing_service_factory=self.mock_rag_factory,
            document_repository_factory=self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        result = service.execute_workflow(
            query="你好",
            session_id="test_session",
            chat_history=[]
        )

        assert isinstance(result, RagState)
        assert result.session_id == "test_session"
        assert result.query == "你好"
        self.mock_logger.info.assert_any_call("执行 Agentic RAG 工作流，session_id=test_session, query=你好")
        self.mock_logger.info.assert_any_call("工作流执行完成，session_id=test_session")

    def test_execute_workflow_returns_error_state_on_exception(self):
        """测试执行异常时返回错误状态"""
        # 让app.invoke抛出异常，我们通过mock实现
        # 需要使用MagicMock来mock编译后的app
        # 由于app在__init__内部编译，我们需要用patch或者通过构造验证
        # 这里我们通过让prompt_port抛出异常来触发错误处理

        self.mock_prompt_port.get_prompt.side_effect = Exception("模拟错误")

        service = LangchainAgenticRagServiceImpl(
            logger=self.mock_logger,
            tool_port=self.mock_tool_port,
            prompt_port=self.mock_prompt_port,
            model_router_port=self.mock_model_router,
            rag_processing_service_factory=self.mock_rag_factory,
            document_repository_factory=self.mock_doc_repo_factory,
            default_retrieve_limit=5
        )

        result = service.execute_workflow(
            query="测试错误",
            session_id="test_error",
            chat_history=[]
        )

        assert isinstance(result, RagState)
        assert result.session_id == "test_error"
        assert result.error is not None
        assert "模拟错误" in result.error
        self.mock_logger.error.assert_called()
