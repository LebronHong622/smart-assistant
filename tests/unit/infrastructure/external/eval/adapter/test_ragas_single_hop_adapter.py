"""
RagasSingleHopAdapter 单元测试
"""
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
from typing import List
from infrastructure.external.eval.adapters.ragas_single_hop_adapter import RagasSingleHopAdapter
from domain.shared.ports.test_dataset_generator_port import TestDatasetGenerationConfig
from domain.entity.eval.generated_test_sample import GeneratedTestDataset, GeneratedTestSample
from domain.entity.document.document import Document


class TestRagasSingleHopAdapter:
    """RagasSingleHopAdapter 的单元测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config_path = "test_config.yaml"
        self.mock_logger = Mock()
        self.adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger
        )

    def test_initialization(self):
        """测试适配器初始化"""
        assert self.adapter.config_path == self.config_path
        assert self.adapter.logger == self.mock_logger
        assert self.adapter._initialized is False
        assert self.adapter._config is None
        assert self.adapter._preparer is None
        assert self.adapter._synthesizer is None

    def test_initialization_with_dependency_injection(self):
        """测试带依赖注入的初始化"""
        mock_preparer = Mock()
        mock_synthesizer = Mock()

        adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger,
            preparer=mock_preparer,
            synthesizer=mock_synthesizer
        )

        assert adapter._preparer == mock_preparer
        assert adapter._synthesizer == mock_synthesizer

    def test_validate_generated_dataset_valid(self):
        """测试验证有效的数据集"""
        data = {
            "question": ["question1", "question2"],
            "contexts": [["context1"], ["context2"]],
            "ground_truth": ["answer1", "answer2"]
        }
        df = pd.DataFrame(data)

        is_valid, errors = self.adapter.validate_generated_dataset(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_generated_dataset_missing_columns(self):
        """测试验证缺少列的数据集"""
        data = {
            "question": ["question1"],
            "contexts": [["context1"]]
        }
        df = pd.DataFrame(data)

        is_valid, errors = self.adapter.validate_generated_dataset(df)

        assert is_valid is False
        assert "缺少必需列: ground_truth" in errors

    def test_validate_generated_dataset_empty(self):
        """测试验证空数据集"""
        data = {
            "question": [],
            "contexts": [],
            "ground_truth": []
        }
        df = pd.DataFrame(data)

        is_valid, errors = self.adapter.validate_generated_dataset(df)

        assert is_valid is False
        assert "数据集为空" in errors

    def test_validate_generated_dataset_with_null_values(self):
        """测试验证包含空值的数据集"""
        data = {
            "question": ["question1", None],
            "contexts": [["context1"], ["context2"]],
            "ground_truth": ["answer1", "answer2"]
        }
        df = pd.DataFrame(data)

        is_valid, errors = self.adapter.validate_generated_dataset(df)

        assert is_valid is False
        assert any("列 question 包含 1 个空值" in error for error in errors)

    def test_validate_generated_dataset_multiple_errors(self):
        """测试验证多个错误的情况"""
        data = {
            "question": [None, "question2"]
        }
        df = pd.DataFrame(data)

        is_valid, errors = self.adapter.validate_generated_dataset(df)

        assert is_valid is False
        assert len(errors) == 3  # missing contexts, missing ground_truth, and question contains null

    def test_convert_to_domain(self):
        """测试将Ragas样本转换为领域实体"""
        # 创建模拟Example对象
        sample1 = Mock()
        sample1.user_input = "What is Python?"
        sample1.reference_contexts = ["Python is a programming language."]
        sample1.reference = "Python is a high-level programming language."
        sample1.episode_done = False

        sample2 = Mock()
        sample2.user_input = "Who created Python?"
        sample2.reference_contexts = ["Python was created by Guido van Rossum."]
        sample2.reference = "Guido van Rossum created Python."
        sample2.episode_done = False

        samples: List[Mock] = [sample1, sample2]

        dataset = self.adapter._convert_to_domain(samples)

        assert isinstance(dataset, GeneratedTestDataset)
        assert dataset.count == 2
        assert len(dataset.samples) == 2
        assert dataset.samples[0].question == "What is Python?"
        assert dataset.samples[1].question == "Who created Python?"
        assert dataset.samples[0].episode_done == False

    @pytest.mark.asyncio
    async def test_generate_from_documents_with_provided_documents(self):
        """测试从提供的文档生成测试集"""
        # 创建模拟依赖
        mock_preparer = Mock()
        mock_prepared_data = Mock()
        mock_prepared_data.knowledge_graph = Mock()
        mock_prepared_data.knowledge_graph.nodes = [1, 2, 3]
        mock_prepared_data.persona_list = [Mock(), Mock()]
        mock_preparer.prepare.return_value = mock_prepared_data

        mock_synthesizer = Mock()
        mock_synthesizer.generate_scenarios = AsyncMock()
        mock_synthesizer.generate_scenarios.return_value = [Mock(), Mock()]
        mock_synthesizer.generate_sample = AsyncMock()

        sample1 = Mock()
        sample1.user_input = "q1"
        sample1.reference_contexts = ["c1"]
        sample1.reference = "a1"
        sample1.episode_done = False
        sample2 = Mock()
        sample2.user_input = "q2"
        sample2.reference_contexts = ["c2"]
        sample2.reference = "a2"
        sample2.episode_done = False

        mock_synthesizer.generate_sample.return_value = [sample1, sample2]

        # 创建适配器并标记为已初始化
        adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger,
            preparer=mock_preparer,
            synthesizer=mock_synthesizer
        )
        adapter._initialized = True
        adapter._config = Mock()

        # 创建测试文档和配置
        test_docs = [Document(id=1, content="test content 1"), Document(id=2, content="test content 2")]
        config = TestDatasetGenerationConfig(test_size=2)

        # 执行测试
        dataset = await adapter.generate_from_documents(test_docs, config)

        # 验证结果
        assert isinstance(dataset, GeneratedTestDataset)
        assert dataset.count == 4  # 2 scenarios * 2 samples each
        assert len(dataset.samples) == 4
        mock_preparer.prepare.assert_called_once_with(test_docs)
        assert self.mock_logger.info.called

    @pytest.mark.asyncio
    async def test_generate_with_prepared_data(self):
        """测试使用已准备数据生成测试集"""
        # 创建模拟依赖
        mock_synthesizer = Mock()
        mock_synthesizer.generate_scenarios = AsyncMock()
        mock_synthesizer.generate_scenarios.return_value = [Mock()]
        mock_synthesizer.generate_sample = AsyncMock()

        sample = Mock()
        sample.user_input = "q1"
        sample.reference_contexts = ["c1"]
        sample.reference = "a1"
        sample.episode_done = False

        mock_synthesizer.generate_sample.return_value = [sample]

        # 创建适配器
        adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger,
            synthesizer=mock_synthesizer
        )
        adapter._initialized = True
        adapter._preparer = Mock()

        # 执行测试
        prepared_data = Mock()
        dataset = await adapter.generate_with_prepared_data(prepared_data, num_questions=1)

        # 验证
        assert isinstance(dataset, GeneratedTestDataset)
        assert dataset.count == 1
        assert len(dataset.samples) == 1
        mock_synthesizer.generate_scenarios.assert_called_once()
        assert self.mock_logger.info.called

    @pytest.mark.asyncio
    async def test_generate_samples_default_num_questions(self):
        """测试生成样本时使用默认问题数量"""
        # 创建模拟依赖
        mock_synthesizer = Mock()
        mock_synthesizer.generate_scenarios = AsyncMock()
        mock_synthesizer.generate_scenarios.return_value = []
        mock_synthesizer.generate_sample = AsyncMock()
        mock_synthesizer.generate_sample.return_value = []

        adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger,
            synthesizer=mock_synthesizer
        )
        adapter._initialized = True

        # 执行测试 - 不提供config和num_questions
        prepared_data = Mock()
        dataset = await adapter._generate_samples(prepared_data)

        # 验证默认值10被使用（generate_scenarios应该被调用）
        assert isinstance(dataset, GeneratedTestDataset)
        assert dataset.count == 0
        mock_synthesizer.generate_scenarios.assert_called_once()

    def test_initialize_already_initialized(self):
        """测试已初始化时调用_initialize不做任何操作"""
        self.adapter._initialized = True
        initial_state = self.adapter._config

        self.adapter._initialize()

        assert self.adapter._initialized is True
        assert self.adapter._config == initial_state
        # logger只在第一次初始化时调用，这里不应该调用
        assert self.mock_logger.info.call_count == 0

    @patch("infrastructure.external.eval.adapters.ragas_single_hop_adapter.EvalSettings")
    @patch("infrastructure.external.eval.adapters.ragas_single_hop_adapter.RagasLLMFactory")
    @patch("infrastructure.external.eval.adapters.ragas_single_hop_adapter.RagasEmbeddingFactory")
    def test_initialize_success(self, mock_embedding_factory, mock_llm_factory, mock_eval_settings):
        """测试成功初始化"""
        # 设置模拟
        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.embedding = Mock()
        mock_config.generation = Mock()
        mock_config.documents = Mock()
        mock_config.documents.input_dir = "test_dir"
        mock_config.documents.file_pattern = "*.txt"
        mock_config.documents.recursive = True
        mock_config.splitter = Mock()
        mock_config.splitter.chunk_size = 1000
        mock_config.splitter.chunk_overlap = 200
        mock_config.splitter.separators = ["\n\n"]

        mock_settings = Mock()
        mock_settings.config = mock_config
        mock_eval_settings.return_value = mock_settings

        mock_llm = Mock()
        mock_llm_factory.from_config.return_value = mock_llm

        mock_embedding = Mock()
        mock_embedding_factory.from_config.return_value = mock_embedding

        # 执行初始化
        self.adapter._initialize()

        # 验证
        assert self.adapter._initialized is True
        assert self.adapter._config == mock_config
        mock_eval_settings.assert_called_once_with(self.config_path)
        mock_llm_factory.from_config.assert_called_once_with(mock_config.llm)
        mock_embedding_factory.from_config.assert_called_once_with(mock_config.embedding)
        assert self.mock_logger.info.called

    @pytest.mark.asyncio
    @patch("infrastructure.external.eval.adapters.ragas_single_hop_adapter.RAGComponentFactory")
    async def test_load_documents_success(self, mock_rag_factory):
        """测试成功加载文档"""
        # 设置模拟
        mock_loader_factory = Mock()
        mock_loader = Mock()
        mock_loader.aload_documents = AsyncMock()
        mock_doc1 = Document(id=1, content="content 1")
        mock_doc2 = Document(id=2, content="content 2")
        mock_loader.aload_documents.return_value = [mock_doc1, mock_doc2]
        mock_loader_factory.create_loader.return_value = mock_loader

        mock_splitter_factory = Mock()
        mock_splitter = Mock()
        mock_splitter.asplit_documents = AsyncMock()
        mock_split_doc1 = Document(id=1, content="split 1a")
        mock_split_doc2 = Document(id=2, content="split 1b")
        mock_split_doc3 = Document(id=3, content="split 2a")
        mock_splitter.asplit_documents.return_value = [mock_split_doc1, mock_split_doc2, mock_split_doc3]
        mock_splitter_factory.create_splitter.return_value = mock_splitter

        mock_rag_factory.get_loader_factory.return_value = mock_loader_factory
        mock_rag_factory.get_splitter_factory.return_value = mock_splitter_factory

        # 设置配置
        mock_config = Mock()
        mock_config.documents = Mock()
        mock_config.documents.input_dir = "test/input"
        mock_config.documents.file_pattern = "*.md"
        mock_config.documents.recursive = True
        mock_config.splitter = Mock()
        mock_config.splitter.chunk_size = 500
        mock_config.splitter.chunk_overlap = 50
        mock_config.splitter.separators = ["\n\n", "\n"]

        self.adapter._config = mock_config
        self.adapter._initialized = True

        # 执行
        result = await self.adapter._load_documents()

        # 验证
        assert len(result) == 3
        mock_loader_factory.create_loader.assert_called_once_with(
            input_dir="test/input",
            file_pattern="*.md",
            recursive=True
        )
        mock_splitter_factory.create_splitter.assert_called_once_with(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n"]
        )
        mock_loader.aload_documents.assert_called_once()
        mock_splitter.asplit_documents.assert_called_once_with([mock_doc1, mock_doc2])
        assert self.mock_logger.info.called

    @pytest.mark.asyncio
    async def test_generate_samples_correct_sample_collection(self):
        """测试生成样本正确收集结果"""
        # 创建模拟
        scenario1 = Mock()
        scenario2 = Mock()

        mock_synthesizer = Mock()
        mock_synthesizer.generate_scenarios = AsyncMock()
        mock_synthesizer.generate_scenarios.return_value = [scenario1, scenario2]
        mock_synthesizer.generate_sample = AsyncMock()

        # 第一个场景生成1个样本
        sample1 = Mock()
        sample1.user_input = "q1"
        sample1.reference_contexts = ["c1"]
        sample1.reference = "a1"
        sample1.episode_done = False

        # 第二个场景生成2个样本
        sample2 = Mock()
        sample2.user_input = "q2"
        sample2.reference_contexts = ["c2"]
        sample2.reference = "a2"
        sample2.episode_done = False
        sample3 = Mock()
        sample3.user_input = "q3"
        sample3.reference_contexts = ["c3"]
        sample3.reference = "a3"
        sample3.episode_done = False

        mock_synthesizer.generate_sample.side_effect = [[sample1], [sample2, sample3]]

        adapter = RagasSingleHopAdapter(
            config_path=self.config_path,
            logger=self.mock_logger,
            synthesizer=mock_synthesizer
        )
        adapter._initialized = True

        # 执行
        prepared_data = Mock()
        config = TestDatasetGenerationConfig(test_size=2)
        dataset = await adapter._generate_samples(prepared_data, config)

        # 验证总共3个样本
        assert dataset.count == 3
        assert len(dataset.samples) == 3
        questions = [sample.question for sample in dataset.samples]
        assert questions == ["q1", "q2", "q3"]
        assert mock_synthesizer.generate_sample.call_count == 2
