"""
单跳生成器单元测试 - 两阶段架构
测试 TestDatasetPreparer 和 ConfigurableSingleHopSynthesizer
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from domain.entity.document.document import Document
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.base import SingleHopScenario
from ragas.testset.synthesizers.base import QueryStyle, QueryLength
from config.eval_settings import GenerationConfig, TransformConfig, RoleConfig, SingleHopConfig
from infrastructure.external.eval.generators.single_hop_generator import (
    TestDatasetPreparer,
    ConfigurableSingleHopSynthesizer,
    SynthesizerConfig,
    PreparedData,
    generate_scenarios,
)


class TestTestDatasetPreparer:
    """阶段1: 数据准备器单元测试"""

    def setup_method(self):
        """测试初始化"""
        self.mock_llm = Mock(spec=BaseRagasLLM)
        self.mock_embedding = Mock(spec=BaseRagasEmbeddings)

    def test_prepare_returns_prepared_data(self):
        """测试prepare返回PreparedData对象"""
        config = GenerationConfig(
            transforms={},
            roles={
                "tester": RoleConfig(
                    enabled=True,
                    questions_per_doc=1,
                    description="测试角色"
                )
            }
        )

        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        docs = [Document(content="测试内容", metadata={"doc_id": "1"})]
        prepared = preparer.prepare(docs)

        assert isinstance(prepared, PreparedData)
        assert isinstance(prepared.knowledge_graph, KnowledgeGraph)
        assert isinstance(prepared.persona_list, list)
        assert len(prepared.persona_list) == 1
        assert prepared.config == config
        assert prepared.documents == docs

    def test_build_knowledge_graph_with_documents(self):
        """测试构建知识图谱包含所有文档"""
        config = GenerationConfig(transforms={}, roles={})
        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        docs = [
            Document(content="内容1", metadata={"doc_id": "1"}),
            Document(content="内容2", metadata={"doc_id": "2"}),
        ]

        kg = preparer._build_knowledge_graph(docs)

        assert len(kg.nodes) == 2
        assert kg.nodes[0].properties.get("page_content") == "内容1"
        assert kg.nodes[1].properties.get("page_content") == "内容2"

    def test_build_persona_list_from_roles(self):
        """测试从角色配置构建persona列表"""
        config = GenerationConfig(
            transforms={},
            roles={
                "concrete": RoleConfig(
                    enabled=True,
                    questions_per_doc=2,
                    description="具体问题提问者"
                ),
                "abstract": RoleConfig(
                    enabled=True,
                    questions_per_doc=1,
                    description="抽象问题提问者"
                ),
                "disabled": RoleConfig(
                    enabled=False,
                    questions_per_doc=1,
                    description="禁用角色"
                )
            }
        )

        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        personas = preparer._build_persona_list()

        assert len(personas) == 2
        assert personas[0].name == "concrete"
        assert personas[1].name == "abstract"

    def test_load_transforms_with_llm_injection(self):
        """测试加载transform并注入LLM"""
        transform_config = TransformConfig(
            enable=True,
            class_name="KeyphrasesExtractor",
            module="ragas.testset.transforms",
            parameters={"is_llm": True, "max_num": 10}
        )
        config = GenerationConfig(
            transforms={"keyphrases": transform_config},
            roles={}
        )

        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        with patch("importlib.import_module") as mock_import:
            mock_cls = Mock()
            mock_module = Mock()
            mock_module.KeyphrasesExtractor = mock_cls
            mock_import.return_value = mock_module

            transforms = preparer._load_transforms()

            assert len(transforms) == 1
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args[1]["llm"] == self.mock_llm
            assert call_args[1]["max_num"] == 10

    def test_load_transforms_skipped_when_disabled(self):
        """测试禁用的transform被跳过"""
        transform_config = TransformConfig(
            enable=False,
            class_name="KeyphrasesExtractor",
            module="ragas.testset.transforms",
            parameters={}
        )
        config = GenerationConfig(
            transforms={"keyphrases": transform_config},
            roles={}
        )

        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        transforms = preparer._load_transforms()

        assert len(transforms) == 0

    def test_dynamic_import_module_not_found(self):
        """测试动态导入模块不存在"""
        config = GenerationConfig(transforms={}, roles={})
        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        with pytest.raises(ImportError, match="无法导入模块: non.existent.module"):
            preparer._load_class("non.existent.module", "NonExistentClass")

    def test_dynamic_import_class_not_found(self):
        """测试动态导入类不存在"""
        config = GenerationConfig(transforms={}, roles={})
        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        with pytest.raises(AttributeError, match="在模块 ragas.testset.transforms 中找不到类: NonExistentClass"):
            preparer._load_class("ragas.testset.transforms", "NonExistentClass")


class TestConfigurableSingleHopSynthesizer:
    """阶段2: 可配置合成器单元测试"""

    def setup_method(self):
        """测试初始化"""
        self.mock_llm = Mock(spec=BaseRagasLLM)

    @pytest.mark.asyncio
    async def test_generate_scenarios_returns_single_hop_scenario_list(self):
        """测试_generate_scenarios返回SingleHopScenario对象列表"""
        config = SynthesizerConfig(node_property_name="keyphrases")
        synthesizer = ConfigurableSingleHopSynthesizer(
            llm=self.mock_llm,
            syntheziser_config=config,
        )

        # Mock theme_persona_matching_prompt.generate
        mock_persona_concepts = Mock()
        mock_persona_concepts.mapping = {"tester": ["主题1"]}
        synthesizer.theme_persona_matching_prompt = Mock()
        synthesizer.theme_persona_matching_prompt.generate = AsyncMock(return_value=mock_persona_concepts)

        # 创建测试知识图谱
        kg = KnowledgeGraph()
        node1 = Node(properties={"page_content": "内容1", "keyphrases": ["主题1"]})
        node2 = Node(properties={"page_content": "内容2", "keyphrases": ["主题2"]})
        kg._add_node(node1)
        kg._add_node(node2)

        # 创建测试角色
        personas = [
            Persona(name="tester", role_description="测试角色")
        ]

        scenarios = await synthesizer._generate_scenarios(
            n=4,
            knowledge_graph=kg,
            persona_list=personas,
        )

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

        # 验证场景类型 - 现在返回的是 SingleHopScenario 对象
        for scenario in scenarios:
            assert isinstance(scenario, SingleHopScenario)
            assert hasattr(scenario, 'nodes')
            assert hasattr(scenario, 'persona')
            assert hasattr(scenario, 'term')
            assert hasattr(scenario, 'style')
            assert hasattr(scenario, 'length')

    @pytest.mark.asyncio
    async def test_generate_scenarios_respects_n_limit(self):
        """测试_generate_scenarios尊重n参数限制"""
        config = SynthesizerConfig()
        synthesizer = ConfigurableSingleHopSynthesizer(
            llm=self.mock_llm,
            syntheziser_config=config,
        )

        # Mock theme_persona_matching_prompt.generate
        mock_persona_concepts = Mock()
        mock_persona_concepts.mapping = {"tester": ["主题"]}
        synthesizer.theme_persona_matching_prompt = Mock()
        synthesizer.theme_persona_matching_prompt.generate = AsyncMock(return_value=mock_persona_concepts)

        kg = KnowledgeGraph()
        for i in range(10):
            node = Node(properties={"page_content": f"内容{i}", "keyphrases": ["主题"]})
            kg._add_node(node)

        personas = [Persona(name="tester", role_description="测试")]

        scenarios = await synthesizer._generate_scenarios(
            n=5,
            knowledge_graph=kg,
            persona_list=personas,
        )

        assert len(scenarios) <= 5

    def test_extract_themes_from_keyphrases(self):
        """测试从keyphrases提取主题"""
        synthesizer = ConfigurableSingleHopSynthesizer(llm=self.mock_llm)

        node = Node(properties={
            "page_content": "测试内容",
            "keyphrases": ["主题1", "主题2", "主题3"]
        })

        themes = synthesizer._extract_themes(node)

        assert themes == ["主题1", "主题2", "主题3"]

    def test_extract_themes_fallback_to_content(self):
        """测试无keyphrases时回退到内容"""
        synthesizer = ConfigurableSingleHopSynthesizer(llm=self.mock_llm)

        node = Node(properties={"page_content": "这是一个很长的测试内容..."})

        themes = synthesizer._extract_themes(node)

        assert len(themes) > 0
        assert "这是一个很长的测试内容" in themes[0]

    def test_extract_themes_with_custom_extractor(self):
        """测试使用自定义主题提取器"""
        custom_extractor = Mock(return_value=["自定义主题"])
        config = SynthesizerConfig(theme_extractor=custom_extractor)

        synthesizer = ConfigurableSingleHopSynthesizer(
            llm=self.mock_llm,
            syntheziser_config=config,
        )

        node = Node(properties={"page_content": "内容"})
        themes = synthesizer._extract_themes(node)

        assert themes == ["自定义主题"]
        custom_extractor.assert_called_once_with(node)

    def test_set_instruction(self):
        """测试set_instruction方法"""
        synthesizer = ConfigurableSingleHopSynthesizer(llm=self.mock_llm)
        
        # Mock get_prompts and set_prompts
        mock_prompt = Mock()
        mock_prompt.instruction = "原始指令"
        synthesizer.get_prompts = Mock(return_value={"generate_query_reference_prompt": mock_prompt})
        synthesizer.set_prompts = Mock()

        # 测试链式调用
        result = synthesizer.set_instruction("自定义指令")
        
        assert result == synthesizer  # 返回self，支持链式调用
        assert mock_prompt.instruction == "自定义指令"
        synthesizer.set_prompts.assert_called_once()

    def test_set_instruction_chaining(self):
        """测试set_instruction链式调用"""
        synthesizer = ConfigurableSingleHopSynthesizer(llm=self.mock_llm)
        
        # Mock get_prompts and set_prompts
        mock_prompt = Mock()
        mock_prompt.instruction = "原始指令"
        synthesizer.get_prompts = Mock(return_value={"generate_query_reference_prompt": mock_prompt})
        synthesizer.set_prompts = Mock()

        # 测试链式调用
        result = synthesizer.set_instruction("指令1").set_instruction("指令2")
        
        assert isinstance(result, ConfigurableSingleHopSynthesizer)
        assert mock_prompt.instruction == "指令2"


class TestGenerateScenarios:
    """集成测试: 生成场景流程"""

    def setup_method(self):
        """测试初始化"""
        self.mock_llm = Mock(spec=BaseRagasLLM)
        self.mock_embedding = Mock(spec=BaseRagasEmbeddings)

    @pytest.mark.asyncio
    async def test_generate_scenarios_integration(self):
        """测试完整的场景生成流程"""
        config = GenerationConfig(
            transforms={},
            roles={
                "concrete": RoleConfig(
                    enabled=True,
                    questions_per_doc=2,
                    description="具体问题"
                )
            },
            single_hop=SingleHopConfig(max_questions_per_doc=10)
        )

        preparer = TestDatasetPreparer(
            llm=self.mock_llm,
            embedding=self.mock_embedding,
            config=config,
        )

        synthesizer = ConfigurableSingleHopSynthesizer(llm=self.mock_llm)
        
        # Mock theme_persona_matching_prompt
        mock_persona_concepts = Mock()
        mock_persona_concepts.mapping = {"concrete": ["主题"]}
        synthesizer.theme_persona_matching_prompt = Mock()
        synthesizer.theme_persona_matching_prompt.generate = AsyncMock(return_value=mock_persona_concepts)

        docs = [
            Document(content="测试内容1", metadata={"doc_id": "1"}),
            Document(content="测试内容2", metadata={"doc_id": "2"}),
        ]

        scenarios = await generate_scenarios(
            preparer=preparer,
            synthesizer=synthesizer,
            documents=docs,
            num_scenarios=5,
        )

        assert isinstance(scenarios, list)
        assert len(scenarios) <= 5
        
        # 验证返回的是 SingleHopScenario 对象
        for scenario in scenarios:
            assert isinstance(scenario, SingleHopScenario)


class TestSynthesizerConfig:
    """合成器配置单元测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SynthesizerConfig()

        assert config.node_property_name == "keyphrases"
        assert config.enable_theme_matching is True
        assert config.node_filter is None
        assert config.theme_extractor is None

    def test_custom_config(self):
        """测试自定义配置"""
        custom_filter = lambda node: True
        custom_extractor = lambda node: ["主题"]

        config = SynthesizerConfig(
            node_property_name="custom_prop",
            enable_theme_matching=False,
            node_filter=custom_filter,
            theme_extractor=custom_extractor,
        )

        assert config.node_property_name == "custom_prop"
        assert config.enable_theme_matching is False
        assert config.node_filter == custom_filter
        assert config.theme_extractor == custom_extractor
