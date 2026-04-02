"""
Ragas单跳测试生成适配器 - 两阶段架构
实现domain层ITestDatasetGenerator接口，整合所有组件
"""
from typing import List, Any, Optional
import pandas as pd
from domain.entity.document.document import Document
from domain.entity.eval.generated_test_sample import (
    GeneratedTestSample,
    GeneratedTestDataset
)

from domain.shared.ports.test_dataset_generator_port import (
    ITestDatasetGenerator,
    TestDatasetGenerationConfig
)
from domain.shared.ports.logger_port import LoggerPort
from config.eval_settings import TestDatasetConfig, EvalSettings
from infrastructure.external.eval.factories.ragas_llm_factory import RagasLLMFactory
from infrastructure.external.eval.factories.ragas_embedding_factory import RagasEmbeddingFactory
from ragas.testset.synthesizers.base import BaseSynthesizer, Scenario
from ragas.dataset_schema import SingleTurnSample
from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class RagasSingleHopAdapter(ITestDatasetGenerator):
    """
    Ragas单跳测试生成适配器 - 两阶段架构
    实现ITestDatasetGenerator接口，使用组合方式持有Ragas合成器
    整合：
    - YAML配置加载
    - LLM/Embedding工厂创建
    - 使用现有RAGComponentFactory加载文档和分割
    - 两阶段生成：场景生成 + 样本生成
    """

    def __init__(
        self,
        config_path: str,
        logger: Optional[LoggerPort] = None,
        preparer: Optional[Any] = None,
        synthesizer: Optional[BaseSynthesizer[Scenario]] = None,
    ):
        self.config_path = config_path
        self.logger = logger or get_app_logger()
        self._initialized = False
        self._config: Optional[TestDatasetConfig] = None
        self._preparer: Optional[Any] = preparer
        self._synthesizer: Optional[BaseSynthesizer[Scenario]] = synthesizer

    async def generate_from_documents(
        self,
        documents: List[Any],
        config: TestDatasetGenerationConfig
    ) -> GeneratedTestDataset:
        """从文档列表生成测试数据集

        Args:
            documents: 如果为空，从配置路径加载；否则使用传入的文档
            config: 生成配置

        Returns:
            生成的测试数据集领域实体
        """
        if not self._initialized:
            self._initialize()

        # 如果documents为空，从配置路径加载
        if not documents:
            documents = await self._load_documents()

        # 调用两阶段生成
        self.logger.info(f"开始生成，文档数量: {len(documents)}")

        # 阶段1: 准备数据
        self.logger.info("阶段1: 准备知识图谱和角色列表...")
        prepared_data = self._preparer.prepare(documents)
        self.logger.info(
            f"准备完成: {len(prepared_data.knowledge_graph.nodes)} 个节点, "
            f"{len(prepared_data.persona_list)} 个角色"
        )

        # 阶段2: 生成样本 - 两阶段方式
        self.logger.info("阶段2: 生成测试样本...")
        dataset = await self._generate_samples(prepared_data, config)

        self.logger.info(f"生成完成，共 {dataset.count} 个问题")

        return dataset

    def validate_generated_dataset(
        self,
        df: pd.DataFrame
    ) -> tuple[bool, List[str]]:
        """验证生成的数据集格式是否正确

        Args:
            df: 生成的数据集

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        required_columns = ["question", "contexts", "ground_truth"]

        for col in required_columns:
            if col not in df.columns:
                errors.append(f"缺少必需列: {col}")

        if df.empty:
            errors.append("数据集为空")

        # 检查空值
        for col in required_columns:
            if col in df.columns and df[col].isna().any():
                count = df[col].isna().sum()
                errors.append(f"列 {col} 包含 {count} 个空值")

        return (len(errors) == 0, errors)

    def _initialize(self) -> None:
        """懒加载初始化所有组件"""
        if self._initialized:
            return

        self.logger.info(f"初始化RagasSingleHopAdapter，配置文件: {self.config_path}")

        # 1. 加载配置
        settings = EvalSettings(self.config_path)
        self._config = settings.config

        # 2. 创建LLM和Embedding
        llm = RagasLLMFactory.from_config(self._config.llm)
        embedding = RagasEmbeddingFactory.from_config(self._config.embedding)

        # 3. 创建阶段1: 数据准备器（如果未注入）
        if self._preparer is None:
            # 动态导入，避免循环依赖
            from infrastructure.external.eval.generators.single_hop_generator import (
                TestDatasetPreparer,
                SynthesizerConfig
            )
            self._preparer = TestDatasetPreparer(
                llm=llm,
                embedding=embedding,
                config=self._config.generation,
            )

        # 4. 创建阶段2: 合成器（如果未注入）
        if self._synthesizer is None:
            # 动态导入
            from infrastructure.external.eval.generators.single_hop_generator import (
                ConfigurableSingleHopSynthesizer
            )
            self._synthesizer = ConfigurableSingleHopSynthesizer(llm=llm)

        self._initialized = True
        self.logger.info("RagasSingleHopAdapter初始化完成")

    async def _load_documents(self) -> List[Document]:
        """使用现有RAG组件加载并分割文档

        Returns:
            分割后的LangChain文档列表
        """
        config = self._config
        loader_factory = RAGComponentFactory.get_loader_factory()
        splitter_factory = RAGComponentFactory.get_splitter_factory()

        # 获取加载器
        documents = await loader_factory.aload_documents(
            loader_type=config.documents.file_pattern,
            file_path=config.documents.input_dir,
        )

        # 分割文档
        split_documents = await splitter_factory.asplit_documents(documents)

        self.logger.info(f"加载并分割完成，原始文档: {len(documents)}, 分割后: {len(split_documents)}")

        return split_documents

    async def generate_with_prepared_data(
        self,
        prepared_data: Any,
        num_questions: Optional[int] = None
    ) -> GeneratedTestDataset:
        """使用已准备的数据生成测试集（支持分阶段调用）

        Args:
            prepared_data: 阶段1准备的数据 (PreparedData类型)
            num_questions: 问题数量

        Returns:
            生成的测试数据集领域实体
        """
        if not self._initialized:
            self._initialize()

        self.logger.info("使用已准备的数据生成测试样本...")
        dataset = await self._generate_samples(prepared_data, num_questions=num_questions)
        self.logger.info(f"生成完成，共 {dataset.count} 个问题")

        return dataset

    async def _generate_samples(
        self,
        prepared_data: Any,
        config: Optional[TestDatasetGenerationConfig] = None,
        num_questions: Optional[int] = None
    ) -> GeneratedTestDataset:
        """两阶段生成样本：先生成场景，再循环生成样本

        Args:
            prepared_data: 准备好的数据
            config: 生成配置
            num_questions: 问题数量

        Returns:
            生成的测试数据集领域实体
        """
        # 阶段1: 生成场景
        num_questions = num_questions or (config.test_size if config else 10)

        self.logger.info(f"生成 {num_questions} 个场景...")
        scenarios = await self._synthesizer.generate_scenarios(
            n=num_questions,
            knowledge_graph=prepared_data.knowledge_graph,
            persona_list=prepared_data.persona_list
        )
        self.logger.info(f"场景生成完成，共 {len(scenarios)} 个场景")

        # 阶段2: 循环生成样本
        # 每个generate_sample返回一个SingleTurnSample，在当前Ragas版本中返回元组列表[(name, value), ...]
        samples: List[Any] = []
        for i, scenario in enumerate(scenarios):
            self.logger.debug(f"生成样本 {i+1}/{len(scenarios)}, 场景: {scenario}")
            sample = await self._synthesizer.generate_sample(scenario=scenario)
            samples.append(sample)

        # 转换为领域实体
        return self._convert_to_domain(samples)

    def _convert_to_domain(self, samples: List[List[tuple[str, Any]]]) -> GeneratedTestDataset:
        """将Ragas样本列表转换为领域实体

        Args:
            samples: 每个样本是Ragas SingleTurnSample的元组列表格式 [(name, value), ...]
            在当前Ragas版本中，generate_sample返回这种格式

        Returns:
            GeneratedTestDataset 领域实体
        """
        domain_samples: List[GeneratedTestSample] = []
        for sample_tuples in samples:
            # 元组列表格式转换为字典
            sample_dict = dict(sample_tuples)
            question = sample_dict.get("user_input", "")
            contexts = sample_dict.get("reference_contexts", [])
            ground_truth = sample_dict.get("reference", "")
            episode_done = sample_dict.get("episode_done")

            domain_sample = GeneratedTestSample(
                question=question or "",
                contexts=contexts or [],
                ground_truth=ground_truth or "",
                episode_done=episode_done
            )
            domain_samples.append(domain_sample)

        return GeneratedTestDataset(samples=domain_samples)
