"""
Ragas单跳测试生成适配器 - 两阶段架构
实现domain层ITestDatasetGenerator接口，整合所有组件
"""
from typing import List, Any, Optional, Callable
import pandas as pd
from domain.entity.document.document import Document

from domain.shared.ports.test_dataset_generator_port import (
    ITestDatasetGenerator,
    TestDatasetGenerationConfig
)
from domain.shared.ports.logger_port import LoggerPort
from config.eval_settings import TestDatasetConfig, EvalSettings
from infrastructure.external.eval.factories.ragas_llm_factory import RagasLLMFactory
from infrastructure.external.eval.factories.ragas_embedding_factory import RagasEmbeddingFactory
from ragas.testset import BaseSynthesizer, Scenario
from ragas.testset.synthesizers.base import Example
from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class RagasSingleHopAdapter(BaseSynthesizer[Scenario], ITestDatasetGenerator):
    """
    Ragas单跳测试生成适配器 - 两阶段架构
    继承BaseSynthesizer[Scenario]并实现ITestDatasetGenerator接口
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
        # 初始化BaseSynthesizer
        super().__init__()
        
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
    ) -> pd.DataFrame:
        """从文档列表生成测试数据集

        Args:
            documents: 如果为空，从配置路径加载；否则使用传入的文档
            config: 生成配置

        Returns:
            生成的测试数据集DataFrame
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
        df = await self._generate_samples(prepared_data, config)

        self.logger.info(f"生成完成，共 {len(df)} 个问题")

        return df

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
        loader = loader_factory.create_loader(
            input_dir=config.documents.input_dir,
            file_pattern=config.documents.file_pattern,
            recursive=config.documents.recursive,
        )

        # 加载文档
        documents = await loader.aload_documents()

        # 获取分割器
        splitter = splitter_factory.create_splitter(
            chunk_size=config.splitter.chunk_size,
            chunk_overlap=config.splitter.chunk_overlap,
            separators=config.splitter.separators,
        )

        # 分割文档
        split_documents = await splitter.asplit_documents(documents)

        self.logger.info(f"加载并分割完成，原始文档: {len(documents)}, 分割后: {len(split_documents)}")

        return split_documents

    async def generate_with_prepared_data(
        self,
        prepared_data: Any,
        num_questions: Optional[int] = None
    ) -> pd.DataFrame:
        """使用已准备的数据生成测试集（支持分阶段调用）

        Args:
            prepared_data: 阶段1准备的数据 (PreparedData类型)
            num_questions: 问题数量

        Returns:
            测试数据集DataFrame
        """
        if not self._initialized:
            self._initialize()

        self.logger.info("使用已准备的数据生成测试样本...")
        df = await self._generate_samples(prepared_data, num_questions=num_questions)
        self.logger.info(f"生成完成，共 {len(df)} 个问题")

        return df

    async def _generate_samples(
        self,
        prepared_data: Any,
        config: Optional[TestDatasetGenerationConfig] = None,
        num_questions: Optional[int] = None
    ) -> pd.DataFrame:
        """两阶段生成样本：先生成场景，再循环生成样本
        
        Args:
            prepared_data: 准备好的数据
            config: 生成配置
            num_questions: 问题数量
            
        Returns:
            测试数据集DataFrame
        """
        # 阶段1: 生成场景
        num_questions = num_questions or (config.num_questions if config and hasattr(config, 'num_questions') else 10)
        
        self.logger.info(f"生成 {num_questions} 个场景...")
        scenarios = await self._synthesizer.generate_scenarios(num_questions=num_questions)
        self.logger.info(f"场景生成完成，共 {len(scenarios)} 个场景")
        
        # 阶段2: 循环生成样本
        samples = []
        for i, scenario in enumerate(scenarios):
            self.logger.debug(f"生成样本 {i+1}/{len(scenarios)}, 场景: {scenario}")
            sample = await self._synthesizer.generate_sample(scenario=scenario)
            samples.extend(sample)
        
        # 转换为DataFrame
        return self._samples_to_dataframe(samples)

    def _samples_to_dataframe(self, samples: List[Example]) -> pd.DataFrame:
        """将样本列表转换为DataFrame
        
        Args:
            samples: 样本列表
            
        Returns:
            DataFrame
        """
        rows = []
        for s in samples:
            rows.append({
                "question": s.question,
                "contexts": s.contexts,
                "ground_truth": s.ground_truth,
                "episode_done": s.episode_done,
            })
        return pd.DataFrame(rows)
