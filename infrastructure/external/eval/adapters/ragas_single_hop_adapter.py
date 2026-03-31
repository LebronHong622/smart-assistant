"""
Ragas单跳测试生成适配器
实现domain层ITestDatasetGenerator接口，整合所有组件
"""
from typing import List, Any, Optional
import pandas as pd
from langchain_core.documents import Document

from domain.shared.ports.test_dataset_generator_port import (
    ITestDatasetGenerator,
    TestDatasetGenerationConfig
)
from domain.shared.ports.logger_port import LoggerPort
from config.eval_settings import TestDatasetConfig, EvalSettings
from infrastructure.external.eval.factories.ragas_llm_factory import RagasLLMFactory
from infrastructure.external.eval.factories.ragas_embedding_factory import RagasEmbeddingFactory
from infrastructure.external.eval.generators.single_hop_generator import SingleHopGenerator
from infrastructure.rag.factory.rag_component_factory import RAGComponentFactory
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class RagasSingleHopAdapter(ITestDatasetGenerator):
    """
    Ragas单跳测试生成适配器
    实现ITestDatasetGenerator接口，整合：
    - YAML配置加载
    - LLM/Embedding工厂创建
    - 使用现有RAGComponentFactory加载文档和分割
    - 单跳生成
    """

    def __init__(
        self,
        config_path: str,
        logger: Optional[LoggerPort] = None,
    ):
        self.config_path = config_path
        self.logger = logger or get_app_logger()
        self._initialized = False
        self._config: Optional[TestDatasetConfig] = None
        self._generator: Optional[SingleHopGenerator] = None

    def generate_from_documents(
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
            documents = self._load_documents()

        # 调用生成器
        self.logger.info(f"开始生成，文档数量: {len(documents)}")
        df = self._generator.generate_from_documents(documents)
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
        self.logger.info(f"初始化RagasSingleHopAdapter，配置文件: {self.config_path}")

        # 1. 加载配置
        settings = EvalSettings(self.config_path)
        self._config = settings.config

        # 2. 创建LLM和Embedding
        llm = RagasLLMFactory.from_config(self._config.llm)
        embedding = RagasEmbeddingFactory.from_config(self._config.embedding)

        # 3. 创建单跳生成器
        self._generator = SingleHopGenerator(
            llm=llm,
            embedding=embedding,
            config=self._config.generation,
            keyword_config=self._config.keyword_extraction,
        )

        self._initialized = True
        self.logger.info("RagasSingleHopAdapter初始化完成")

    def _load_documents(self) -> List[Document]:
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
        documents = loader.load_documents()

        # 获取分割器
        splitter = splitter_factory.create_splitter(
            chunk_size=config.splitter.chunk_size,
            chunk_overlap=config.splitter.chunk_overlap,
            separators=config.splitter.separators,
        )

        # 分割文档
        split_documents = splitter.split_documents(documents)

        self.logger.info(f"加载并分割完成，原始文档: {len(documents)}, 分割后: {len(split_documents)}")

        return split_documents
