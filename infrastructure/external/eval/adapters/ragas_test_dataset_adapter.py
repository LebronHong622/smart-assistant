"""
Ragas测试数据集生成适配器
使用Ragas库从文档生成测试数据集
"""
from typing import List, Any, Optional
import pandas as pd
from langchain.schema import Document
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificSynthesizer,
    MultiHopSpecificSynthesizer
)

from domain.shared.ports.test_dataset_generator_port import (
    ITestDatasetGenerator,
    TestDatasetGenerationConfig
)
from domain.shared.ports.logger_port import LoggerPort
from infrastructure.core.log.adapters.logger_adapter import get_app_logger


class RagasTestDatasetAdapter(ITestDatasetGenerator):
    """Ragas测试数据集生成适配器

    使用Ragas框架从RAG文档生成评测用的测试数据集
    作为防腐层隔离Ragas框架与领域层
    """

    def __init__(
        self,
        llm: Any,
        embedding_model: Any,
        logger: Optional[LoggerPort] = None
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.logger = logger or get_app_logger()
        self._generator: Optional[TestsetGenerator] = None

    def _get_generator(self) -> TestsetGenerator:
        """获取或创建Ragas生成器实例"""
        if self._generator is None:
            self._generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embedding_model
            )
            self.logger.info("Ragas TestsetGenerator 初始化完成")
        return self._generator

    def generate_from_documents(
        self,
        documents: List[Any],
        config: TestDatasetGenerationConfig
    ) -> pd.DataFrame:
        """从文档列表生成测试数据集

        Args:
            documents: LangChain Document 列表
            config: 生成配置

        Returns:
            标准格式的DataFrame
        """
        self.logger.info(
            f"开始生成测试数据集: documents={len(documents)}, "
            f"test_size={config.test_size}, distribution={config.distribution}"
        )

        # 转换文档格式（如果需要）
        ragas_docs = self._convert_documents(documents)

        # 创建生成器
        generator = self._get_generator()

        # 根据distribution配置选择合成器
        if config.distribution == "simple":
            synthesizer = SingleHopSpecificSynthesizer()
        elif config.distribution == "complex":
            synthesizer = MultiHopSpecificSynthesizer()
        else:
            # 默认使用简单合成器
            synthesizer = SingleHopSpecificSynthesizer()

        # 生成测试集
        try:
            testset = generator.generate_with_langchain_docs(
                documents=ragas_docs,
                test_size=config.test_size,
                synthesizers=[synthesizer]
            )

            # 转换为DataFrame
            df = self._convert_to_dataframe(testset)

            self.logger.info(
                f"测试数据集生成完成: {len(df)} 条记录"
            )
            return df

        except Exception as e:
            self.logger.error(f"Ragas测试集生成失败: {str(e)}")
            raise RuntimeError(f"测试集生成失败: {str(e)}") from e

    def _convert_documents(self, documents: List[Any]) -> List[Document]:
        """转换文档为Ragas兼容格式

        如果输入已经是LangChain Document，直接返回
        否则尝试转换
        """
        converted = []
        for doc in documents:
            if isinstance(doc, Document):
                converted.append(doc)
            elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # 可能是其他框架的文档对象，转换为LangChain Document
                converted.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=getattr(doc, 'metadata', {})
                    )
                )
            elif isinstance(doc, dict):
                # 字典格式
                converted.append(
                    Document(
                        page_content=doc.get('content', doc.get('text', '')),
                        metadata=doc.get('metadata', {})
                    )
                )
            elif isinstance(doc, str):
                # 纯文本
                converted.append(Document(page_content=doc, metadata={}))
            else:
                self.logger.warning(f"无法转换文档类型: {type(doc)}")
        
        return converted

    def _convert_to_dataframe(self, testset) -> pd.DataFrame:
        """将Ragas测试集转换为标准DataFrame格式"""
        records = []
        
        for sample in testset.samples:
            record = {
                'question': getattr(sample, 'question', ''),
                'contexts': getattr(sample, 'contexts', []),
                'ground_truth': getattr(sample, 'ground_truth', ''),
            }
            # 可选字段
            if hasattr(sample, 'evolution_type'):
                record['evolution_type'] = sample.evolution_type
            
            records.append(record)

        df = pd.DataFrame(records)
        
        # 确保标准列存在
        required_columns = ['question', 'contexts', 'ground_truth']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        return df

    def validate_generated_dataset(
        self,
        df: pd.DataFrame
    ) -> tuple[bool, List[str]]:
        """验证生成的数据集格式"""
        errors = []

        if df.empty:
            errors.append("生成的数据集为空")
            return False, errors

        # 检查必需列
        required_columns = ['question', 'contexts', 'ground_truth']
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"缺少必需列: {col}")

        # 检查数据质量
        if 'question' in df.columns:
            empty_questions = df['question'].isna().sum()
            if empty_questions > 0:
                errors.append(f"有 {empty_questions} 条空问题")

        if 'contexts' in df.columns:
            empty_contexts = df['contexts'].apply(
                lambda x: not x or (isinstance(x, list) and len(x) == 0)
            ).sum()
            if empty_contexts > 0:
                errors.append(f"有 {empty_contexts} 条空上下文")

        is_valid = len(errors) == 0
        if is_valid:
            self.logger.info(f"数据集验证通过: {len(df)} 条记录")
        else:
            self.logger.warning(f"数据集验证失败: {errors}")

        return is_valid, errors
