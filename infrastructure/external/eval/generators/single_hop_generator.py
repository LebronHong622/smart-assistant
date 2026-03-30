"""
单跳测试集生成器
参考ragas官方文档实现：https://docs.ragas.org.cn/en/stable/howtos/customizations/testgenerator/_testgen-custom-single-hop/
使用ragas内置KeyphrasesExtractor作为transform，不单独封装
"""
from typing import List, Dict
import pandas as pd
from langchain_core.documents import Document
from ragas.testset.transforms import KeyphrasesExtractor, apply_transforms
from ragas.testset.synthesizers import (
    SingleHopSpecificSynthesizer,
    SingleHopAbstractSynthesizer,
)
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from domain.eval.test_dataset_config import GenerationConfig, KeywordExtractionConfig


class SingleHopGenerator:
    """
    单跳测试集生成器
    - 每个文档片段单独生成问题，保持单跳特性
    - 支持两种角色：具体问题提问者 / 抽象问题提问者
    - 使用ragas内置KeyphrasesExtractor提取关键词
    """

    def __init__(
        self,
        llm: BaseRagasLLM,
        embedding: BaseRagasEmbeddings,
        config: GenerationConfig,
        keyword_config: KeywordExtractionConfig,
    ):
        self.llm = llm
        self.embedding = embedding
        self.config = config
        self.keyword_config = keyword_config

        # 初始化合成器 - 两种角色对应ragas两种内置合成器
        # 具体问题提问者 -> SingleHopSpecificSynthesizer
        if config.roles.get("concrete", RoleConfig()).enabled:
            self.concrete_synthesizer = SingleHopSpecificSynthesizer(llm=self.llm)

        # 抽象问题提问者 -> SingleHopAbstractSynthesizer
        if config.roles.get("abstract", RoleConfig()).enabled:
            self.abstract_synthesizer = SingleHopAbstractSynthesizer(llm=self.llm)

    def generate_from_documents(
        self,
        documents: List[Document],
    ) -> pd.DataFrame:
        """
        从文档列表生成单跳测试集
        每个文档片段单独生成问题，保持单跳特性

        Args:
            documents: 分割后的LangChain文档列表

        Returns:
            DataFrame 包含列：question, contexts, ground_truth, evolution_type, doc_id, source
        """
        # 1. 构建知识图谱并提取关键词
        kg = self._build_knowledge_graph(documents)

        # 2. 逐个节点（文档片段）生成问题
        rows = []
        for node in kg.nodes:
            # 对每个节点，根据配置生成对应角色的问题
            node_rows = self._generate_for_node(node)
            rows.extend(node_rows)

        # 3. 转换为DataFrame返回
        return pd.DataFrame(rows)

    def _build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        """构建知识图谱
        使用ragas内置KeyphrasesExtractor作为transform，直接通过apply_transforms处理
        """
        # 创建知识图谱
        kg = KnowledgeGraph()
        for doc in documents:
            # 创建节点
            node = Node(
                properties={
                    "text": doc.page_content,
                    **doc.metadata,
                }
            )
            kg.add_node(node)

        # 使用ragas内置transforms提取关键词短语
        # KeyphrasesExtractor直接作为transform使用
        if self.keyword_config.enabled:
            keyphrase_extractor = KeyphrasesExtractor(
                llm=self.llm,
                top_n=self.keyword_config.top_n,
            )
            apply_transforms(kg, [keyphrase_extractor])

        return kg

    def _generate_for_node(self, node: Node) -> List[Dict]:
        """为单个节点生成问题"""
        rows = []
        doc_text = node.properties["text"]

        # 具体问题生成
        if (
            "concrete" in self.config.roles
            and self.config.roles["concrete"].enabled
            and hasattr(self, "concrete_synthesizer")
        ):
            role_config = self.config.roles["concrete"]
            q_per_doc = role_config.questions_per_doc
            for _ in range(q_per_doc):
                result = self.concrete_synthesizer.generate(
                    node=node,
                    prompts=self._get_role_prompt(role_config),
                )
                row = {
                    "question": result.question,
                    "contexts": [doc_text],
                    "ground_truth": result.ground_truth,
                    "evolution_type": "concrete",
                    "doc_id": node.properties.get("doc_id"),
                    "source": node.properties.get("source"),
                }
                rows.append(row)

        # 抽象问题生成
        if (
            "abstract" in self.config.roles
            and self.config.roles["abstract"].enabled
            and hasattr(self, "abstract_synthesizer")
        ):
            role_config = self.config.roles["abstract"]
            q_per_doc = role_config.questions_per_doc
            for _ in range(q_per_doc):
                result = self.abstract_synthesizer.generate(
                    node=node,
                    prompts=self._get_role_prompt(role_config),
                )
                row = {
                    "question": result.question,
                    "contexts": [doc_text],
                    "ground_truth": result.ground_truth,
                    "evolution_type": "abstract",
                    "doc_id": node.properties.get("doc_id"),
                    "source": node.properties.get("source"),
                }
                rows.append(row)

        return rows

    def _get_role_prompt(self, role_config: Dict) -> Dict:
        """获取角色提示词"""
        return {
            "role_description": role_config.description
        }
