"""
单跳测试集生成器 - 两阶段架构
参考ragas官方文档实现：https://docs.ragas.org.cn/en/stable/howtos/customizations/testgenerator/_testgen-custom-single-hop/

阶段1: TestDatasetPreparer - 使用组合模式准备 KnowledgeGraph 和 Persona
阶段2: ConfigurableSingleHopSynthesizer - 使用继承模式生成场景和样本
"""
import importlib
from typing import List, Dict, Any, Type, Optional, Callable
import pandas as pd
from domain.entity.document.document import Document
from pydantic import BaseModel, Field, ConfigDict
from ragas.testset.transforms import apply_transforms, BaseGraphTransformation
from ragas.testset.synthesizers.single_hop.base import SingleHopQuerySynthesizer
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from config.eval_settings import GenerationConfig, TransformConfig, RoleConfig


class PreparedData(BaseModel):
    """阶段1准备的数据"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    knowledge_graph: KnowledgeGraph
    persona_list: List[Persona]
    config: GenerationConfig
    documents: List[Document]


class TestDatasetPreparer:
    """
    阶段1: 测试数据集准备器
    使用组合模式准备生成所需的数据结构
    """

    def __init__(
        self,
        llm: BaseRagasLLM,
        embedding: BaseRagasEmbeddings,
        config: GenerationConfig,
    ):
        self.llm = llm
        self.embedding = embedding
        self.config = config
        self.transforms: List[BaseGraphTransformation] = []

    def prepare(self, documents: List[Document]) -> PreparedData:
        """准备生成所需的所有数据

        Args:
            documents: LangChain文档列表

        Returns:
            PreparedData 包含知识图谱、角色列表等
        """
        # 1. 构建知识图谱并应用transforms
        knowledge_graph = self._build_knowledge_graph(documents)

        # 2. 构建角色列表
        persona_list = self._build_persona_list()

        return PreparedData(
            knowledge_graph=knowledge_graph,
            persona_list=persona_list,
            config=self.config,
            documents=documents,
        )

    def _build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        """构建知识图谱并应用所有transforms"""
        # 加载transforms
        self.transforms = self._load_transforms()

        # 创建知识图谱
        kg = KnowledgeGraph()
        for doc in documents:
            node = Node(
                properties={
                    "page_content": doc.content,
                    **doc.metadata,
                }
            )
            kg._add_node(node)

        # 应用所有启用的transforms
        if self.transforms:
            apply_transforms(kg, self.transforms, llm=self.llm)

        return kg

    def _load_transforms(self) -> List[BaseGraphTransformation]:
        """从配置加载所有启用的transforms"""
        transforms: List[BaseGraphTransformation] = []

        for name, transform_config in self.config.transforms.items():
            if not transform_config.enable:
                continue

            # 动态加载类
            cls = self._load_class(transform_config.module, transform_config.class_name)

            # 处理参数
            params = transform_config.parameters.copy()

            # 如果需要LLM，注入LLM实例
            if params.pop("is_llm", False):
                params["llm"] = self.llm

            # 创建实例并添加到列表
            transform = cls(**params)
            transforms.append(transform)

        return transforms

    def _build_persona_list(self) -> List[Persona]:
        """从配置构建角色列表"""
        personas = []

        for role_name, role_config in self.config.roles.items():
            if not role_config.enabled:
                continue

            # 创建Persona实例
            persona = Persona(
                name=role_name,
                role_description=role_config.description,
            )
            personas.append(persona)

        return personas

    def _load_class(self, module_name: str, class_name: str) -> Type:
        """动态从模块路径加载类"""
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        except ImportError:
            raise ImportError(f"无法导入模块: {module_name}")
        except AttributeError:
            raise AttributeError(f"在模块 {module_name} 中找不到类: {class_name}")


class SynthesizerConfig(BaseModel):
    """可配置单跳合成器配置"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_property_name: str = "keyphrases"
    enable_theme_matching: bool = True
    node_filter: Optional[Callable[[Node], bool]] = Field(default=None)
    theme_extractor: Optional[Callable[[Node], List[str]]] = Field(default=None)


class ConfigurableSingleHopSynthesizer(SingleHopQuerySynthesizer):
    """
    阶段2: 可配置单跳查询合成器
    继承自 SingleHopQuerySynthesizer，重写 _generate_scenarios
    """

    def __init__(
        self,
        llm: BaseRagasLLM,
        syntheziser_config: Optional[SynthesizerConfig] = None,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        self.syntheziser_config = syntheziser_config or SynthesizerConfig()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: List[Persona],
        callbacks=None,
    ) -> List[Dict[str, Any]]:
        """
        重写场景生成逻辑

        Args:
            n: 需要生成的场景数量
            knowledge_graph: 知识图谱
            persona_list: 角色列表
            callbacks: 回调函数

        Returns:
            场景列表，每个场景包含生成查询所需的信息
        """
        scenarios = []

        # 获取所有节点
        nodes = knowledge_graph.nodes

        # 应用节点过滤器（如果配置）
        if self.syntheziser_config.node_filter:
            nodes = [node for node in nodes if self.syntheziser_config.node_filter(node)]

        if not nodes:
            return scenarios

        # 计算每个节点需要生成的场景数
        scenarios_per_node = max(1, n // len(nodes))

        for node in nodes:
            # 提取主题（用于主题-角色匹配）
            themes = self._extract_themes(node)

            # 为每个角色生成场景
            for persona in persona_list:
                # 主题匹配检查
                if self.syntheziser_config.enable_theme_matching:
                    if not self._check_theme_persona_match(themes, persona):
                        continue

                # 获取节点的指定属性值
                property_values = node.properties.get(
                    self.syntheziser_config.node_property_name, []
                )
                if not property_values:
                    property_values = [node.properties.get("page_content", "")]

                # 创建场景
                for _ in range(scenarios_per_node):
                    scenario = {
                        "node": node,
                        "persona": persona,
                        "themes": themes,
                        "property_name": self.syntheziser_config.node_property_name,
                        "property_values": property_values,
                    }
                    scenarios.append(scenario)

                if len(scenarios) >= n:
                    break

            if len(scenarios) >= n:
                break

        return scenarios[:n]

    def _extract_themes(self, node: Node) -> List[str]:
        """从节点提取主题"""
        # 如果配置了自定义主题提取器，使用它
        if self.syntheziser_config.theme_extractor:
            return self.syntheziser_config.theme_extractor(node)

        # 默认从keyphrases提取主题
        themes = node.properties.get("keyphrases", [])
        if not themes:
            # 如果没有keyphrases，使用文档内容的前几个字作为主题
            content = node.properties.get("page_content", "")[:50]
            themes = [content] if content else ["general"]
        return themes

    def _check_theme_persona_match(self, themes: List[str], persona: Persona) -> bool:
        """检查主题和角色是否匹配"""
        # 简化版本：总是返回True
        # 实际实现可以基于嵌入相似度或LLM判断
        return True

    def generate_samples(
        self,
        prepared_data: PreparedData,
        num_questions: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        从准备好的数据生成测试样本

        Args:
            prepared_data: 阶段1准备的数据
            num_questions: 生成问题数量（默认从配置读取）

        Returns:
            DataFrame 包含 question, contexts, ground_truth, evolution_type 等列
        """
        if num_questions is None:
            num_questions = prepared_data.config.single_hop.max_questions_per_doc

        # 这里我们手动实现生成逻辑，而不是调用父类的generate
        # 因为父类的generate需要TestsetGenerator上下文
        rows = []

        for node in prepared_data.knowledge_graph.nodes:
            node_text = node.properties.get("page_content", "")

            for persona in prepared_data.persona_list:
                # 获取该角色配置
                role_config = prepared_data.config.roles.get(persona.name)
                # 检查角色是否存在且启用
                if not role_config or not role_config.enabled:
                    continue

                q_per_doc = role_config.questions_per_doc

                for _ in range(q_per_doc):
                    # 创建查询生成提示
                    prompt = self._create_generation_prompt(node, persona)

                    # 这里简化处理，实际应该调用LLM生成
                    # 返回模拟数据用于测试
                    row = {
                        "question": f"[{persona.name}] 基于以下内容的问题",
                        "contexts": [node_text],
                        "ground_truth": "答案",
                        "evolution_type": persona.name,
                        "doc_id": node.properties.get("doc_id"),
                        "source": node.properties.get("source"),
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def _create_generation_prompt(self, node: Node, persona: Persona) -> Dict[str, str]:
        """创建问题生成提示"""
        content = node.properties.get("page_content", "")
        keyphrases = node.properties.get("keyphrases", [])

        return {
            "context": content,
            "keyphrases": ", ".join(keyphrases) if keyphrases else "",
            "persona_description": persona.role_description,
        }


def generate_test_dataset(
    preparer: TestDatasetPreparer,
    synthesizer: ConfigurableSingleHopSynthesizer,
    documents: List[Document],
    num_questions: Optional[int] = None,
) -> pd.DataFrame:
    """
    便捷函数：两阶段生成测试数据集

    Args:
        preparer: 数据准备器（阶段1）
        synthesizer: 查询合成器（阶段2）
        documents: 输入文档
        num_questions: 问题数量

    Returns:
        测试数据集DataFrame
    """
    # 阶段1: 准备数据
    prepared_data = preparer.prepare(documents)

    # 阶段2: 生成样本
    return synthesizer.generate_samples(prepared_data, num_questions)
