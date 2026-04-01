"""
单跳测试集生成器 - 两阶段架构
参考ragas官方文档实现：https://docs.ragas.org.cn/en/stable/howtos/customizations/testgenerator/_testgen-custom-single-hop/

阶段1: TestDatasetPreparer - 使用组合模式准备 KnowledgeGraph 和 Persona
阶段2: ConfigurableSingleHopSynthesizer - 使用继承模式生成场景和样本
"""
import importlib
from typing import List, Dict, Any, Type, Optional, Callable
import numpy as np
from domain.entity.document.document import Document
from pydantic import BaseModel, Field, ConfigDict
from ragas.testset.transforms import apply_transforms, BaseGraphTransformation
from ragas.testset.synthesizers.single_hop.base import (
    SingleHopQuerySynthesizer,
    SingleHopScenario,
)
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.persona import Persona
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.prompt import PydanticPrompt
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
                    "document_metadata": doc.metadata,
                }
            )
            kg._add_node(node)

        # 应用所有启用的transforms
        if self.transforms:
            apply_transforms(kg, self.transforms)

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
    继承自 SingleHopQuerySynthesizer，采用Ragas官方推荐的LLM智能匹配方式
    """

    def __init__(
        self,
        llm: BaseRagasLLM,
        syntheziser_config: Optional[SynthesizerConfig] = None,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        self.syntheziser_config = syntheziser_config or SynthesizerConfig()
        self.theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: List[Persona],
        callbacks=None,
    ) -> List[SingleHopScenario]:
        """
        重写场景生成逻辑，采用Ragas官方推荐的LLM智能匹配方式

        Args:
            n: 需要生成的场景数量
            knowledge_graph: 知识图谱
            persona_list: 角色列表
            callbacks: 回调函数

        Returns:
            SingleHopScenario对象列表
        """
        scenarios: List[SingleHopScenario] = []

        # 获取所有节点
        nodes = knowledge_graph.nodes

        # 应用节点过滤器（如果配置）
        if self.syntheziser_config.node_filter:
            nodes = [node for node in nodes if self.syntheziser_config.node_filter(node)]

        if not nodes:
            return scenarios

        # 计算每个节点需要生成的场景数
        samples_per_node = max(1, int(np.ceil(n / len(nodes))))

        for node in nodes:
            if len(scenarios) >= n:
                break

            # 提取主题
            themes = self._extract_themes(node)
            if not themes:
                continue

            # 使用LLM智能匹配主题和角色
            prompt_input = ThemesPersonasInput(themes=themes, personas=persona_list)
            persona_concepts = await self.theme_persona_matching_prompt.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )

            # 准备所有可能的组合
            base_scenarios = self.prepare_combinations(
                node,
                themes,
                personas=persona_list,
                persona_concepts=persona_concepts.mapping,
            )

            # 采样指定数量的场景
            sampled = self.sample_combinations(base_scenarios, samples_per_node)
            scenarios.extend(sampled)

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

    def set_instruction(self, instruction: str) -> "ConfigurableSingleHopSynthesizer":
        """
        设置自定义指令，用于自定义查询生成提示

        Args:
            instruction: 自定义指令字符串

        Returns:
            返回self，支持链式调用

        示例:
            yes_no_instruction = '''Generate a Yes/No query and answer based on the specified conditions.
            The query should be answerable with only "Yes" or "No".'''
            synthesizer = ConfigurableSingleHopSynthesizer(llm=llm).set_instruction(yes_no_instruction)
        """
        prompts = self.get_prompts()
        if "generate_query_reference_prompt" in prompts:
            prompt = prompts["generate_query_reference_prompt"]
            prompt.instruction = instruction
            self.set_prompts(**{"generate_query_reference_prompt": prompt})
        return self


async def generate_scenarios(
    preparer: TestDatasetPreparer,
    synthesizer: ConfigurableSingleHopSynthesizer,
    documents: List[Document],
    num_scenarios: Optional[int] = None,
) -> List[SingleHopScenario]:
    """
    便捷函数：生成测试场景列表

    Args:
        preparer: 数据准备器（阶段1）
        synthesizer: 查询合成器（阶段2）
        documents: 输入文档
        num_scenarios: 生成场景数量

    Returns:
        SingleHopScenario对象列表
    """
    # 阶段1: 准备数据
    prepared_data = preparer.prepare(documents)

    # 阶段2: 生成场景
    return await synthesizer.generate_scenarios(
        n=num_scenarios or 10,
        knowledge_graph=prepared_data.knowledge_graph,
        persona_list=prepared_data.persona_list,
    )
