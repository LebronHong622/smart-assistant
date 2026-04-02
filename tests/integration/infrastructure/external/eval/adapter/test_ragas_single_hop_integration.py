"""
RagasSingleHopAdapter 集成测试
使用真实文档和真实API进行端到端测试
"""
import os
import json
import pytest
from typing import List
from infrastructure.external.eval.adapters.ragas_single_hop_adapter import RagasSingleHopAdapter
from domain.shared.ports.test_dataset_generator_port import TestDatasetGenerationConfig
from domain.entity.eval.generated_test_sample import GeneratedTestDataset
from domain.entity.document.document import Document


# 检查环境变量是否存在，决定是否跳过测试
# 由于项目已经在 config/settings.py 中加载了API密钥，这里只检查是否为空
requires_api_key = pytest.mark.skipif(
    not (os.getenv("DEEPSEEK_API_KEY") and os.getenv("DASHSCOPE_API_KEY")),
    reason="缺少DEEPSEEK_API_KEY或DASHSCOPE_API_KEY环境变量，跳过集成测试"
)


class TestRagasSingleHopAdapterIntegration:
    """RagasSingleHopAdapter 的端到端集成测试类

    使用真实售后服务政策文档和真实LLM/Embedding API执行完整生成流程
    """

    def setup_method(self):
        """测试前准备 - 加载测试文档路径和配置路径"""
        self.config_path = "config/eval/test_dataset_config.yaml"
        self.test_docs_path = "data/documents/test/after_sales_policy_data.json"

    def load_test_documents(self) -> List[Document]:
        """从JSON文件加载测试文档并转换为领域实体

        Returns:
            List[Document]: 转换后的文档列表
        """
        with open(self.test_docs_path, "r", encoding="utf-8") as f:
            policies = json.load(f)

        documents = []
        for policy in policies:
            doc = Document(
                content=policy["content"],
                metadata=policy  # 将整个policy对象保存为metadata
            )
            documents.append(doc)

        return documents

    @requires_api_key
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_from_documents_full_flow(self):
        """完整端到端流程测试

        测试从真实文档加载 → 转换 → 初始化 → 知识图谱准备 → 场景生成 → 样本生成 → 结果转换的全流程
        只要不抛出异常即为测试通过
        """
        # 加载测试文档
        documents = self.load_test_documents()

        # 创建适配器（真实初始化所有组件）
        adapter = RagasSingleHopAdapter(config_path=self.config_path)

        # 配置生成参数 - 使用较小的test_size控制API成本
        config = TestDatasetGenerationConfig(test_size=5)

        # 执行完整生成流程
        dataset = await adapter.generate_from_documents(documents, config)

        # 基础验证
        assert dataset is not None
        assert isinstance(dataset, GeneratedTestDataset)

    @requires_api_key
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_generate_from_documents_result_validation(self):
        """验证生成结果格式和内容正确性

        验证:
        - 返回类型正确 (GeneratedTestDataset)
        - 样本数量符合预期范围
        - 每个样本字段完整非空
        - 转换正确
        """
        # 加载测试文档
        documents = self.load_test_documents()

        # 创建适配器
        adapter = RagasSingleHopAdapter(config_path=self.config_path)

        # 配置生成参数
        test_size = 5
        config = TestDatasetGenerationConfig(test_size=test_size)

        # 执行生成
        dataset = await adapter.generate_from_documents(documents, config)

        # 验证类型
        assert isinstance(dataset, GeneratedTestDataset)

        # 验证样本数量合理
        assert dataset.count > 0, "至少应该生成一个样本"
        assert dataset.count <= test_size, f"样本数量不应超过test_size={test_size}"
        assert len(dataset.samples) == dataset.count

        # 验证每个样本字段完整
        for sample in dataset.samples:
            assert sample.question is not None
            assert len(sample.question.strip()) > 0, "问题不应为空"
            assert sample.ground_truth is not None
            assert len(sample.ground_truth.strip()) > 0, "标准答案不应为空"
            assert sample.contexts is not None
            assert len(sample.contexts) > 0, "至少应有一个上下文"
            for context in sample.contexts:
                assert len(context.strip()) > 0, "上下文不应为空"
