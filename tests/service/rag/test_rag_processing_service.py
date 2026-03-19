"""
单元测试：RAG处理服务实现
使用 mock 模拟 DocumentRepository，不依赖实际数据库
"""
import unittest
from unittest.mock import Mock, MagicMock
from typing import List, Optional, Union
from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from application.services.rag.rag_processing_service_impl import (
    RAGProcessingServiceImpl,
    RAGProcessingServiceFactoryImpl
)


class TestRAGProcessingServiceImpl(unittest.TestCase):
    """RAG处理服务单元测试"""

    def setUp(self):
        """测试前置：创建模拟 repository"""
        self.mock_repository = Mock(spec=DocumentRepository)

    def test_initialization_success(self):
        """测试服务成功初始化"""
        service = RAGProcessingServiceImpl(
            domain="test",
            document_repository=self.mock_repository
        )
        self.assertIsNotNone(service)
        self.assertEqual(service._domain, "test")
        self.assertEqual(service._document_repository, self.mock_repository)

    def test_initialization_missing_repository(self):
        """测试未提供repository时抛出错误"""
        with self.assertRaises(ValueError):
            RAGProcessingServiceImpl(
                domain="test",
                document_repository=None
            )

    def test_add_documents_success(self):
        """测试添加文档返回ID列表"""
        # 准备测试数据
        doc1 = Document(content="测试文档1", metadata={"source": "test"})
        doc1.id = 1
        doc2 = Document(content="测试文档2", metadata={"source": "test"})
        doc2.id = 2

        # 设置 mock 返回值
        self.mock_repository.save_all.return_value = [doc1, doc2]

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.add_documents([doc1, doc2])

        # 验证
        self.assertEqual(result, ["1", "2"])
        self.mock_repository.save_all.assert_called_once()

    def test_retrieve_similar(self):
        """测试相似文档检索"""
        # 准备测试数据
        doc1 = Document(content="结果文档1", metadata={"score": 0.9})
        doc1.id = 1
        doc2 = Document(content="结果文档2", metadata={"score": 0.8})
        doc2.id = 2

        # 设置 mock 返回值
        self.mock_repository.search_by_text.return_value = [doc1, doc2]

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.retrieve_similar("查询词", limit=2, score_threshold=0.7)

        # 验证
        self.assertEqual(len(result), 2)
        self.mock_repository.search_by_text.assert_called_once_with(
            query="查询词",
            limit=2,
            score_threshold=0.7
        )

    def test_delete_documents_success(self):
        """测试删除成功返回True"""
        # 设置 mock
        self.mock_repository.delete_all = MagicMock()

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.delete_documents(["1", "2", "3"])

        # 验证
        self.assertTrue(result)
        self.mock_repository.delete_all.assert_called_once()
        # 验证参数类型已转换为 int
        call_args = self.mock_repository.delete_all.call_args[0][0]
        self.assertEqual(call_args, [1, 2, 3])
        self.assertIsInstance(call_args[0], int)

    def test_delete_documents_failure(self):
        """测试删除异常返回False"""
        # 设置 mock 抛出异常
        self.mock_repository.delete_all.side_effect = Exception("数据库错误")

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.delete_documents(["1", "2"])

        # 验证
        self.assertFalse(result)

    def test_delete_documents_converts_types(self):
        """验证字符串ID正确转换为整数ID，保留非数字ID为字符串"""
        # 设置 mock
        self.mock_repository.delete_all = MagicMock()

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        # 混合数字ID和非数字ID
        result = service.delete_documents(["123", "abc", "456", "doc-789"])

        # 验证
        self.assertTrue(result)
        call_args = self.mock_repository.delete_all.call_args[0][0]
        self.assertEqual(len(call_args), 4)
        self.assertEqual(call_args[0], 123)
        self.assertEqual(call_args[1], "abc")
        self.assertEqual(call_args[2], 456)
        self.assertEqual(call_args[3], "doc-789")
        self.assertIsInstance(call_args[0], int)
        self.assertIsInstance(call_args[1], str)

    def test_get_document_success(self):
        """测试成功获取文档"""
        # 准备测试数据
        doc = Document(content="测试文档", metadata={"id": 1})
        doc.id = 1

        # 设置 mock
        self.mock_repository.find_by_id.return_value = doc

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.get_document("1")

        # 验证
        self.assertIsNotNone(result)
        self.assertEqual(result.id, 1)
        self.mock_repository.find_by_id.assert_called_once_with(1)

    def test_get_document_not_found(self):
        """测试文档不存在返回None"""
        # 设置 mock
        self.mock_repository.find_by_id.return_value = None

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.get_document("999")

        # 验证
        self.assertIsNone(result)

    def test_get_document_invalid_id_format(self):
        """测试非整数ID格式返回None"""
        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.get_document("not-a-number")

        # 验证
        self.assertIsNone(result)
        # 验证 find_by_id 未被调用，因为转换失败提前返回
        self.mock_repository.find_by_id.assert_not_called()

    def test_get_document_other_exception(self):
        """测试其他异常返回None"""
        # 设置 mock 抛出异常
        self.mock_repository.find_by_id.side_effect = Exception("数据库连接失败")

        # 创建服务并执行
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=self.mock_repository
        )
        result = service.get_document("1")

        # 验证
        self.assertIsNone(result)

    def test_process_document(self):
        """测试单个文档处理流程"""
        # 准备测试数据
        doc = Document(content="这是一个测试文档内容", metadata={"source": "test"})

        # 模拟分块结果
        split_docs = [doc]

        # 设置 mock
        self.mock_repository.save_all.return_value = [doc]
        doc.id = 1

        # 创建服务并执行（这里需要给 TextSplitterFactory 打 patch，但简单起见我们不测试分块）
        from application.services.rag.rag_processing_service_impl import TextSplitterFactory
        original_split = TextSplitterFactory.split_documents
        TextSplitterFactory.split_documents = Mock(return_value=split_docs)

        try:
            service = RAGProcessingServiceImpl(
                domain="default",
                document_repository=self.mock_repository
            )
            result = service.process_document(doc)

            # 验证
            self.assertIsNotNone(result)
            TextSplitterFactory.split_documents.assert_called_once()
        finally:
            TextSplitterFactory.split_documents = original_split

    def test_batch_process_documents(self):
        """测试批量文档处理"""
        # 准备测试数据
        docs = [
            Document(content="文档1", metadata={"source": "test"}),
            Document(content="文档2", metadata={"source": "test"}),
        ]

        # 设置 mock
        for i, doc in enumerate(docs):
            doc.id = i + 1
        self.mock_repository.save_all.return_value = docs

        # patch TextSplitterFactory
        from application.services.rag.rag_processing_service_impl import TextSplitterFactory
        original_split = TextSplitterFactory.split_documents
        TextSplitterFactory.split_documents = Mock(return_value=docs)

        try:
            service = RAGProcessingServiceImpl(
                domain="default",
                document_repository=self.mock_repository
            )
            result = service.batch_process_documents(docs)

            # 验证
            self.assertEqual(len(result), 2)
            TextSplitterFactory.split_documents.assert_called_once()
        finally:
            TextSplitterFactory.split_documents = original_split

    def test_auto_sets_collection_name_when_has_setter(self):
        """测试当 document_repository 有 collection_name setter 时，RAG 自动设置"""
        # 创建一个 mock repository 带有 collection_name property setter
        class MockRepoWithSetter:
            def __init__(self):
                self._collection_name = None
                # mock 所有必要方法
                self.save_all = Mock()
                self.search_by_text = Mock()
                self.delete_all = Mock()
                self.find_by_id = Mock()

            @property
            def collection_name(self):
                return self._collection_name

            @collection_name.setter
            def collection_name(self, value):
                self._collection_name = value

        mock_repo = MockRepoWithSetter()
        self.assertIsNone(mock_repo.collection_name)

        # 创建 RAG 服务
        service = RAGProcessingServiceImpl(
            domain="my_test_domain",
            document_repository=mock_repo
        )

        # 验证 RAG 自动设置了 collection_name
        # 根据规则，domain="my_test_domain" 对应 collection_name="doc_my_test_domain"
        self.assertEqual(mock_repo.collection_name, "doc_my_test_domain")

    def test_skips_setting_when_no_setter(self):
        """测试当 document_repository 没有 collection_name setter 时，自动跳过不报错"""
        # 普通 mock 没有 setter，验证不会报错
        mock_repo = Mock(spec=DocumentRepository)

        # 初始化应该成功，不会抛出异常
        service = RAGProcessingServiceImpl(
            domain="default",
            document_repository=mock_repo
        )

        self.assertIsNotNone(service)
        # mock 上不应该有 collection_name 属性被设置
        self.assertFalse(hasattr(mock_repo, 'collection_name'))


class TestRAGProcessingServiceFactoryImpl(unittest.TestCase):
    """RAG处理服务工厂单元测试"""

    def setUp(self):
        """测试前置"""
        self.mock_repository = Mock(spec=DocumentRepository)
        self.factory = RAGProcessingServiceFactoryImpl()

    def test_create_service_new_domain(self):
        """测试创建新领域服务"""
        service = self.factory.create_service(
            domain="test_factory",
            document_repository=self.mock_repository
        )
        self.assertIsNotNone(service)
        self.assertIn("test_factory", self.factory.list_domains())

    def test_create_service_missing_repository(self):
        """测试创建新服务时未提供repository抛出错误"""
        with self.assertRaises(ValueError):
            self.factory.create_service(
                domain="new_domain",
                document_repository=None
            )

    def test_create_service_get_existing(self):
        """测试获取已创建的服务实例（单例）"""
        service1 = self.factory.create_service(
            domain="test_singleton",
            document_repository=self.mock_repository
        )
        service2 = self.factory.create_service(
            domain="test_singleton",
            document_repository=self.mock_repository
        )
        # 应该返回同一个实例
        self.assertIs(service1, service2)

    def test_remove_service(self):
        """测试移除服务实例"""
        self.factory.create_service(
            domain="test_remove",
            document_repository=self.mock_repository
        )
        self.assertIn("test_remove", self.factory.list_domains())

        self.factory.remove_service("test_remove")
        self.assertNotIn("test_remove", self.factory.list_domains())


if __name__ == "__main__":
    unittest.main()
