"""
Unit tests for LangChainDocumentRepository
使用 mock 进行单元测试，不需要外部 Milvus 服务
"""
import pytest
from unittest.mock import MagicMock, Mock, patch
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from domain.document.entity.document import Document
from infrastructure.persistence.vector.repository.langchain_document_repository_impl import LangChainDocumentRepository


class TestLangChainDocumentRepository:
    """LangChainDocumentRepository 单元测试"""

    def test_initialization_with_embedding_function(self):
        """测试使用 embedding_function 初始化"""
        mock_embeddings = Mock(spec=Embeddings)

        with patch('infrastructure.rag.embeddings.VectorStoreFactory') as mock_factory:
            mock_vector_store = Mock(spec=VectorStore)
            mock_factory.create_store.return_value = mock_vector_store

            repo = LangChainDocumentRepository(
                collection_name="test_collection",
                embedding_function=mock_embeddings
            )

            mock_factory.create_store.assert_called_once_with(
                embedding=mock_embeddings,
                collection_name="test_collection"
            )
            assert repo._collection_name == "test_collection"
            assert repo._embedding_function == mock_embeddings
            assert repo._vector_store == mock_vector_store

    def test_initialization_with_vector_store(self):
        """测试注入 vector_store 初始化"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_embeddings = Mock(spec=Embeddings)

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            embedding_function=mock_embeddings,
            vector_store=mock_vector_store
        )

        assert repo._collection_name == "test_collection"
        assert repo._vector_store == mock_vector_store

    def test_initialization_raises_error_when_no_parameters(self):
        """测试未提供参数时抛出错误"""
        with pytest.raises(ValueError, match="必须提供 vector_store 或 embedding_function 参数"):
            LangChainDocumentRepository(collection_name="test_collection")

    def test_save_document_assigns_id_correctly(self):
        """测试保存文档正确分配ID"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.add_documents.return_value = ["123"]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        document = Document(content="测试内容", metadata={"key": "value"})
        assert document.id is None

        result = repo.save(document)

        mock_vector_store.add_documents.assert_called_once()
        args = mock_vector_store.add_documents.call_args
        lc_docs = args[0][0]
        assert len(lc_docs) == 1
        assert isinstance(lc_docs[0], LCDocument)
        assert lc_docs[0].page_content == "测试内容"
        assert lc_docs[0].metadata == {"key": "value"}
        assert result.id == 123

    def test_save_handles_non_integer_id_gracefully(self):
        """测试优雅处理无法转换为整数的ID"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.add_documents.return_value = ["uuid-abc-123"]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        document = Document(content="测试内容")
        result = repo.save(document)

        # ID 保持字符串形式（实际上无法转换为整数，所以不会被赋值，仍然是 None？）
        # 当前实现中，如果转换失败，不会赋值，所以仍然是 None
        assert result.id is None

    def test_save_all_bulk_operation(self):
        """测试批量保存操作"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.add_documents.return_value = ["1", "2", "3"]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        documents = [
            Document(content="内容1"),
            Document(content="内容2"),
            Document(content="内容3"),
        ]

        results = repo.save_all(documents)

        mock_vector_store.add_documents.assert_called_once()
        assert len(results) == 3
        assert results[0].id == 1
        assert results[1].id == 2
        assert results[2].id == 3

    def test_find_by_id_with_mock_collection(self):
        """测试通过ID查找（使用模拟的Milvus collection）"""
        mock_collection = Mock()
        mock_collection.query.return_value = [
            {
                "id": 42,
                "content": "测试内容",
                "metadata": '{"key": "value"}'
            }
        ]

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        result = repo.find_by_id(42)

        mock_collection.query.assert_called_once()
        assert result is not None
        assert result.id == 42
        assert result.content == "测试内容"
        assert result.metadata == {"key": "value"}

    def test_find_by_id_returns_none_when_not_found(self):
        """测试文档不存在时返回None"""
        mock_collection = Mock()
        mock_collection.query.return_value = []

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        result = repo.find_by_id(999)

        assert result is None

    def test_find_all_returns_correct_list(self):
        """测试find_all分页返回正确的文档列表"""
        mock_collection = Mock()
        mock_collection.query.return_value = [
            {"id": 1, "content": "内容1", "metadata": "{}"},
            {"id": 2, "content": "内容2", "metadata": '{"key": "value"}'},
        ]
        mock_collection.load = Mock()

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        results = repo.find_all(limit=2, offset=0)

        mock_collection.load.assert_called_once()
        mock_collection.query.assert_called_once()
        assert len(results) == 2
        assert results[0].id == 1
        assert results[1].id == 2
        assert results[1].metadata == {"key": "value"}

    def test_delete_by_id_calls_correct_method(self):
        """测试删除操作调用正确方法"""
        mock_collection = Mock()
        mock_collection.delete = Mock()
        mock_collection.flush = Mock()

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection
        # Don't set delete attribute at all so hasattr returns False
        del mock_vector_store.delete

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        repo.delete_by_id(42)

        mock_collection.delete.assert_called_once_with("id == 42")
        mock_collection.flush.assert_called_once()

    def test_delete_all_bulk_deletion(self):
        """测试批量删除"""
        mock_collection = Mock()
        mock_collection.delete = Mock()
        mock_collection.flush = Mock()

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection
        # Don't set delete attribute at all so hasattr returns False
        del mock_vector_store.delete

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        repo.delete_all([1, 2, 3])

        mock_collection.delete.assert_called_once_with("id in [1, 2, 3]")
        mock_collection.flush.assert_called_once()

    def test_count_returns_correct_number(self):
        """测试计数返回正确的文档数量"""
        mock_collection = Mock()
        mock_collection.num_entities = 42

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        count = repo.count()

        assert count == 42

    def test_search_by_vector_returns_documents(self):
        """测试向量搜索返回文档列表"""
        mock_doc1 = LCDocument(page_content="内容1", metadata={"id": "1", "key1": "value1"})
        mock_doc2 = LCDocument(page_content="内容2", metadata={"id": "2", "key2": "value2"})

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.similarity_search_by_vector.return_value = [mock_doc1, mock_doc2]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        query_embedding = [0.1, 0.2, 0.3]
        results = repo.search_by_vector(query_embedding, limit=2)

        mock_vector_store.similarity_search_by_vector.assert_called_once()
        assert len(results) == 2
        assert results[0].id == 1
        assert results[0].content == "内容1"
        assert results[0].metadata == {"key1": "value1"}
        assert results[1].id == 2
        assert results[1].metadata == {"key2": "value2"}

    def test_search_by_text_returns_documents(self):
        """测试文本搜索返回文档列表"""
        mock_doc1 = LCDocument(page_content="Python 教程", metadata={"id": "1", "category": "programming"})
        mock_doc2 = LCDocument(page_content="Python 高级编程", metadata={"id": "2", "category": "programming"})

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.similarity_search.return_value = [mock_doc1, mock_doc2]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        results = repo.search_by_text("Python", limit=2)

        mock_vector_store.similarity_search.assert_called_once()
        assert len(results) == 2
        assert results[0].id == 1
        assert results[0].content == "Python 教程"
        assert results[0].metadata == {"category": "programming"}

    def test_get_vector_store_returns_instance(self):
        """测试get_vector_store返回正确实例"""
        mock_vector_store = Mock(spec=VectorStore)

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        result = repo.get_vector_store()

        assert result == mock_vector_store

    def test_get_retriever_returns_retriever_instance(self):
        """测试get_retriever返回检索器实例"""
        mock_retriever = Mock()
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.as_retriever.return_value = mock_retriever

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        result = repo.get_retriever(search_kwargs={"k": 5})

        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
        assert result == mock_retriever

    def test_set_embedding_function_rebuilds_vector_store(self):
        """测试set_embedding_function重新构建VectorStore"""
        mock_vector_store1 = Mock(spec=VectorStore)
        mock_vector_store2 = Mock(spec=VectorStore)
        mock_embeddings1 = Mock(spec=Embeddings)
        mock_embeddings2 = Mock(spec=Embeddings)

        with patch('infrastructure.rag.embeddings.VectorStoreFactory') as mock_factory:
            mock_factory.create_store.return_value = mock_vector_store1
            repo = LangChainDocumentRepository(
                collection_name="test_collection",
                embedding_function=mock_embeddings1
            )
            assert repo._vector_store == mock_vector_store1

            mock_factory.create_store.return_value = mock_vector_store2
            repo.set_embedding_function(mock_embeddings2)

            assert repo._embedding_function == mock_embeddings2
            assert repo._vector_store == mock_vector_store2
            mock_factory.create_store.assert_called_with(
                embedding=mock_embeddings2,
                collection_name="test_collection"
            )
