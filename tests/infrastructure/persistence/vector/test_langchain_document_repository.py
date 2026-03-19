"""
Unit tests for LangChainDocumentRepository
使用 mock 进行单元测试，不需要外部 Milvus 服务
"""
import pytest
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from domain.document.entity.document import Document
from infrastructure.persistence.vector.repository.langchain_document_repository_impl import LangChainDocumentRepository


class TestLangChainDocumentRepository:
    """LangChainDocumentRepository 单元测试"""

    def test_initialization_auto_create(self):
        """测试单参数自动初始化（从工厂创建 embedding 和 vector_store）"""
        with patch('infrastructure.rag.embeddings.EmbeddingFactory') as mock_emb_factory, \
             patch('infrastructure.rag.embeddings.VectorStoreFactory') as mock_store_factory:

            mock_embedding_generator = Mock()
            mock_embeddings = Mock(spec=Embeddings)
            mock_embedding_generator.to_langchain_embeddings.return_value = mock_embeddings
            mock_emb_factory.create_embedding.return_value = mock_embedding_generator

            mock_vector_store = Mock(spec=VectorStore)
            mock_store_factory.create_store.return_value = mock_vector_store

            repo = LangChainDocumentRepository(
                collection_name="test_collection"
            )

            mock_emb_factory.create_embedding.assert_called_once()
            mock_store_factory.create_store.assert_called_once_with(
                embedding=mock_embeddings,
                collection_name="test_collection"
            )
            assert repo._collection_name == "test_collection"
            assert repo._embedding_function == mock_embeddings
            assert repo._vector_store == mock_vector_store

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

        # ID 保持 None（无法转换为整数，所以不会被赋值）
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

    def test_find_by_id_raises_not_implemented(self):
        """测试 find_by_id 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        with pytest.raises(NotImplementedError, match="find_by_id not implemented yet"):
            repo.find_by_id(42)

    def test_find_all_raises_not_implemented(self):
        """测试 find_all 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        with pytest.raises(NotImplementedError, match="find_all not implemented yet"):
            repo.find_all(limit=2, offset=0)

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

        mock_collection.delete.assert_called_once_with("pk == 42")
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

        mock_collection.delete.assert_called_once_with("pk in [1, 2, 3]")
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

    def test_search_by_vector_raises_not_implemented(self):
        """测试 search_by_vector 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        query_embedding = [0.1, 0.2, 0.3]
        with pytest.raises(NotImplementedError, match="search_by_vector with search_type not implemented yet"):
            repo.search_by_vector(query_embedding, limit=2)

    def test_search_by_text_default_similarity(self):
        """测试文本搜索默认使用 similarity"""
        mock_doc1 = LCDocument(page_content="Python 教程", metadata={"id": "1", "category": "programming"})
        mock_doc2 = LCDocument(page_content="Python 高级编程", metadata={"id": "2", "category": "programming"})

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = [mock_doc1, mock_doc2]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        results = repo.search_by_text("Python", limit=2)

        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs['search_type'] == "similarity"
        assert call_args.kwargs['k'] == 2
        assert call_args.kwargs['query'] == "Python"
        assert len(results) == 2
        assert results[0].id == 1
        assert results[0].content == "Python 教程"
        assert results[0].metadata == {"category": "programming"}

    def test_search_by_text_similarity(self):
        """测试文本搜索支持 similarity 类型"""
        mock_doc1 = LCDocument(page_content="Python 教程", metadata={"id": "1", "category": "programming"})

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = [mock_doc1]

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        results = repo.search_by_text("Python", limit=1, search_type="similarity", score_threshold=0.8)

        mock_vector_store.search.assert_called_once()
        call_args = mock_vector_store.search.call_args
        assert call_args.kwargs['search_type'] == "similarity"
        assert call_args.kwargs['k'] == 1
        assert call_args.kwargs['score_threshold'] == 0.8
        assert len(results) == 1

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

    # ========== 异步方法测试 ==========

    @pytest.mark.asyncio
    async def test_asave_document_assigns_id_correctly(self):
        """测试异步保存文档正确分配ID"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.aadd_documents = AsyncMock(return_value=["123"])

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        document = Document(content="测试内容", metadata={"key": "value"})
        assert document.id is None

        result = await repo.asave(document)

        mock_vector_store.aadd_documents.assert_called_once()
        assert result.id == 123

    @pytest.mark.asyncio
    async def test_asave_all_bulk_operation(self):
        """测试异步批量保存操作"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.aadd_documents = AsyncMock(return_value=["1", "2", "3"])

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        documents = [
            Document(content="内容1"),
            Document(content="内容2"),
        ]

        results = await repo.asave_all(documents)

        mock_vector_store.aadd_documents.assert_called_once()
        assert len(results) == 2
        assert results[0].id == 1
        assert results[1].id == 2

    @pytest.mark.asyncio
    async def test_afind_by_id_raises_not_implemented(self):
        """测试异步 find_by_id 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        with pytest.raises(NotImplementedError, match="afind_by_id not implemented yet"):
            await repo.afind_by_id(42)

    @pytest.mark.asyncio
    async def test_afind_all_raises_not_implemented(self):
        """测试异步 find_all 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        with pytest.raises(NotImplementedError, match="afind_all not implemented yet"):
            await repo.afind_all(limit=2)

    @pytest.mark.asyncio
    async def test_adelete_by_id_calls_async_method(self):
        """测试异步删除调用正确方法"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.adelete = AsyncMock()

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        await repo.adelete_by_id(42)

        mock_vector_store.adelete.assert_called_once_with(["42"])

    @pytest.mark.asyncio
    async def test_adelete_all_calls_async_method(self):
        """测试异步批量删除调用正确方法"""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.adelete = AsyncMock()

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        await repo.adelete_all([1, 2, 3])

        mock_vector_store.adelete.assert_called_once_with(["1", "2", "3"])

    @pytest.mark.asyncio
    async def test_acount_reuses_sync(self):
        """测试异步计数复用同步逻辑"""
        mock_collection = Mock()
        mock_collection.num_entities = 42
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.col = mock_collection

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        count = await repo.acount()
        assert count == 42

    @pytest.mark.asyncio
    async def test_asearch_by_text_default_similarity(self):
        """测试异步文本搜索默认使用 similarity"""
        mock_doc1 = LCDocument(page_content="Python 教程", metadata={"id": "1"})
        mock_doc2 = LCDocument(page_content="Python 高级编程", metadata={"id": "2"})

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.asearch = AsyncMock(return_value=[mock_doc1, mock_doc2])

        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        results = await repo.asearch_by_text("Python", limit=2)

        mock_vector_store.asearch.assert_called_once()
        call_args = mock_vector_store.asearch.call_args
        assert call_args.kwargs['search_type'] == "similarity"
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_asearch_by_vector_raises_not_implemented(self):
        """测试异步 search_by_vector 抛出 NotImplementedError"""
        mock_vector_store = Mock(spec=VectorStore)
        repo = LangChainDocumentRepository(
            collection_name="test_collection",
            vector_store=mock_vector_store
        )

        query_embedding = [0.1, 0.2, 0.3]
        with pytest.raises(NotImplementedError, match="asearch_by_vector with search_type not implemented yet"):
            await repo.asearch_by_vector(query_embedding, limit=2)
