"""
RAG Pipeline 单元测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from domain.document.entity.document import Document
from domain.document.entity.document_collection import DocumentCollection
from infrastructure.rag.document_loader.loader_factory import DocumentLoaderFactory
from infrastructure.rag.text_splitter.splitter_factory import TextSplitterFactory
from infrastructure.rag.embeddings.embedding_factory import EmbeddingFactory
from infrastructure.rag.pipeline.rag_pipeline import RAGPipeline, RAGPipelineFactory


class TestDocumentLoaderFactory:
    """文档加载器工厂测试"""

    def test_list_supported_loaders(self):
        """测试列出支持的加载器"""
        loaders = DocumentLoaderFactory.list_supported_loaders()
        assert isinstance(loaders, list)
        assert "pdf" in loaders
        assert "text" in loaders

    def test_get_loader_class_builtin(self):
        """测试获取内置加载器类"""
        from langchain_community.document_loaders import PyPDFLoader
        
        loader_class = DocumentLoaderFactory.get_loader_class("pdf")
        assert loader_class == PyPDFLoader

    def test_register_custom_loader(self):
        """测试注册自定义加载器"""
        mock_loader = Mock()
        DocumentLoaderFactory.register_loader("custom_test", mock_loader)
        
        result = DocumentLoaderFactory.get_loader_class("custom_test")
        assert result == mock_loader


class TestTextSplitterFactory:
    """文本分块器工厂测试"""

    def test_list_supported_splitters(self):
        """测试列出支持的分块器"""
        splitters = TextSplitterFactory.list_supported_splitters()
        assert isinstance(splitters, list)
        assert "recursive" in splitters
        assert "character" in splitters

    def test_create_splitter(self):
        """测试创建分块器"""
        splitter = TextSplitterFactory.create_splitter("recursive")
        assert splitter is not None

    def test_split_text(self):
        """测试文本分块"""
        text = "这是第一段。\n\n这是第二段。" * 100
        chunks = TextSplitterFactory.split_text(text, "recursive", chunk_size=100, chunk_overlap=20)
        assert isinstance(chunks, list)
        assert len(chunks) > 1


class TestEmbeddingFactory:
    """嵌入函数工厂测试"""

    def test_list_supported_providers(self):
        """测试列出支持的嵌入函数"""
        providers = EmbeddingFactory.list_supported_providers()
        assert isinstance(providers, list)
        assert "dashscope" in providers

    def test_get_embedding_dimension(self):
        """测试获取嵌入向量维度"""
        dimension = EmbeddingFactory.get_embedding_dimension("dashscope")
        assert dimension == 1536

    def test_register_custom_embedding(self):
        """测试注册自定义嵌入函数"""
        from langchain_core.embeddings import Embeddings
        
        class CustomEmbedding(Embeddings):
            def embed_documents(self, texts):
                return [[0.1] * 768 for _ in texts]
            
            def embed_query(self, text):
                return [0.1] * 768
        
        EmbeddingFactory.register_embedding("custom", CustomEmbedding)
        assert "custom" in EmbeddingFactory.list_supported_providers()


class TestRAGPipeline:
    """RAG Pipeline 测试"""

    @patch('infrastructure.rag.pipeline.rag_pipeline.LangChainDocumentCollectionRepository')
    @patch('infrastructure.rag.pipeline.rag_pipeline.LangChainDocumentRepository')
    def test_pipeline_initialization(self, mock_doc_repo, mock_collection_repo):
        """测试 Pipeline 初始化"""
        pipeline = RAGPipeline(domain="test_domain")
        
        assert pipeline.domain == "test_domain"
        assert pipeline.collection_name.startswith("doc_")

    @patch('infrastructure.rag.pipeline.rag_pipeline.LangChainDocumentCollectionRepository')
    def test_pipeline_factory(self, mock_repo):
        """测试 Pipeline 工厂"""
        pipeline1 = RAGPipelineFactory.get_pipeline("domain1")
        pipeline2 = RAGPipelineFactory.get_pipeline("domain1")
        
        assert pipeline1 is pipeline2  # 同一领域返回同一实例
        
        pipeline3 = RAGPipelineFactory.get_pipeline("domain2")
        assert pipeline1 is not pipeline3  # 不同领域返回不同实例
        
        domains = RAGPipelineFactory.list_domains()
        assert "domain1" in domains
        assert "domain2" in domains


class TestDocumentEntity:
    """文档实体测试"""

    def test_document_creation(self):
        """测试文档创建"""
        doc = Document(
            content="测试内容",
            metadata={"source": "test"},
        )
        
        assert doc.content == "测试内容"
        assert doc.metadata == {"source": "test"}
        assert doc.id is None  # 自增ID模式：插入前ID为None
        assert doc.embedding is None

    def test_document_with_id(self):
        """测试带ID的文档（模拟插入后的状态）"""
        doc = Document(
            id=123,
            content="测试内容",
            metadata={"source": "test"},
        )
        
        assert doc.id == 123

    def test_document_properties(self):
        """测试文档属性"""
        doc = Document(content="测试内容", embedding=[0.1, 0.2, 0.3])
        
        assert doc.text_length == 4
        assert doc.has_embedding is True

    def test_document_without_embedding(self):
        """测试无嵌入向量的文档"""
        doc = Document(content="测试内容")
        
        assert doc.has_embedding is False


class TestDocumentCollectionEntity:
    """文档集合实体测试"""

    def test_collection_creation(self):
        """测试集合创建"""
        collection = DocumentCollection(
            name="test_collection",
            description="测试集合",
        )
        
        assert collection.name == "test_collection"
        assert collection.description == "测试集合"
        assert collection.documents == []
        assert collection.id is not None

    def test_collection_add_document(self):
        """测试添加文档到集合"""
        collection = DocumentCollection(name="test_collection")
        doc = Document(content="测试文档")
        doc.id = 1  # 模拟数据库分配的ID
        
        collection.add_document(doc)
        
        assert collection.get_document_count() == 1
        assert collection.get_document(doc.id) == doc

    def test_collection_remove_document(self):
        """测试从集合移除文档"""
        collection = DocumentCollection(name="test_collection")
        doc = Document(content="测试文档")
        doc.id = 1  # 模拟数据库分配的ID
        
        collection.add_document(doc)
        collection.remove_document(doc.id)
        
        assert collection.get_document_count() == 0
        assert collection.get_document(doc.id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
