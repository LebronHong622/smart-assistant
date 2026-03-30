"""
文档管理应用服务
负责文档CRUD操作的协调和编排，不包含纯业务逻辑
"""

from typing import List, Optional
from uuid import UUID
from domain.entity.document.document import Document
from domain.vo.document.document_metadata import DocumentMetadata
from domain.repository.document.document_repository import DocumentRepository
from domain.shared.ports.logger_port import LoggerPort
from domain.service.document.document_chunking_service import DocumentChunkingService, ChunkingConfig
from domain.service.document.document_validation_service import DocumentValidationService


class DocumentManagementService:
    """
    文档管理应用服务
    协调文档的CRUD操作，处理基础设施和领域服务之间的交互
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        logger: LoggerPort,
        chunking_service: Optional[DocumentChunkingService] = None,
        validation_service: Optional[DocumentValidationService] = None
    ):
        self.document_repository = document_repository
        self.logger = logger
        self.chunking_service = chunking_service or DocumentChunkingService()
        self.validation_service = validation_service or DocumentValidationService()

    def create_document(self, content: str, metadata: DocumentMetadata, chunk_config: Optional[ChunkingConfig] = None) -> List[Document]:
        """
        创建文档（可选项：自动分块）

        Args:
            content: 文档内容
            metadata: 文档元数据
            chunk_config: 分块配置，如果为None则不分块

        Returns:
            文档实体列表（如果分块则返回多个文档块，否则返回单个文档）
        """
        self.logger.info(f"创建文档: {metadata.title}")

        # 验证文档内容
        validation_result = self.validation_service.validate_document(content, metadata.model_dump())
        if not validation_result.is_valid:
            error_messages = [f"{e.field}: {e.message}" if e.field else e.message for e in validation_result.errors if e.severity == "error"]
            raise ValueError(f"文档验证失败: {', '.join(error_messages)}")

        if chunk_config:
            # 分块处理
            chunks = self.chunking_service.chunk_document(content, chunk_config)
            self.logger.info(f"文档已分块: {len(chunks)} 个块")

            documents = []
            for i, chunk_content in enumerate(chunks):
                # 验证分块
                chunk_validation = self.validation_service.validate_chunk(chunk_content)
                if not chunk_validation.is_valid:
                    self.logger.warning(f"跳过无效分块 {i+1}: {chunk_validation.errors[0].message if chunk_validation.errors else '未知错误'}")
                    continue

                # 为每个块创建文档
                chunk_metadata = metadata.model_dump().copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["is_chunk"] = True

                document = Document(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    embedding=None
                )

                # 保存文档
                saved_document = self.document_repository.save(document)
                documents.append(saved_document)

            self.logger.info(f"创建了 {len(documents)} 个文档块")
            return documents
        else:
            # 不分块，创建单个文档
            document = Document(
                content=content,
                metadata=metadata.model_dump(),
                embedding=None
            )

            saved_document = self.document_repository.save(document)
            self.logger.info(f"创建了单个文档: ID={saved_document.id}")
            return [saved_document]

    def update_document(self, document_id: UUID, content: Optional[str] = None, metadata: Optional[DocumentMetadata] = None) -> Document:
        """
        更新文档

        Args:
            document_id: 文档ID
            content: 新的文档内容（可选）
            metadata: 新的文档元数据（可选）

        Returns:
            更新后的文档
        """
        self.logger.info(f"更新文档: {document_id}")

        # 获取现有文档
        document = self.document_repository.find_by_id(document_id)
        if not document:
            raise RuntimeError(f"文档不存在: {document_id}")

        # 验证更新内容
        if content is not None:
            validation_result = self.validation_service.validate_content(content)
            if not validation_result.is_valid:
                error_messages = [e.message for e in validation_result.errors]
                raise ValueError(f"内容验证失败: {', '.join(error_messages)}")

        # 更新内容
        if content is not None:
            document.content = content

        # 更新元数据
        if metadata is not None:
            metadata_dict = metadata.model_dump()
            validation_result = self.validation_service.validate_metadata(metadata_dict)
            if not validation_result.is_valid:
                error_messages = [f"{e.field}: {e.message}" if e.field else e.message for e in validation_result.errors if e.severity == "error"]
                raise ValueError(f"元数据验证失败: {', '.join(error_messages)}")
            document.metadata = metadata_dict

        # 保存更新
        updated_document = self.document_repository.save(document)
        self.logger.info(f"文档更新成功: ID={updated_document.id}")
        return updated_document

    def delete_document(self, document_id: UUID) -> None:
        """
        删除文档

        Args:
            document_id: 文档ID
        """
        self.logger.info(f"删除文档: {document_id}")
        self.document_repository.delete_by_id(document_id)
        self.logger.info(f"文档删除成功: {document_id}")

    def get_document(self, document_id: UUID) -> Optional[Document]:
        """
        获取文档

        Args:
            document_id: 文档ID

        Returns:
            文档实体，如果不存在则返回None
        """
        return self.document_repository.find_by_id(document_id)

    def list_documents(self, limit: int = 10, offset: int = 0) -> List[Document]:
        """
        列表获取文档

        Args:
            limit: 每页数量
            offset: 偏移量

        Returns:
            文档列表
        """
        self.logger.debug(f"获取文档列表: limit={limit}, offset={offset}")
        return self.document_repository.find_all(limit=limit, offset=offset)

    def search_documents_by_metadata(self, metadata: dict) -> List[Document]:
        """
        根据元数据搜索文档

        Args:
            metadata: 元数据查询条件

        Returns:
            匹配的文档列表
        """
        self.logger.debug(f"根据元数据搜索文档: {metadata}")

        # 获取所有文档并筛选
        all_documents = self.document_repository.find_all()

        filtered_docs = []
        for doc in all_documents:
            match = True
            for key, value in metadata.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break

            if match:
                filtered_docs.append(doc)

        self.logger.info(f"根据元数据搜索到 {len(filtered_docs)} 个文档")
        return filtered_docs

    def count_documents(self) -> int:
        """
        统计文档数量

        Returns:
            文档总数
        """
        count = self.document_repository.count()
        self.logger.debug(f"文档总数: {count}")
        return count

    def get_chunking_config(self, strategy: str = "recursive_character", chunk_size: int = 500, chunk_overlap: int = 50) -> ChunkingConfig:
        """
        获取分块配置

        Args:
            strategy: 分块策略
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            分块配置对象
        """
        return ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy
        )
