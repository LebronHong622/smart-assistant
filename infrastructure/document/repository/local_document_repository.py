"""
DocumentRepository 接口的本地磁盘存储实现
"""
import json
import os
from pathlib import Path
from typing import List, Optional
from uuid import UUID
from domain.document.entity.document import Document
from domain.document.repository.document_repository import DocumentRepository
from config.settings import settings
from infrastructure.log import app_logger


class LocalDocumentRepository(DocumentRepository):
    """
    基于本地文件系统的文档仓库实现
    将文档以 JSON 格式存储到本地磁盘
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = Path(storage_path or settings.document_storage.document_storage_path)
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, document_id: UUID) -> Path:
        """获取文档存储的文件路径"""
        return self.storage_path / f"{document_id}.json"

    def _document_to_dict(self, document: Document) -> dict:
        """将文档对象转换为字典"""
        return {
            "id": str(document.id),
            "content": document.content,
            "metadata": document.metadata,
            "embedding": document.embedding
        }

    def _dict_to_document(self, doc_dict: dict) -> Document:
        """将字典转换为文档对象"""
        return Document(
            id=UUID(doc_dict["id"]),
            content=doc_dict["content"],
            metadata=doc_dict.get("metadata", {}),
            embedding=doc_dict.get("embedding")
        )

    def save(self, document: Document) -> Document:
        """保存文档"""
        app_logger.info(f"保存文档: {document.id}")

        file_path = self._get_file_path(document.id)
        doc_dict = self._document_to_dict(document)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
            app_logger.info(f"文档保存成功: {file_path}")
        except Exception as e:
            app_logger.error(f"保存文档失败: {str(e)}")
            raise RuntimeError(f"保存文档失败: {str(e)}")

        return document

    def save_all(self, documents: List[Document]) -> List[Document]:
        """保存多个文档"""
        app_logger.info(f"批量保存文档: {len(documents)} 个")

        for doc in documents:
            self.save(doc)

        return documents

    def find_by_id(self, document_id: UUID) -> Optional[Document]:
        """根据 ID 查找文档"""
        app_logger.debug(f"查找文档: {document_id}")

        file_path = self._get_file_path(document_id)

        if not file_path.exists():
            app_logger.debug(f"文档不存在: {document_id}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_dict = json.load(f)
            return self._dict_to_document(doc_dict)
        except Exception as e:
            app_logger.error(f"读取文档失败: {str(e)}")
            return None

    def find_all(self) -> List[Document]:
        """查找所有文档"""
        app_logger.debug("查找所有文档")

        documents = []

        try:
            # 遍历存储目录中的所有 JSON 文件
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_dict = json.load(f)
                    documents.append(self._dict_to_document(doc_dict))
                except Exception as e:
                    app_logger.warning(f"读取文件 {file_path} 失败: {str(e)}")
                    continue

            app_logger.info(f"找到 {len(documents)} 个文档")
        except Exception as e:
            app_logger.error(f"查找所有文档失败: {str(e)}")

        return documents

    def delete(self, document: Document) -> None:
        """删除文档"""
        self.delete_by_id(document.id)

    def delete_by_id(self, document_id: UUID) -> None:
        """根据 ID 删除文档"""
        app_logger.info(f"删除文档: {document_id}")

        file_path = self._get_file_path(document_id)

        if not file_path.exists():
            app_logger.warning(f"文档不存在，无需删除: {document_id}")
            return

        try:
            os.remove(file_path)
            app_logger.info(f"文档删除成功: {document_id}")
        except Exception as e:
            app_logger.error(f"删除文档失败: {str(e)}")
            raise RuntimeError(f"删除文档失败: {str(e)}")

    def count(self) -> int:
        """统计文档数量"""
        try:
            # 统计存储目录中的 JSON 文件数量
            count = len(list(self.storage_path.glob("*.json")))
            app_logger.debug(f"文档总数: {count}")
            return count
        except Exception as e:
            app_logger.error(f"统计文档数量失败: {str(e)}")
            return 0
