"""
文档管理 API 接口
基于 FastAPI 的文档操作接口
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from uuid import UUID, uuid4
from domain.document.entity.document import Document
from domain.document.value_object.document_metadata import DocumentMetadata, DocumentType, DocumentSource
from application.document.document_retrieval_service_impl import MilvusDocumentRetrievalService

router = APIRouter()

# 文档管理路由前缀
DOCUMENT_API_PREFIX = "/documents"


@router.post("/upload")
async def upload_document(content: str, title: str, document_type: str = "txt", source: str = "upload"):
    """
    上传文档
    """
    try:
        # 创建文档元数据
        metadata = DocumentMetadata(
            title=title,
            document_type=DocumentType[document_type.upper()],
            source=DocumentSource[source.upper()],
            created_at="2024-01-01 00:00:00",
            updated_at="2024-01-01 00:00:00"
        )

        # 创建文档实体
        document = Document(
            id=uuid4(),
            content=content,
            metadata=metadata.__dict__
        )

        # 添加到检索集合
        retrieval_service = MilvusDocumentRetrievalService()
        retrieval_service.add_document_to_collection(document)

        return {
            "success": True,
            "document_id": str(document.id),
            "title": title,
            "message": "文档上传成功"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve")
async def retrieve_documents(query: str, limit: int = 5, score_threshold: float = 0.5):
    """
    检索文档
    """
    try:
        retrieval_service = MilvusDocumentRetrievalService()
        results = retrieval_service.retrieve_similar_documents(
            query=query,
            limit=limit,
            score_threshold=score_threshold
        )

        # 格式化结果
        formatted_results = {
            "success": True,
            "query": query,
            "result_count": len(results),
            "results": []
        }

        for result in results:
            formatted_results["results"].append({
                "document_id": str(result.document_id),
                "content": result.content,
                "metadata": result.metadata,
                "similarity_score": round(result.similarity_score, 4),
                "distance": round(result.distance, 4)
            })

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_collection_info():
    """
    获取集合信息
    """
    try:
        from infrastructure.vector.vector_store import MilvusVectorStore
        vector_store = MilvusVectorStore()
        info = vector_store.get_collection_info()

        return {
            "success": True,
            "collection_info": info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    删除文档
    """
    try:
        retrieval_service = MilvusDocumentRetrievalService()
        retrieval_service.remove_document_from_collection(document_id)

        return {
            "success": True,
            "document_id": document_id,
            "message": "文档删除成功"
        }

    except NotImplementedError:
        raise HTTPException(status_code=501, detail="删除文档功能未实现")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))