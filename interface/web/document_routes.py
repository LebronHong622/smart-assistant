"""
文档管理 API 接口
基于 FastAPI 的文档操作接口
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from datetime import datetime

from domain.document.entity.document import Document
from domain.document.value_object.document_metadata import DocumentMetadata, DocumentType, DocumentSource
from interface.container import container
from interface.web.dto import (
    UploadDocumentRequestDTO,
    UploadDocumentResponseDTO,
    RetrieveDocumentsRequestDTO,
    RetrieveDocumentsResponseDTO,
    RetrieveDocumentsResultDTO,
    CreateCollectionRequestDTO,
    CreateCollectionResponseDTO,
    CollectionInfoResponseDTO,
    DeleteDocumentResponseDTO,
    ListCollectionsResponseDTO,
    DeleteCollectionResponseDTO
)

router = APIRouter()

# 文档管理路由前缀
DOCUMENT_API_PREFIX = "/documents"


@router.post("/upload", response_model=UploadDocumentResponseDTO, summary="上传文档", description="上传文档到指定集合")
async def upload_document(request: UploadDocumentRequestDTO):
    """
    上传文档

    将文档内容嵌入向量并存储到 Milvus 集合中
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 创建文档元数据
        metadata = DocumentMetadata(
            title=request.title,
            document_type=DocumentType[request.document_type.upper()],
            source=DocumentSource[request.source.upper()],
            created_at=current_time,
            updated_at=current_time
        )

        # 创建文档实体，将 extra_fields 作为顶层字段
        doc_data = {
            "content": request.content,
            "metadata": metadata.model_dump()
        }

        # 添加额外字段到顶层
        if request.extra_fields:
            for key, value in request.extra_fields.items():
                doc_data[key] = value

        document = Document(**doc_data)

        # 通过容器获取RAG处理服务
        rag_service = container.get_rag_processing_service()
        processed_document = rag_service.process_document(document)

        return UploadDocumentResponseDTO(
            success=True,
            document_id=str(processed_document.id),
            title=request.title,
            message="文档上传成功"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrieveDocumentsResponseDTO, summary="检索文档", description="根据查询文本检索相似文档")
async def retrieve_documents(request: RetrieveDocumentsRequestDTO):
    """
    检索文档

    基于向量相似度检索相关文档
    """
    try:
        # 通过容器获取RAG处理服务
        rag_service = container.get_rag_processing_service()
        results = rag_service.retrieve_similar(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        # 转换为响应 DTO
        result_items = []
        for doc in results:
            result_items.append(RetrieveDocumentsResultDTO(
                document_id=str(doc.id),
                content=doc.content,
                metadata=doc.metadata,
                similarity_score=None,
                distance=None
            ))

        return RetrieveDocumentsResponseDTO(
            success=True,
            query=request.query,
            result_count=len(results),
            results=result_items
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=CollectionInfoResponseDTO, summary="获取集合信息", description="获取默认集合的信息")
async def get_collection_info(collection_name: Optional[str] = None):
    """
    获取集合信息

    返回集合的统计信息、schema 配置等
    """
    try:
        # 通过容器获取应用服务
        collection_service = container.get_collection_service()
        if collection_name:
            collection = collection_service.get_collection_by_name(collection_name)
            if not collection:
                raise HTTPException(status_code=404, detail=f"集合不存在: {collection_name}")
            info = collection_service.get_collection_info(collection.id)
        else:
            # 使用默认集合
            from config.settings import settings
            default_collection_name = settings.milvus.milvus_collection_name
            collection = collection_service.get_collection_by_name(default_collection_name)
            if not collection:
                # 创建默认集合
                collection = collection_service.create_collection(default_collection_name, "默认文档集合")
            info = collection_service.get_collection_info(collection.id)

        return CollectionInfoResponseDTO(
            success=True,
            collection_info=info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DeleteDocumentResponseDTO, summary="删除文档", description="从集合中删除指定文档")
async def delete_document(document_id: str, collection_name: Optional[str] = None):
    """
    删除文档

    根据文档 ID 从指定集合中删除文档
    """
    try:
        # 通过容器获取应用服务
        retrieval_service = container.get_document_retrieval_service()
        retrieval_service.remove_document_from_collection(document_id, collection_name=collection_name)

        return DeleteDocumentResponseDTO(
            success=True,
            document_id=document_id,
            message="文档删除成功"
        )

    except NotImplementedError:
        raise HTTPException(status_code=501, detail="删除文档功能未实现")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Collection 管理接口 ====================

@router.get("/collections", response_model=ListCollectionsResponseDTO, summary="列出所有集合", description="获取所有 Milvus 集合列表")
async def list_collections():
    """
    列出所有 Collection

    返回系统中所有可用的文档集合
    """
    try:
        # 通过容器获取应用服务
        collection_service = container.get_collection_service()
        collections = collection_service.list_collections()
        collection_names = [col.name for col in collections]
        return ListCollectionsResponseDTO(
            success=True,
            collections=collection_names,
            count=len(collection_names)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections", response_model=CreateCollectionResponseDTO, summary="创建集合", description="创建新的文档集合")
async def create_collection(request: CreateCollectionRequestDTO):
    """
    创建新的 Collection

    根据配置创建新的向量集合
    """
    try:
        # 通过容器获取应用服务
        collection_service = container.get_collection_service()
        collection = collection_service.create_collection(
            name=request.collection_name,
            description=request.description
        )

        return CreateCollectionResponseDTO(
            success=True,
            collection_name=collection.name,
            message="Collection 创建成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}", response_model=DeleteCollectionResponseDTO, summary="删除集合", description="删除指定的文档集合")
async def delete_collection(collection_name: str):
    """
    删除 Collection

    删除指定名称的集合及其所有数据
    """
    try:
        # 通过容器获取应用服务
        collection_service = container.get_collection_service()
        collection = collection_service.get_collection_by_name(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"集合不存在: {collection_name}")

        collection_service.delete_collection(collection.id)

        return DeleteCollectionResponseDTO(
            success=True,
            collection_name=collection_name,
            message="Collection 删除成功"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/info", response_model=CollectionInfoResponseDTO, summary="获取集合详细信息", description="获取指定集合的详细信息")
async def get_specific_collection_info(collection_name: str):
    """
    获取指定 Collection 的详细信息

    返回集合的统计信息、schema 配置等
    """
    try:
        # 通过容器获取应用服务
        collection_service = container.get_collection_service()
        collection = collection_service.get_collection_by_name(collection_name)
        if not collection:
            raise HTTPException(status_code=404, detail=f"集合不存在: {collection_name}")

        info = collection_service.get_collection_info(collection.id)

        return CollectionInfoResponseDTO(
            success=True,
            collection_info=info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))