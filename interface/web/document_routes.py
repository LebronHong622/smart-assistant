"""
文档管理 API 接口（重构后）
基于 FastAPI 的文档操作接口 - Interface 层仅做参数校验和响应格式化
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from interface.container import container
from interface.web.dto.document_dto import (
    UploadDocumentRequestDTO,
    UploadDocumentResponseDTO,
    RetrieveDocumentsRequestDTO,
    RetrieveDocumentsResponseDTO,
    CreateCollectionRequestDTO,
    CreateCollectionResponseDTO,
    CollectionInfoResponseDTO,
    DeleteDocumentResponseDTO,
    ListCollectionsResponseDTO,
    DeleteCollectionResponseDTO,
)

# 导入 Application 层 Command
from application.services.document.commands import (
    UploadDocumentCommand,
    RetrieveDocumentsCommand,
    CreateCollectionCommand,
)

router = APIRouter()

# 文档管理路由前缀
DOCUMENT_API_PREFIX = "/documents"


@router.post("/upload", response_model=UploadDocumentResponseDTO, summary="上传文档", description="上传文档到指定集合")
async def upload_document(request: UploadDocumentRequestDTO):
    """
    上传文档
    Interface 层：接收 DTO → 转换为 Command → 调 Application Service → 返回 Response DTO
    """
    try:
        app_service = container.get_document_app_service()

        # DTO → Command（简单字段映射）
        command = UploadDocumentCommand(**request.model_dump())

        # 调用 Application 层（转换逻辑下沉到这里）
        result = app_service.upload_document(command)

        # Result → Response DTO
        return UploadDocumentResponseDTO(**result.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieve", response_model=RetrieveDocumentsResponseDTO, summary="检索文档", description="根据查询文本检索相似文档")
async def retrieve_documents(request: RetrieveDocumentsRequestDTO):
    """
    检索文档
    Interface 层：接收 DTO → 转换为 Command → 调 Application Service → 返回 Response DTO
    """
    try:
        app_service = container.get_document_app_service(domain=request.domain)

        # DTO → Command
        command = RetrieveDocumentsCommand(**request.model_dump())

        # 调用 Application 层（Entity → Result 转换下沉到这里）
        result = app_service.retrieve_documents(command)

        # Result → Response DTO
        return RetrieveDocumentsResponseDTO(**result.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info", response_model=CollectionInfoResponseDTO, summary="获取集合信息", description="获取默认集合的信息")
async def get_collection_info(collection_name: Optional[str] = None):
    """获取集合信息"""
    try:
        app_service = container.get_document_app_service()
        result = app_service.get_collection_info(collection_name)
        return CollectionInfoResponseDTO(**result.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DeleteDocumentResponseDTO, summary="删除文档", description="从集合中删除指定文档")
async def delete_document(document_id: str, collection_name: Optional[str] = None):
    """删除文档"""
    try:
        app_service = container.get_document_app_service()
        result = app_service.delete_document(document_id, collection_name)
        return DeleteDocumentResponseDTO(**result.model_dump())
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="删除文档功能未实现")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Collection 管理接口 ====================

@router.get("/collections", response_model=ListCollectionsResponseDTO, summary="列出所有集合", description="获取所有 Milvus 集合列表")
async def list_collections():
    """列出所有 Collection"""
    try:
        app_service = container.get_document_app_service()
        result = app_service.list_collections()
        return ListCollectionsResponseDTO(**result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collections", response_model=CreateCollectionResponseDTO, summary="创建集合", description="创建新的文档集合")
async def create_collection(request: CreateCollectionRequestDTO):
    """创建新的 Collection"""
    try:
        app_service = container.get_document_app_service()
        command = CreateCollectionCommand(**request.model_dump())
        result = app_service.create_collection(command)
        return CreateCollectionResponseDTO(**result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/collections/{collection_name}", response_model=DeleteCollectionResponseDTO, summary="删除集合", description="删除指定的文档集合")
async def delete_collection(collection_name: str):
    """删除 Collection"""
    try:
        app_service = container.get_document_app_service()
        result = app_service.delete_collection(collection_name)
        return DeleteCollectionResponseDTO(**result.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{collection_name}/info", response_model=CollectionInfoResponseDTO, summary="获取集合详细信息", description="获取指定集合的详细信息")
async def get_specific_collection_info(collection_name: str):
    """获取指定 Collection 的详细信息"""
    try:
        app_service = container.get_document_app_service()
        result = app_service.get_specific_collection_info(collection_name)
        return CollectionInfoResponseDTO(**result.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
