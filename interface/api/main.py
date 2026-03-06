from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import argparse
import uvicorn

from interface.api.handle import router, init_qa_agent, cleanup_qa_agent
from interface.api.document_routes import router as document_router
from infrastructure.log import app_logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    try:
        init_qa_agent()
    except Exception as e:
        app_logger.error(f"应用初始化失败: {str(e)}")
        raise

    yield

    # 关闭时
    try:
        cleanup_qa_agent()
    except Exception as e:
        app_logger.error(f"应用清理失败: {str(e)}")

# 创建FastAPI应用实例
app = FastAPI(
    title="智能助手API",
    description="多任务问答助手的FastAPI接口",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用add_route加载路由
app.include_router(router, prefix="", tags=["assistant"])
app.include_router(document_router, prefix="/documents", tags=["documents"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动智能助手API服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", default=True, help="启用自动重载")

    args = parser.parse_args()

    uvicorn.run(
        "interface.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )