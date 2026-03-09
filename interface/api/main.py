from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import argparse
import uvicorn

from interface.api.handle import router, init_qa_agent, cleanup_qa_agent
from interface.api.document_routes import router as document_router
from interface.container import container


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger = container.get_logger()
    # 启动时
    try:
        # 初始化底层组件
        from application.common.app_initializer import AppInitializer
        app_initializer = AppInitializer.get_instance()
        app_initializer.initialize()
        # 初始化QA代理
        init_qa_agent()
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        raise

    yield

    # 关闭时
    try:
        cleanup_qa_agent()
        # 关闭底层组件
        from application.common.app_initializer import AppInitializer
        app_initializer = AppInitializer.get_instance()
        app_initializer.shutdown()
    except Exception as e:
        logger.error(f"应用清理失败: {str(e)}")

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


@app.get("/health", summary="健康检查接口", description="返回应用和所有底层组件的健康状态")
async def health_check():
    """健康检查接口"""
    from application.common.app_initializer import AppInitializer
    app_initializer = AppInitializer.get_instance()
    return app_initializer.health_check()


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