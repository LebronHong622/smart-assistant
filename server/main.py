import sys
import os
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 获取当前脚本的父目录的路径，即`qanything_server`目录
current_dir = os.path.dirname(current_script_path)

# 获取`qanything_server`目录的父目录，即`qanything_kernel`
parent_dir = os.path.dirname(current_dir)

# 获取根目录：`qanything_kernel`的父目录
root_dir = os.path.dirname(parent_dir)

# 将项目根目录添加到sys.path
sys.path.append(root_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import argparse
import uvicorn

from handle import router, init_qa_agent, cleanup_qa_agent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    try:
        init_qa_agent()
    except Exception as e:
        logger.error(f"应用初始化失败: {str(e)}")
        raise
    
    yield
    
    # 关闭时
    try:
        cleanup_qa_agent()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动智能助手API服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", default=True, help="启用自动重载")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
