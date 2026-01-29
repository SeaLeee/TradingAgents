"""
增强版TradingAgents API应用
提供丰富的市场数据分析和回测功能
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入API路由
from app.api import market, backtest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("Starting Enhanced TradingAgents API...")
    yield
    # 关闭时清理
    print("Shutting down Enhanced TradingAgents API...")


# 创建FastAPI应用
app = FastAPI(
    title="TradingAgents Enhanced API",
    description="提供ETF因子分析、市场热力图、增强回测等功能",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(market.router)
app.include_router(backtest.router)


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "TradingAgents Enhanced API",
        "version": "1.0.0",
        "endpoints": {
            "market": "/api/market",
            "backtest": "/api/backtest",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "enhanced_app:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )