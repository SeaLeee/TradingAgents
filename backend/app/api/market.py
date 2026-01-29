"""
市场数据API路由
提供ETF因子分析、市场热力图数据等接口
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
from app.modules.etf_factors import ETFFactorAnalyzer
from app.modules.enhanced_backtest import EnhancedBacktest

router = APIRouter(
    prefix="/api/market",
    tags=["market"]
)

# 初始化因子分析器
factor_analyzer = ETFFactorAnalyzer()

@router.get("/etf-factors")
async def get_etf_factors(
    etf_type: str = Query("sector", description="ETF类型: sector(行业) 或 broad(宽基)"),
    period_days: int = Query(365, description="历史数据天数")
):
    """获取ETF因子分析数据"""
    try:
        # 根据类型选择ETF列表
        if etf_type == "sector":
            symbols = list(factor_analyzer.SECTOR_ETFS.keys())
        elif etf_type == "broad":
            symbols = list(factor_analyzer.BROAD_ETFS.keys())
        else:
            symbols = list(factor_analyzer.all_etfs.keys())

        # 计算开始日期
        start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

        # 计算所有因子
        factors_df = factor_analyzer.calculate_all_factors(symbols, start_date)

        if factors_df.empty:
            return {
                "success": False,
                "message": "无法获取数据",
                "data": []
            }

        # 转换为字典列表
        factors_data = factors_df.to_dict('records')

        # 处理NaN值
        for record in factors_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    record[key] = round(float(value), 2)

        return {
            "success": True,
            "data": factors_data,
            "update_time": datetime.now().isoformat(),
            "etf_count": len(factors_data)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/etf-heatmap")
async def get_etf_heatmap(
    sort_by: str = Query("1d", description="排序字段"),
    ascending: bool = Query(False, description="是否升序")
):
    """获取ETF热力图数据"""
    try:
        # 获取所有ETF的因子数据
        symbols = list(factor_analyzer.all_etfs.keys())
        factors_df = factor_analyzer.calculate_all_factors(symbols)

        if factors_df.empty:
            return {
                "success": False,
                "message": "无法获取数据",
                "data": []
            }

        # 根据指定字段排序
        if sort_by in factors_df.columns:
            factors_df = factors_df.sort_values(by=sort_by, ascending=ascending)

        # 准备热力图数据
        heatmap_data = []
        for _, row in factors_df.iterrows():
            heatmap_item = {
                "symbol": row['symbol'],
                "name": row['name'],
                "returns": {
                    "1d": row.get('1d', 0),
                    "5d": row.get('5d', 0),
                    "10d": row.get('10d', 0),
                    "20d": row.get('20d', 0),
                    "60d": row.get('60d', 0),
                    "120d": row.get('120d', 0),
                    "250d": row.get('250d', 0),
                },
                "factors": {
                    "momentum_20d": row.get('momentum_20d', 0),
                    "momentum_60d": row.get('momentum_60d', 0),
                    "volatility_20d": row.get('volatility_20d', 0),
                    "volatility_60d": row.get('volatility_60d', 0),
                    "risk_adj_momentum_20d": row.get('risk_adj_momentum_20d', 0),
                    "risk_adj_momentum_60d": row.get('risk_adj_momentum_60d', 0),
                    "rsi_14d": row.get('rsi_14d', 0),
                    "relative_strength_20d": row.get('relative_strength_20d', 0),
                    "max_drawdown": row.get('max_drawdown', 0)
                }
            }

            # 处理NaN值
            for key in ['returns', 'factors']:
                for subkey, value in heatmap_item[key].items():
                    if pd.isna(value):
                        heatmap_item[key][subkey] = None
                    else:
                        heatmap_item[key][subkey] = round(float(value), 2)

            heatmap_data.append(heatmap_item)

        return {
            "success": True,
            "data": heatmap_data,
            "update_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sector-rotation-signals")
async def get_sector_rotation_signals(
    strategy: str = Query("momentum", description="策略类型: momentum, low_volatility, risk_adjusted")
):
    """获取板块轮动信号"""
    try:
        # 获取行业ETF因子数据
        symbols = list(factor_analyzer.SECTOR_ETFS.keys())
        factors_df = factor_analyzer.calculate_all_factors(symbols)

        if factors_df.empty:
            return {
                "success": False,
                "message": "无法获取数据",
                "signals": []
            }

        # 生成轮动信号
        signals = factor_analyzer.get_sector_rotation_signals(factors_df, strategy)

        # 格式化信号数据
        formatted_signals = []
        for signal in signals:
            formatted_signals.append({
                "symbol": signal['symbol'],
                "name": signal['name'],
                "score": round(signal['score'], 2),
                "weight": signal['weight'],
                "signal_time": signal['signal_time'].isoformat()
            })

        return {
            "success": True,
            "strategy": strategy,
            "signals": formatted_signals,
            "update_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation-matrix")
async def get_correlation_matrix(
    symbols: Optional[List[str]] = Query(None, description="ETF代码列表"),
    period_days: int = Query(60, description="历史数据天数")
):
    """获取ETF相关性矩阵"""
    try:
        # 如果没有指定符号，使用默认的行业ETF
        if not symbols:
            symbols = list(factor_analyzer.SECTOR_ETFS.keys())[:10]  # 限制数量

        # 计算开始日期
        start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')

        # 获取价格数据
        prices = factor_analyzer.fetch_etf_data(symbols, start_date=start_date)

        if prices.empty:
            return {
                "success": False,
                "message": "无法获取数据",
                "data": []
            }

        # 计算相关性矩阵
        correlation_matrix = factor_analyzer.calculate_correlation_matrix(prices)

        # 格式化输出
        matrix_data = []
        for i, row_symbol in enumerate(correlation_matrix.index):
            for j, col_symbol in enumerate(correlation_matrix.columns):
                matrix_data.append({
                    "row": factor_analyzer.all_etfs.get(row_symbol, row_symbol),
                    "col": factor_analyzer.all_etfs.get(col_symbol, col_symbol),
                    "value": round(correlation_matrix.iloc[i, j], 2)
                })

        return {
            "success": True,
            "data": matrix_data,
            "symbols": [factor_analyzer.all_etfs.get(s, s) for s in symbols],
            "update_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-performers")
async def get_top_performers(
    factor: str = Query("momentum_20d", description="排序因子"),
    top_n: int = Query(10, description="返回数量"),
    etf_type: str = Query("all", description="ETF类型: all, sector, broad")
):
    """获取表现最佳的ETF"""
    try:
        # 根据类型选择ETF列表
        if etf_type == "sector":
            symbols = list(factor_analyzer.SECTOR_ETFS.keys())
        elif etf_type == "broad":
            symbols = list(factor_analyzer.BROAD_ETFS.keys())
        else:
            symbols = list(factor_analyzer.all_etfs.keys())

        # 计算因子数据
        factors_df = factor_analyzer.calculate_all_factors(symbols)

        if factors_df.empty:
            return {
                "success": False,
                "message": "无法获取数据",
                "data": []
            }

        # 获取表现最佳的ETF
        top_performers = factor_analyzer.get_top_performers(factors_df, factor, top_n)

        # 格式化输出
        performers_data = []
        for _, row in top_performers.iterrows():
            performer = {
                "rank": len(performers_data) + 1,
                "symbol": row['symbol'],
                "name": row['name'],
                "factor_value": round(row.get(factor, 0), 2),
                "returns": {
                    "1d": round(row.get('1d', 0), 2),
                    "5d": round(row.get('5d', 0), 2),
                    "20d": round(row.get('20d', 0), 2),
                    "60d": round(row.get('60d', 0), 2)
                }
            }
            performers_data.append(performer)

        return {
            "success": True,
            "factor": factor,
            "data": performers_data,
            "update_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-overview")
async def get_market_overview():
    """获取市场整体概览"""
    try:
        # 获取宽基指数数据
        broad_symbols = list(factor_analyzer.BROAD_ETFS.keys())
        broad_factors = factor_analyzer.calculate_all_factors(broad_symbols)

        # 获取行业ETF数据
        sector_symbols = list(factor_analyzer.SECTOR_ETFS.keys())
        sector_factors = factor_analyzer.calculate_all_factors(sector_symbols)

        if broad_factors.empty and sector_factors.empty:
            return {
                "success": False,
                "message": "无法获取数据"
            }

        # 计算市场统计
        overview = {
            "broad_indices": {
                "count": len(broad_factors),
                "avg_return_1d": round(broad_factors['1d'].mean(), 2) if not broad_factors.empty else 0,
                "avg_return_20d": round(broad_factors['20d'].mean(), 2) if not broad_factors.empty else 0,
                "best_performer": {
                    "name": broad_factors.iloc[0]['name'] if not broad_factors.empty else "",
                    "return_20d": round(broad_factors.iloc[0]['20d'], 2) if not broad_factors.empty else 0
                } if not broad_factors.empty else None
            },
            "sectors": {
                "count": len(sector_factors),
                "avg_return_1d": round(sector_factors['1d'].mean(), 2) if not sector_factors.empty else 0,
                "avg_return_20d": round(sector_factors['20d'].mean(), 2) if not sector_factors.empty else 0,
                "best_sector": {
                    "name": sector_factors.iloc[0]['name'] if not sector_factors.empty else "",
                    "return_20d": round(sector_factors.iloc[0]['20d'], 2) if not sector_factors.empty else 0
                } if not sector_factors.empty else None,
                "worst_sector": {
                    "name": sector_factors.iloc[-1]['name'] if not sector_factors.empty else "",
                    "return_20d": round(sector_factors.iloc[-1]['20d'], 2) if not sector_factors.empty else 0
                } if not sector_factors.empty else None
            },
            "market_breadth": {
                "advancing": len(sector_factors[sector_factors['1d'] > 0]) if not sector_factors.empty else 0,
                "declining": len(sector_factors[sector_factors['1d'] < 0]) if not sector_factors.empty else 0,
                "unchanged": len(sector_factors[sector_factors['1d'] == 0]) if not sector_factors.empty else 0
            },
            "volatility": {
                "avg_volatility_20d": round(sector_factors['volatility_20d'].mean(), 2) if not sector_factors.empty else 0,
                "max_volatility": {
                    "name": sector_factors.nlargest(1, 'volatility_20d').iloc[0]['name'] if not sector_factors.empty else "",
                    "value": round(sector_factors.nlargest(1, 'volatility_20d').iloc[0]['volatility_20d'], 2) if not sector_factors.empty else 0
                } if not sector_factors.empty else None
            },
            "update_time": datetime.now().isoformat()
        }

        return {
            "success": True,
            "data": overview
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))