"""
增强版回测API路由
提供详细的回测统计数据和图表数据
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from app.modules.enhanced_backtest import EnhancedBacktest, BacktestMetrics
from app.modules.etf_factors import ETFFactorAnalyzer

router = APIRouter(
    prefix="/api/backtest",
    tags=["backtest"]
)

class BacktestRequest(BaseModel):
    """回测请求模型"""
    strategy_type: str  # 策略类型
    symbols: List[str]  # 股票/ETF代码列表
    start_date: str  # 开始日期
    end_date: str  # 结束日期
    initial_capital: float = 1000000  # 初始资金
    benchmark: Optional[str] = "510300.SS"  # 基准指数
    parameters: Optional[Dict[str, Any]] = {}  # 策略参数


class StrategyManager:
    """策略管理器"""

    @staticmethod
    def sector_rotation_strategy(data: pd.DataFrame, **params) -> List[Dict]:
        """板块轮动策略"""
        signals = []
        lookback = params.get('lookback', 20)
        top_n = params.get('top_n', 3)
        rebalance_freq = params.get('rebalance_freq', 20)

        # 检查是否需要调仓
        if len(data) % rebalance_freq != 0:
            return signals

        # 计算动量
        returns = data.pct_change(lookback).iloc[-1]
        top_performers = returns.nlargest(top_n)

        # 生成买卖信号
        current_date = data.index[-1]
        weight = 1.0 / top_n

        # 卖出不在前N的持仓
        for symbol in data.columns:
            if symbol not in top_performers.index:
                signals.append({
                    'date': current_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0  # 全部卖出
                })

        # 买入前N的ETF
        for symbol in top_performers.index:
            signals.append({
                'date': current_date,
                'symbol': symbol,
                'action': 'BUY',
                'weight': weight
            })

        return signals

    @staticmethod
    def mean_reversion_strategy(data: pd.DataFrame, **params) -> List[Dict]:
        """均值回归策略"""
        signals = []
        lookback = params.get('lookback', 20)
        z_threshold = params.get('z_threshold', 2)

        if len(data) < lookback:
            return signals

        # 计算Z-score
        for symbol in data.columns:
            prices = data[symbol]
            mean = prices.rolling(window=lookback).mean().iloc[-1]
            std = prices.rolling(window=lookback).std().iloc[-1]
            current_price = prices.iloc[-1]

            if std > 0:
                z_score = (current_price - mean) / std

                if z_score < -z_threshold:  # 超卖，买入信号
                    signals.append({
                        'date': data.index[-1],
                        'symbol': symbol,
                        'action': 'BUY',
                        'weight': 0.1
                    })
                elif z_score > z_threshold:  # 超买，卖出信号
                    signals.append({
                        'date': data.index[-1],
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': 0
                    })

        return signals

    @staticmethod
    def momentum_strategy(data: pd.DataFrame, **params) -> List[Dict]:
        """动量策略"""
        signals = []
        short_window = params.get('short_window', 10)
        long_window = params.get('long_window', 30)

        if len(data) < long_window:
            return signals

        for symbol in data.columns:
            prices = data[symbol]
            short_ma = prices.rolling(window=short_window).mean().iloc[-1]
            long_ma = prices.rolling(window=long_window).mean().iloc[-1]
            prev_short_ma = prices.rolling(window=short_window).mean().iloc[-2]
            prev_long_ma = prices.rolling(window=long_window).mean().iloc[-2]

            # 金叉买入
            if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                signals.append({
                    'date': data.index[-1],
                    'symbol': symbol,
                    'action': 'BUY',
                    'weight': 1.0 / len(data.columns)
                })
            # 死叉卖出
            elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                signals.append({
                    'date': data.index[-1],
                    'symbol': symbol,
                    'action': 'SELL',
                    'quantity': 0
                })

        return signals


@router.post("/run")
async def run_backtest(request: BacktestRequest):
    """运行回测"""
    try:
        # 获取历史数据
        data = pd.DataFrame()
        for symbol in request.symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=request.start_date, end=request.end_date)
            if not hist.empty:
                data[symbol] = hist['Close']

        if data.empty:
            raise HTTPException(status_code=400, detail="无法获取历史数据")

        # 获取基准数据
        benchmark_data = None
        if request.benchmark:
            ticker = yf.Ticker(request.benchmark)
            bench_hist = ticker.history(start=request.start_date, end=request.end_date)
            if not bench_hist.empty:
                benchmark_data = pd.DataFrame({'benchmark': bench_hist['Close']})

        # 选择策略
        strategy_manager = StrategyManager()
        if request.strategy_type == "sector_rotation":
            strategy_func = strategy_manager.sector_rotation_strategy
        elif request.strategy_type == "mean_reversion":
            strategy_func = strategy_manager.mean_reversion_strategy
        elif request.strategy_type == "momentum":
            strategy_func = strategy_manager.momentum_strategy
        else:
            raise HTTPException(status_code=400, detail="不支持的策略类型")

        # 运行回测
        backtest = EnhancedBacktest(initial_capital=request.initial_capital)
        results = backtest.run_backtest(
            strategy_func,
            data,
            benchmark_data,
            **request.parameters
        )

        # 格式化输出
        metrics = results['metrics']
        formatted_metrics = {
            "basic_metrics": {
                "total_return": round(metrics.total_return, 2),
                "annual_return": round(metrics.annual_return, 2),
                "sharpe_ratio": round(metrics.sharpe_ratio, 2),
                "max_drawdown": round(metrics.max_drawdown, 2),
                "volatility": round(metrics.volatility, 2),
                "calmar_ratio": round(metrics.calmar_ratio, 2),
            },
            "risk_metrics": {
                "sortino_ratio": round(metrics.sortino_ratio, 2),
                "downside_deviation": round(metrics.downside_deviation, 2),
                "var_95": round(metrics.var_95, 2),
                "cvar_95": round(metrics.cvar_95, 2),
                "beta": round(metrics.beta, 2),
                "alpha": round(metrics.alpha, 2),
            },
            "trading_metrics": {
                "total_trades": metrics.total_trades,
                "win_rate": round(metrics.win_rate, 2),
                "profit_loss_ratio": round(metrics.profit_loss_ratio, 2),
                "winning_trades": metrics.winning_trades,
                "losing_trades": metrics.losing_trades,
                "avg_trade_return": round(metrics.avg_trade_return, 2),
                "max_consecutive_wins": metrics.max_consecutive_wins,
                "max_consecutive_losses": metrics.max_consecutive_losses,
            },
            "performance_metrics": {
                "best_day": round(metrics.best_day, 2),
                "worst_day": round(metrics.worst_day, 2),
                "best_month": round(metrics.best_month, 2),
                "worst_month": round(metrics.worst_month, 2),
                "positive_months": metrics.positive_months,
                "negative_months": metrics.negative_months,
            },
            "benchmark_comparison": {
                "excess_return": round(metrics.excess_return, 2),
                "tracking_error": round(metrics.tracking_error, 2),
                "information_ratio": round(metrics.information_ratio, 2),
            }
        }

        # 准备图表数据
        portfolio_values = [
            {
                "date": v['date'].isoformat() if hasattr(v['date'], 'isoformat') else str(v['date']),
                "value": round(v['value'], 2),
                "cash": round(v['cash'], 2),
                "positions_value": round(v['positions_value'], 2)
            }
            for v in results['portfolio_value']
        ]

        drawdown_data = [
            {
                "date": portfolio_values[i]['date'],
                "drawdown": round(dd, 2)
            }
            for i, dd in enumerate(results['drawdown'])
        ]

        # 月度收益
        monthly_returns = [
            {
                "month": month.isoformat() if hasattr(month, 'isoformat') else str(month),
                "return": round(ret, 2)
            }
            for month, ret in results['monthly_returns'].items()
        ]

        # 交易记录
        trades = [
            {
                "date": t['date'].isoformat() if hasattr(t['date'], 'isoformat') else str(t['date']),
                "symbol": t['symbol'],
                "action": t['action'],
                "quantity": t.get('quantity', 0),
                "price": round(t.get('price', 0), 2),
                "amount": round(t.get('cost', t.get('proceeds', 0)), 2)
            }
            for t in backtest.trades
        ]

        return {
            "success": True,
            "metrics": formatted_metrics,
            "portfolio_values": portfolio_values,
            "drawdown": drawdown_data,
            "monthly_returns": monthly_returns,
            "trades": trades[:50],  # 限制返回前50条
            "final_portfolio_value": round(portfolio_values[-1]['value'], 2) if portfolio_values else 0,
            "total_return_amount": round(portfolio_values[-1]['value'] - request.initial_capital, 2) if portfolio_values else 0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_available_strategies():
    """获取可用的策略列表"""
    strategies = [
        {
            "id": "sector_rotation",
            "name": "板块轮动策略",
            "description": "基于动量选择表现最好的板块ETF",
            "risk_level": "moderate",
            "parameters": [
                {"name": "lookback", "type": "int", "default": 20, "description": "动量计算周期"},
                {"name": "top_n", "type": "int", "default": 3, "description": "选择前N个板块"},
                {"name": "rebalance_freq", "type": "int", "default": 20, "description": "调仓频率（天）"}
            ]
        },
        {
            "id": "mean_reversion",
            "name": "均值回归策略",
            "description": "利用价格偏离均值的回归特性进行交易",
            "risk_level": "high",
            "parameters": [
                {"name": "lookback", "type": "int", "default": 20, "description": "均值计算周期"},
                {"name": "z_threshold", "type": "float", "default": 2, "description": "Z-score阈值"}
            ]
        },
        {
            "id": "momentum",
            "name": "双均线动量策略",
            "description": "基于短期和长期均线交叉的动量策略",
            "risk_level": "moderate",
            "parameters": [
                {"name": "short_window", "type": "int", "default": 10, "description": "短期均线周期"},
                {"name": "long_window", "type": "int", "default": 30, "description": "长期均线周期"}
            ]
        }
    ]

    return {
        "success": True,
        "strategies": strategies
    }


@router.get("/portfolio-optimization")
async def optimize_portfolio(
    symbols: List[str] = None,
    optimization_method: str = "max_sharpe"
):
    """组合优化"""
    try:
        if not symbols:
            # 使用默认的ETF列表
            factor_analyzer = ETFFactorAnalyzer()
            symbols = list(factor_analyzer.BROAD_ETFS.keys())[:5]

        # 获取历史数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        data = pd.DataFrame()
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                data[symbol] = hist['Close']

        if data.empty:
            raise HTTPException(status_code=400, detail="无法获取历史数据")

        # 计算收益率
        returns = data.pct_change().dropna()

        # 计算预期收益和协方差矩阵
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # 简单的等权重分配（实际应用中应使用更复杂的优化算法）
        n_assets = len(symbols)
        weights = np.array([1/n_assets] * n_assets)

        # 计算组合指标
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0

        # 准备输出
        optimization_result = {
            "weights": [
                {
                    "symbol": symbol,
                    "weight": round(weight * 100, 2)
                }
                for symbol, weight in zip(symbols, weights)
            ],
            "expected_return": round(portfolio_return * 100, 2),
            "expected_volatility": round(portfolio_std * 100, 2),
            "expected_sharpe": round(portfolio_sharpe, 2),
            "correlation_matrix": cov_matrix.corr().round(2).to_dict()
        }

        return {
            "success": True,
            "optimization_method": optimization_method,
            "result": optimization_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))