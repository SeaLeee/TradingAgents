"""
批量回测服务
对某个股票执行所有策略的回测，按胜率排序，区分按天/按月交易
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .database import (
    get_db, BatchBacktestJob, Strategy,
    get_strategies, get_batch_backtest_job, update_batch_backtest_job,
    create_batch_backtest_job
)
from .backtest_engine import run_strategy_backtest, STRATEGY_REGISTRY


def start_batch_backtest(
    user_id: int,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    trading_frequency: str = "daily"
) -> BatchBacktestJob:
    """创建并返回批量回测任务"""
    with get_db() as db:
        job = create_batch_backtest_job(
            db=db,
            user_id=user_id,
            ticker=ticker.upper(),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trading_frequency=trading_frequency
        )
        return job


def execute_batch_backtest(job_id: int) -> Dict:
    """
    执行批量回测（后台任务）

    流程:
    1. 获取所有活跃策略
    2. 对每个策略执行回测
    3. 收集结果并排序
    4. 选出最佳策略
    5. 更新任务状态
    """
    with get_db() as db:
        job = get_batch_backtest_job(db, job_id)
        if not job:
            return {"error": "Job not found"}

        # 更新状态为运行中
        update_batch_backtest_job(db, job, status="running", progress=0)

        try:
            # 获取所有活跃策略
            strategies = get_strategies(db, public_only=True, active_only=True)
            total_strategies = len(strategies)
            update_batch_backtest_job(db, job, total_strategies=total_strategies)

            all_results = []
            completed = 0

            for strategy in strategies:
                try:
                    # 获取策略参数
                    params = json.loads(strategy.default_params) if strategy.default_params else {}

                    # 执行回测
                    result = run_strategy_backtest(
                        strategy_type=strategy.strategy_type,
                        ticker=job.ticker,
                        start_date=job.start_date,
                        end_date=job.end_date,
                        initial_capital=float(job.initial_capital),
                        params=params
                    )

                    # 如果是按月交易，需要调整结果
                    if job.trading_frequency == "monthly":
                        # 重新运行月度回测
                        result = run_monthly_backtest(
                            strategy_type=strategy.strategy_type,
                            ticker=job.ticker,
                            start_date=job.start_date,
                            end_date=job.end_date,
                            initial_capital=float(job.initial_capital),
                            params=params
                        )

                    # 收集结果
                    strategy_result = {
                        "strategy_id": strategy.id,
                        "strategy_name": strategy.name,
                        "strategy_type": strategy.strategy_type,
                        "win_rate": result.get("win_rate", 0),
                        "total_return": result.get("total_return", 0),
                        "annualized_return": result.get("annualized_return", 0),
                        "sharpe_ratio": result.get("sharpe_ratio", 0),
                        "max_drawdown": result.get("max_drawdown", 0),
                        "total_trades": result.get("total_trades", 0),
                        "winning_trades": result.get("winning_trades", 0),
                        "losing_trades": result.get("losing_trades", 0),
                        "profit_factor": result.get("profit_factor", 0),
                        "final_value": result.get("final_value", float(job.initial_capital)),
                        "error": result.get("error")
                    }
                    all_results.append(strategy_result)

                except Exception as e:
                    all_results.append({
                        "strategy_id": strategy.id,
                        "strategy_name": strategy.name,
                        "strategy_type": strategy.strategy_type,
                        "error": str(e),
                        "win_rate": 0,
                        "total_return": 0
                    })

                completed += 1
                progress = int((completed / total_strategies) * 100)
                update_batch_backtest_job(
                    db, job,
                    completed_strategies=completed,
                    progress=progress
                )

            # 按胜率排序
            sorted_results = rank_strategies_by_winrate(all_results)

            # 找出最佳策略（过滤掉有错误的）
            valid_results = [r for r in sorted_results if not r.get("error")]
            best = valid_results[0] if valid_results else None

            # 更新任务完成状态
            update_data = {
                "status": "completed",
                "progress": 100,
                "all_results": json.dumps(sorted_results),
                "completed_at": datetime.utcnow()
            }

            if best:
                update_data.update({
                    "best_strategy_id": best["strategy_id"],
                    "best_strategy_name": best["strategy_name"],
                    "best_win_rate": str(best["win_rate"]),
                    "best_total_return": str(best["total_return"]),
                    "best_sharpe_ratio": str(best["sharpe_ratio"])
                })

            update_batch_backtest_job(db, job, **update_data)

            return {
                "status": "completed",
                "total_strategies": total_strategies,
                "best_strategy": best,
                "all_results": sorted_results
            }

        except Exception as e:
            update_batch_backtest_job(
                db, job,
                status="failed",
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            return {"error": str(e)}


def run_monthly_backtest(
    strategy_type: str,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    params: Dict = None
) -> Dict:
    """
    执行月度回测

    将日度数据重采样为月度数据后执行回测
    这样可以模拟每月只交易一次的情况
    """
    import yfinance as yf
    from .backtest_engine import (
        STRATEGY_REGISTRY, STRATEGY_ALIASES, apply_default_params
    )

    try:
        # 下载日度数据
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            return {"error": f"No data available for {ticker}"}

        # 重采样为月度数据（月末）
        df_monthly = df.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(df_monthly) < 5:
            return {"error": "Insufficient data for monthly resampling (need at least 5 months)"}

        # 获取策略
        resolved_type = STRATEGY_ALIASES.get(strategy_type, strategy_type)
        strategy = STRATEGY_REGISTRY.get(resolved_type)
        if not strategy:
            return {"error": f"Unknown strategy type: {strategy_type}"}

        # 调整参数为月度（窗口期缩小）
        adjusted_params = params.copy() if params else {}

        # 月度数据窗口调整映射
        window_adjustments = {
            'short_window': lambda x: max(2, x // 10),  # 20 -> 2
            'long_window': lambda x: max(3, x // 10),   # 50 -> 5
            'ma_period': lambda x: max(2, x // 10),
            'lookback_period': lambda x: max(2, x // 10),
            'period': lambda x: max(2, x // 10),
            'rsi_period': lambda x: max(3, x // 3),     # 14 -> 5
            'fast_period': lambda x: max(2, x // 3),    # 12 -> 4
            'slow_period': lambda x: max(3, x // 3),    # 26 -> 9
            'signal_period': lambda x: max(2, x // 3),  # 9 -> 3
        }

        for key, adjuster in window_adjustments.items():
            if key in adjusted_params:
                adjusted_params[key] = adjuster(adjusted_params[key])

        # 合并默认参数
        merged_params = apply_default_params(adjusted_params, strategy["defaults"])

        # 再次调整合并后的参数（针对默认值）
        for key, adjuster in window_adjustments.items():
            if key in merged_params and key not in adjusted_params:
                merged_params[key] = adjuster(merged_params[key])

        # 直接使用月度数据执行策略处理函数
        handler = strategy["handler"]
        result = handler(df_monthly, initial_capital, **merged_params)

        # 标记为月度回测
        result["trading_frequency"] = "monthly"
        result["data_points"] = len(df_monthly)

        return result

    except Exception as e:
        return {"error": str(e)}


def rank_strategies_by_winrate(results: List[Dict]) -> List[Dict]:
    """按胜率排序策略结果（降序）"""
    return sorted(
        results,
        key=lambda x: (
            0 if x.get("error") else 1,  # 有错误的排最后
            x.get('win_rate', 0),
            x.get('total_return', 0),
            x.get('sharpe_ratio', 0)
        ),
        reverse=True
    )


def rank_strategies_by_return(results: List[Dict]) -> List[Dict]:
    """按总收益率排序策略结果（降序）"""
    return sorted(
        results,
        key=lambda x: (
            0 if x.get("error") else 1,
            x.get('total_return', 0),
            x.get('win_rate', 0)
        ),
        reverse=True
    )


def rank_strategies_by_sharpe(results: List[Dict]) -> List[Dict]:
    """按夏普比率排序策略结果（降序）"""
    return sorted(
        results,
        key=lambda x: (
            0 if x.get("error") else 1,
            x.get('sharpe_ratio', 0),
            x.get('total_return', 0)
        ),
        reverse=True
    )


def get_batch_backtest_summary(job: BatchBacktestJob) -> Dict:
    """获取批量回测任务的摘要"""
    if not job.all_results:
        return {
            "status": job.status,
            "progress": job.progress,
            "message": "No results yet"
        }

    results = json.loads(job.all_results)
    valid_results = [r for r in results if not r.get("error")]
    error_results = [r for r in results if r.get("error")]

    # 计算统计
    profitable_count = sum(1 for r in valid_results if r.get("total_return", 0) > 0)

    # 获取前3名
    top3 = valid_results[:3] if len(valid_results) >= 3 else valid_results

    return {
        "status": job.status,
        "progress": job.progress,
        "ticker": job.ticker,
        "trading_frequency": job.trading_frequency,
        "total_strategies_tested": len(results),
        "successful_tests": len(valid_results),
        "failed_tests": len(error_results),
        "profitable_strategies": profitable_count,
        "best_strategy": {
            "id": job.best_strategy_id,
            "name": job.best_strategy_name,
            "win_rate": float(job.best_win_rate) if job.best_win_rate else None,
            "total_return": float(job.best_total_return) if job.best_total_return else None,
            "sharpe_ratio": float(job.best_sharpe_ratio) if job.best_sharpe_ratio else None
        } if job.best_strategy_id else None,
        "top3_strategies": [
            {
                "id": r["strategy_id"],
                "name": r["strategy_name"],
                "win_rate": r["win_rate"],
                "total_return": r["total_return"],
                "sharpe_ratio": r["sharpe_ratio"]
            }
            for r in top3
        ],
        "all_results": results
    }
