"""
股票策略适配库服务
分析股票"性格"，管理股票与策略的匹配关系
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import yfinance as yf

from .database import (
    get_db, StockStrategyMatch, StockPersonality, Strategy, BacktestResult,
    get_stock_strategy_matches, get_stock_personality,
    create_or_update_stock_personality, create_stock_strategy_match,
    get_strategies, get_strategy_by_id
)
from .backtest_engine import calculate_technical_indicators


def analyze_stock_personality(ticker: str, period: str = "1y") -> Dict:
    """
    分析股票"性格"

    分析维度:
    1. 波动性 (volatility) - 使用历史波动率
    2. 趋势性 (trend) - 使用均值回归测试
    3. 动量 (momentum) - 使用动量指标
    4. Beta值 - 与市场的相关性
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            return {"error": f"无法获取 {ticker} 的数据"}

        # 计算技术指标
        df = calculate_technical_indicators(df)

        # 获取股票基本信息
        try:
            info = stock.info
            stock_name = info.get('shortName') or info.get('longName') or ticker
            sector = info.get('sector', '')
            industry = info.get('industry', '')
            beta = info.get('beta', None)
        except Exception:
            stock_name = ticker
            sector = ''
            industry = ''
            beta = None

        # 1. 计算波动性
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 年化波动率

        if volatility < 15:
            volatility_level = "low"
        elif volatility < 25:
            volatility_level = "medium"
        elif volatility < 40:
            volatility_level = "high"
        else:
            volatility_level = "extreme"

        # 平均日波动幅度
        avg_daily_range = ((df['High'] - df['Low']) / df['Close']).mean() * 100

        # 2. 分析趋势性
        # 使用价格与移动平均线的关系判断趋势
        if 'MA_50' in df.columns:
            price_vs_ma = (df['Close'].iloc[-1] - df['MA_50'].iloc[-1]) / df['MA_50'].iloc[-1]
            ma_slope = (df['MA_50'].iloc[-1] - df['MA_50'].iloc[-20]) / df['MA_50'].iloc[-20] if len(df) > 20 else 0

            if abs(ma_slope) > 0.05:
                trend_tendency = "trending"
                trend_strength = "strong" if abs(ma_slope) > 0.1 else "moderate"
            elif abs(ma_slope) < 0.02:
                trend_tendency = "mean_reverting"
                trend_strength = "weak"
            else:
                trend_tendency = "random"
                trend_strength = "moderate"
        else:
            trend_tendency = "unknown"
            trend_strength = "unknown"
            price_vs_ma = 0

        # 3. 分析动量
        if 'Momentum' in df.columns:
            momentum = df['Momentum'].iloc[-1]
            momentum_avg = df['Momentum'].mean()

            if momentum > 0.05:
                momentum_profile = "positive"
            elif momentum < -0.05:
                momentum_profile = "negative"
            else:
                momentum_profile = "neutral"

            # 动量持续性（使用自相关）
            momentum_series = df['Momentum'].dropna()
            if len(momentum_series) > 20:
                autocorr = momentum_series.autocorr(lag=5)
                if autocorr > 0.5:
                    momentum_persistence = "high"
                elif autocorr > 0.2:
                    momentum_persistence = "medium"
                else:
                    momentum_persistence = "low"
            else:
                momentum_persistence = "unknown"
        else:
            momentum_profile = "unknown"
            momentum_persistence = "unknown"
            momentum = 0

        # 4. 推荐策略类型
        recommended_strategies = []
        avoided_strategies = []

        # 基于波动性推荐
        if volatility_level in ["high", "extreme"]:
            recommended_strategies.extend(["momentum", "breakout"])
            avoided_strategies.extend(["mean_reversion"])
        elif volatility_level == "low":
            recommended_strategies.extend(["mean_reversion", "value"])
            avoided_strategies.extend(["momentum"])

        # 基于趋势性推荐
        if trend_tendency == "trending":
            recommended_strategies.extend(["trend_following", "macd"])
        elif trend_tendency == "mean_reverting":
            recommended_strategies.extend(["mean_reversion", "rsi"])

        # 基于动量推荐
        if momentum_profile == "positive" and momentum_persistence == "high":
            recommended_strategies.append("growth")
        elif momentum_profile == "negative":
            recommended_strategies.append("value")

        # 去重
        recommended_strategies = list(set(recommended_strategies))
        avoided_strategies = list(set(avoided_strategies))

        # 移除重复的（推荐和避免冲突的）
        avoided_strategies = [s for s in avoided_strategies if s not in recommended_strategies]

        # 生成分析备注
        analysis_notes = f"""
股票分析摘要 ({ticker}):
- 年化波动率: {volatility:.1f}% ({volatility_level})
- 平均日波动: {avg_daily_range:.2f}%
- 趋势倾向: {trend_tendency} (强度: {trend_strength})
- 动量状态: {momentum_profile} (持续性: {momentum_persistence})
- Beta值: {beta if beta else '未知'}

推荐策略: {', '.join(recommended_strategies) if recommended_strategies else '无特别推荐'}
不推荐策略: {', '.join(avoided_strategies) if avoided_strategies else '无'}
        """.strip()

        return {
            "ticker": ticker.upper(),
            "stock_name": stock_name,
            "sector": sector,
            "industry": industry,
            "volatility_level": volatility_level,
            "volatility_pct": round(volatility, 2),
            "avg_daily_range": round(avg_daily_range, 2),
            "beta": round(beta, 2) if beta else None,
            "trend_tendency": trend_tendency,
            "trend_strength": trend_strength,
            "momentum_profile": momentum_profile,
            "momentum_persistence": momentum_persistence,
            "recommended_strategies": recommended_strategies,
            "avoided_strategies": avoided_strategies,
            "analysis_notes": analysis_notes,
            "analyzed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {"error": str(e)}


def save_stock_personality(ticker: str, user_id: int = None) -> Dict:
    """分析并保存股票性格到数据库"""
    # 分析股票
    analysis = analyze_stock_personality(ticker)

    if "error" in analysis:
        return analysis

    # 保存到数据库
    with get_db() as db:
        personality = create_or_update_stock_personality(
            db=db,
            ticker=ticker,
            user_id=user_id,
            stock_name=analysis.get("stock_name"),
            volatility_level=analysis.get("volatility_level"),
            avg_daily_range=str(analysis.get("avg_daily_range")),
            beta=str(analysis.get("beta")) if analysis.get("beta") else None,
            trend_tendency=analysis.get("trend_tendency"),
            trend_strength=analysis.get("trend_strength"),
            momentum_profile=analysis.get("momentum_profile"),
            momentum_persistence=analysis.get("momentum_persistence"),
            recommended_strategies=analysis.get("recommended_strategies"),
            avoided_strategies=analysis.get("avoided_strategies"),
            analysis_notes=analysis.get("analysis_notes")
        )

        return {
            "success": True,
            "personality_id": personality.id,
            "analysis": analysis
        }


def get_stock_library(user_id: int) -> List[Dict]:
    """获取用户的股票策略适配库"""
    with get_db() as db:
        # 获取所有股票-策略匹配
        matches = get_stock_strategy_matches(db, user_id)

        # 按股票分组
        stocks = {}
        for match in matches:
            ticker = match.ticker
            if ticker not in stocks:
                # 尝试获取股票性格
                personality = get_stock_personality(db, ticker)
                stocks[ticker] = {
                    "ticker": ticker,
                    "stock_name": match.stock_name or ticker,
                    "personality": personality.to_dict() if personality else None,
                    "matched_strategies": []
                }

            stocks[ticker]["matched_strategies"].append({
                "strategy_id": match.strategy_id,
                "strategy_name": match.strategy_name,
                "strategy_type": match.strategy_type,
                "backtest_win_rate": float(match.backtest_win_rate) if match.backtest_win_rate else None,
                "backtest_return": float(match.backtest_return) if match.backtest_return else None,
                "simulation_win_rate": float(match.simulation_win_rate) if match.simulation_win_rate else None,
                "simulation_return": float(match.simulation_return) if match.simulation_return else None,
                "confidence_score": float(match.confidence_score) if match.confidence_score else None,
                "match_grade": match.match_grade,
                "is_recommended": match.is_recommended
            })

        return list(stocks.values())


def get_stock_detail(user_id: int, ticker: str) -> Dict:
    """获取单只股票的详细信息"""
    with get_db() as db:
        # 获取股票性格
        personality = get_stock_personality(db, ticker)

        # 获取匹配的策略
        matches = get_stock_strategy_matches(db, user_id, ticker)

        # 获取相关回测历史
        backtests = db.query(BacktestResult)\
            .filter(BacktestResult.user_id == user_id)\
            .filter(BacktestResult.ticker == ticker.upper())\
            .order_by(BacktestResult.created_at.desc())\
            .limit(20)\
            .all()

        return {
            "ticker": ticker.upper(),
            "personality": personality.to_dict() if personality else None,
            "matched_strategies": [
                {
                    "id": m.id,
                    "strategy_id": m.strategy_id,
                    "strategy_name": m.strategy_name,
                    "strategy_type": m.strategy_type,
                    "confidence_score": float(m.confidence_score) if m.confidence_score else None,
                    "match_grade": m.match_grade,
                    "backtest_win_rate": float(m.backtest_win_rate) if m.backtest_win_rate else None,
                    "simulation_win_rate": float(m.simulation_win_rate) if m.simulation_win_rate else None,
                    "is_recommended": m.is_recommended
                }
                for m in matches
            ],
            "backtest_history": [bt.to_dict() for bt in backtests]
        }


def add_strategy_match(
    user_id: int,
    ticker: str,
    strategy_id: int,
    backtest_id: int = None,
    simulation_id: int = None,
    metrics: Dict = None
) -> Dict:
    """添加股票-策略匹配"""
    with get_db() as db:
        # 获取策略信息
        strategy = get_strategy_by_id(db, strategy_id)
        if not strategy:
            return {"error": "策略不存在"}

        # 计算置信度和评分
        confidence = 50  # 默认
        grade = "C"

        if metrics:
            win_rate = metrics.get("win_rate", 0)
            total_return = metrics.get("total_return", 0)

            if win_rate >= 70 and total_return >= 10:
                confidence = 90
                grade = "A"
            elif win_rate >= 60 and total_return >= 5:
                confidence = 75
                grade = "B"
            elif win_rate >= 50 and total_return > 0:
                confidence = 60
                grade = "C"
            else:
                confidence = 40
                grade = "D"

        # 创建匹配记录
        match = create_stock_strategy_match(
            db=db,
            user_id=user_id,
            ticker=ticker,
            strategy_id=strategy_id,
            strategy_name=strategy.name,
            strategy_type=strategy.strategy_type,
            backtest_id=backtest_id,
            simulation_id=simulation_id,
            backtest_win_rate=metrics.get("win_rate") if metrics else None,
            backtest_return=metrics.get("total_return") if metrics else None,
            confidence_score=confidence,
            match_grade=grade
        )

        return {
            "success": True,
            "match_id": match.id,
            "confidence_score": confidence,
            "grade": grade
        }


def remove_strategy_match(user_id: int, ticker: str, strategy_id: int) -> Dict:
    """移除股票-策略匹配"""
    with get_db() as db:
        match = db.query(StockStrategyMatch)\
            .filter(StockStrategyMatch.user_id == user_id)\
            .filter(StockStrategyMatch.ticker == ticker.upper())\
            .filter(StockStrategyMatch.strategy_id == strategy_id)\
            .first()

        if not match:
            return {"error": "未找到匹配记录"}

        # 软删除
        match.is_active = False
        match.updated_at = datetime.utcnow()
        db.commit()

        return {"success": True, "message": "已移除策略匹配"}


def get_best_strategies_for_stock(ticker: str, user_id: int = None, limit: int = 5) -> List[Dict]:
    """获取最适合该股票的策略（基于历史数据和性格分析）"""
    # 先分析股票性格
    analysis = analyze_stock_personality(ticker)

    if "error" in analysis:
        return []

    recommended = analysis.get("recommended_strategies", [])
    avoided = analysis.get("avoided_strategies", [])

    with get_db() as db:
        # 获取所有策略
        strategies = get_strategies(db, public_only=True, active_only=True)

        # 根据推荐排序
        result = []
        for strategy in strategies:
            score = 50  # 基础分

            if strategy.strategy_type in recommended:
                score += 30
            if strategy.strategy_type in avoided:
                score -= 30

            # 如果有历史最佳指标，加分
            if strategy.best_win_rate:
                score += min(20, float(strategy.best_win_rate) / 5)
            if strategy.best_total_return:
                score += min(10, float(strategy.best_total_return) / 10)

            result.append({
                "strategy_id": strategy.id,
                "strategy_name": strategy.name,
                "strategy_type": strategy.strategy_type,
                "category": strategy.category,
                "score": score,
                "is_recommended": strategy.strategy_type in recommended,
                "is_avoided": strategy.strategy_type in avoided,
                "best_win_rate": float(strategy.best_win_rate) if strategy.best_win_rate else None,
                "best_total_return": float(strategy.best_total_return) if strategy.best_total_return else None
            })

        # 排序
        result.sort(key=lambda x: x["score"], reverse=True)

        return result[:limit]


def get_all_stock_personalities(user_id: int = None) -> List[Dict]:
    """获取所有股票性格记录"""
    with get_db() as db:
        query = db.query(StockPersonality)
        if user_id:
            query = query.filter(
                (StockPersonality.user_id == user_id) |
                (StockPersonality.user_id == None)
            )
        personalities = query.all()
        return [p.to_dict() for p in personalities]
