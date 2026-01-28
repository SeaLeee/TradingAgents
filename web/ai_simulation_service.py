"""
AI实时模拟交易服务
对回测好的策略进行模拟交易，每日自动检查信号并执行交易
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf

from .database import (
    get_db, AISimulationSession, Strategy, StockStrategyMatch,
    get_ai_simulation_session, update_ai_simulation_session,
    create_ai_simulation_session, get_strategy_by_id,
    create_stock_strategy_match, get_active_ai_simulation_sessions
)
from .signal_generator import generate_signal


def start_simulation(
    user_id: int,
    ticker: str,
    strategy_id: int,
    duration_days: int = 14,
    initial_capital: float = 100000,
    check_interval: str = "daily"
) -> AISimulationSession:
    """启动模拟交易会话"""
    with get_db() as db:
        # 验证策略存在
        strategy = get_strategy_by_id(db, strategy_id)
        if not strategy:
            raise ValueError(f"Strategy {strategy_id} not found")

        session = create_ai_simulation_session(
            db=db,
            user_id=user_id,
            ticker=ticker.upper(),
            strategy_id=strategy_id,
            duration_days=duration_days,
            initial_capital=initial_capital,
            check_interval=check_interval
        )
        return session


def check_signal_and_trade(session_id: int) -> Dict:
    """
    检查信号并执行交易

    流程:
    1. 获取会话信息
    2. 获取当前价格
    3. 生成策略信号
    4. 根据信号和当前持仓决定操作
    5. 执行交易（模拟）
    6. 更新会话状态
    7. 记录交易历史
    """
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        if session.status != "active":
            return {"error": f"Session is not active (status: {session.status})"}

        # 获取策略信息
        strategy = get_strategy_by_id(db, session.strategy_id)
        if not strategy:
            return {"error": "Strategy not found"}

        # 获取当前价格
        try:
            stock = yf.Ticker(session.ticker)
            current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
            if not current_price:
                hist = stock.history(period='1d')
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                else:
                    return {"error": "Cannot get current price"}
        except Exception as e:
            return {"error": f"Failed to get price: {str(e)}"}

        # 获取策略参数
        params = json.loads(strategy.default_params) if strategy.default_params else {}

        # 生成信号
        signal_result = generate_signal(
            ticker=session.ticker,
            strategy_type=strategy.strategy_type,
            params=params
        )

        signal = signal_result.get("signal", "hold")
        reason = signal_result.get("reason", "")

        # 记录每日检查
        daily_check = {
            "date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "price": current_price,
            "signal": signal,
            "reason": reason,
            "position_before": session.current_position
        }

        # 根据信号和当前持仓决定操作
        action_taken = None
        trade_record = None

        if signal == "buy" and session.current_position == "cash":
            # 买入
            trade_result = execute_buy(session, current_price, db)
            action_taken = "买入"
            trade_record = trade_result

        elif signal == "sell" and session.current_position == "long":
            # 卖出
            trade_result = execute_sell(session, current_price, db)
            action_taken = "卖出"
            trade_record = trade_result

        elif signal == "hold":
            action_taken = "持仓观望"

        else:
            # 信号与持仓状态冲突，不执行操作
            action_taken = "信号与持仓不匹配，跳过"

        daily_check["action_taken"] = action_taken
        daily_check["position_after"] = session.current_position

        # 更新每日检查记录
        daily_checks = json.loads(session.daily_checks) if session.daily_checks else []
        daily_checks.append(daily_check)

        # 更新当前价值
        current_value = calculate_current_value(session, current_price)

        # 增加天数
        new_day = session.current_day + 1

        # 检查是否到期
        if new_day >= session.duration_days:
            # 如果还有持仓，强制平仓
            if session.current_position == "long":
                execute_sell(session, current_price, db)

            # 完成模拟
            complete_result = complete_simulation_internal(session, db)

            update_ai_simulation_session(
                db, session,
                current_day=new_day,
                current_value=str(current_value),
                daily_checks=json.dumps(daily_checks),
                last_check_at=datetime.utcnow(),
                status="completed",
                end_date=datetime.utcnow().strftime('%Y-%m-%d')
            )

            return {
                "status": "completed",
                "signal": signal,
                "action_taken": action_taken,
                "current_price": current_price,
                "current_value": current_value,
                "summary": complete_result
            }

        # 更新会话
        update_ai_simulation_session(
            db, session,
            current_day=new_day,
            current_value=str(current_value),
            daily_checks=json.dumps(daily_checks),
            last_check_at=datetime.utcnow()
        )

        return {
            "status": "active",
            "signal": signal,
            "reason": reason,
            "action_taken": action_taken,
            "current_price": current_price,
            "current_value": current_value,
            "current_position": session.current_position,
            "day": new_day,
            "days_remaining": session.duration_days - new_day,
            "trade_record": trade_record
        }


def execute_buy(session: AISimulationSession, price: float, db) -> Dict:
    """执行模拟买入"""
    # 计算可买入股数（用全部资金）
    available_cash = float(session.current_value or session.initial_capital)
    shares = int(available_cash / price)

    if shares <= 0:
        return {"error": "Insufficient funds"}

    cost = shares * price

    # 创建交易记录
    trade_record = {
        "type": "buy",
        "date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        "price": price,
        "shares": shares,
        "cost": cost
    }

    # 更新交易历史
    trade_history = json.loads(session.trade_history) if session.trade_history else []
    trade_history.append(trade_record)

    # 更新会话
    update_ai_simulation_session(
        db, session,
        current_position="long",
        shares=shares,
        entry_price=str(price),
        entry_date=datetime.utcnow().strftime('%Y-%m-%d'),
        trade_history=json.dumps(trade_history),
        total_trades=session.total_trades + 1
    )

    return trade_record


def execute_sell(session: AISimulationSession, price: float, db) -> Dict:
    """执行模拟卖出"""
    if session.shares <= 0:
        return {"error": "No shares to sell"}

    shares = session.shares
    entry_price = float(session.entry_price)
    proceeds = shares * price
    cost = shares * entry_price
    pnl = proceeds - cost
    pnl_percent = (pnl / cost) * 100 if cost > 0 else 0

    # 判断是否盈利
    is_winning = pnl > 0

    # 创建交易记录
    trade_record = {
        "type": "sell",
        "date": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        "price": price,
        "shares": shares,
        "proceeds": proceeds,
        "entry_price": entry_price,
        "pnl": pnl,
        "pnl_percent": pnl_percent,
        "is_winning": is_winning
    }

    # 更新交易历史
    trade_history = json.loads(session.trade_history) if session.trade_history else []
    trade_history.append(trade_record)

    # 更新统计
    new_winning = session.winning_trades + (1 if is_winning else 0)
    new_losing = session.losing_trades + (0 if is_winning else 1)
    new_total_pnl = float(session.total_pnl or 0) + pnl

    # 更新会话
    update_ai_simulation_session(
        db, session,
        current_position="cash",
        shares=0,
        entry_price=None,
        entry_date=None,
        current_value=str(proceeds),
        trade_history=json.dumps(trade_history),
        total_trades=session.total_trades + 1,
        winning_trades=new_winning,
        losing_trades=new_losing,
        total_pnl=str(new_total_pnl)
    )

    return trade_record


def calculate_current_value(session: AISimulationSession, current_price: float) -> float:
    """计算当前价值"""
    if session.current_position == "long" and session.shares > 0:
        return session.shares * current_price
    else:
        return float(session.current_value or session.initial_capital)


def calculate_session_statistics(session: AISimulationSession) -> Dict:
    """计算会话统计数据"""
    initial = float(session.initial_capital)
    current = float(session.current_value or initial)
    total_pnl = float(session.total_pnl or 0)
    total_trades = session.total_trades
    winning_trades = session.winning_trades
    losing_trades = session.losing_trades

    # 计算胜率（仅计算平仓交易）
    closed_trades = winning_trades + losing_trades
    win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0

    # 计算收益率
    total_return = ((current - initial) / initial * 100) if initial > 0 else 0

    return {
        "initial_capital": initial,
        "current_value": current,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "total_trades": total_trades,
        "closed_trades": closed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "days_elapsed": session.current_day,
        "days_total": session.duration_days
    }


def complete_simulation_internal(session: AISimulationSession, db) -> Dict:
    """
    完成模拟交易（内部函数）

    流程:
    1. 计算最终统计
    2. 评估是否合格加入适配库
    3. 如果合格，自动创建StockStrategyMatch记录
    """
    stats = calculate_session_statistics(session)

    # 评估是否合格
    qualified = should_qualify_for_library(stats)

    if qualified:
        # 获取策略信息
        strategy = get_strategy_by_id(db, session.strategy_id)

        if strategy:
            # 计算置信度和评分
            confidence, grade = calculate_confidence_and_grade(stats)

            # 创建股票-策略匹配记录
            create_stock_strategy_match(
                db=db,
                user_id=session.user_id,
                ticker=session.ticker,
                strategy_id=session.strategy_id,
                strategy_name=strategy.name,
                strategy_type=strategy.strategy_type,
                simulation_id=session.id,
                simulation_win_rate=stats["win_rate"],
                simulation_return=stats["total_return"],
                confidence_score=confidence,
                match_grade=grade
            )

        # 标记为合格
        update_ai_simulation_session(
            db, session,
            qualified_for_library=True,
            qualification_date=datetime.utcnow()
        )

    return {
        "statistics": stats,
        "qualified_for_library": qualified,
        "recommendation": "该策略在模拟期间表现良好，已加入股票策略适配库" if qualified else "该策略在模拟期间表现一般，未加入适配库"
    }


def complete_simulation(session_id: int) -> Dict:
    """完成模拟交易（外部调用）"""
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        # 获取当前价格并强制平仓（如果有持仓）
        if session.current_position == "long" and session.shares > 0:
            try:
                stock = yf.Ticker(session.ticker)
                current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
                if current_price:
                    execute_sell(session, current_price, db)
            except Exception:
                pass

        result = complete_simulation_internal(session, db)

        update_ai_simulation_session(
            db, session,
            status="completed",
            end_date=datetime.utcnow().strftime('%Y-%m-%d')
        )

        return result


def stop_simulation(session_id: int) -> Dict:
    """停止模拟交易"""
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        # 强制平仓（如果有持仓）
        if session.current_position == "long" and session.shares > 0:
            try:
                stock = yf.Ticker(session.ticker)
                current_price = stock.info.get('regularMarketPrice') or stock.info.get('currentPrice')
                if current_price:
                    execute_sell(session, current_price, db)
            except Exception:
                pass

        update_ai_simulation_session(
            db, session,
            status="stopped",
            end_date=datetime.utcnow().strftime('%Y-%m-%d')
        )

        return {
            "status": "stopped",
            "statistics": calculate_session_statistics(session)
        }


def pause_simulation(session_id: int) -> Dict:
    """暂停模拟交易"""
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        update_ai_simulation_session(db, session, status="paused")
        return {"status": "paused"}


def resume_simulation(session_id: int) -> Dict:
    """恢复模拟交易"""
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        if session.status != "paused":
            return {"error": "Session is not paused"}

        update_ai_simulation_session(db, session, status="active")
        return {"status": "active"}


def should_qualify_for_library(stats: Dict) -> bool:
    """
    判断是否合格加入股票策略适配库

    条件（可配置）:
    - 胜率 >= 50%
    - 收益率 > 0
    - 至少完成2笔完整交易（平仓）
    """
    win_rate = stats.get("win_rate", 0)
    total_return = stats.get("total_return", 0)
    closed_trades = stats.get("closed_trades", 0)

    return (
        win_rate >= 50 and
        total_return > 0 and
        closed_trades >= 2
    )


def calculate_confidence_and_grade(stats: Dict) -> tuple:
    """
    计算置信度和评分

    返回: (confidence_score, grade)
    """
    win_rate = stats.get("win_rate", 0)
    total_return = stats.get("total_return", 0)
    closed_trades = stats.get("closed_trades", 0)

    # 基础分数
    score = 0

    # 胜率评分 (最高40分)
    if win_rate >= 70:
        score += 40
    elif win_rate >= 60:
        score += 30
    elif win_rate >= 50:
        score += 20
    else:
        score += 10

    # 收益率评分 (最高40分)
    if total_return >= 10:
        score += 40
    elif total_return >= 5:
        score += 30
    elif total_return >= 2:
        score += 20
    elif total_return > 0:
        score += 10

    # 交易次数评分 (最高20分)
    if closed_trades >= 5:
        score += 20
    elif closed_trades >= 3:
        score += 15
    elif closed_trades >= 2:
        score += 10
    else:
        score += 5

    # 确定等级
    if score >= 80:
        grade = "A"
    elif score >= 60:
        grade = "B"
    elif score >= 40:
        grade = "C"
    else:
        grade = "D"

    return score, grade


def manual_qualify_for_library(session_id: int) -> Dict:
    """手动将策略加入股票适配库"""
    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            return {"error": "Session not found"}

        if session.qualified_for_library:
            return {"error": "Already qualified for library"}

        strategy = get_strategy_by_id(db, session.strategy_id)
        if not strategy:
            return {"error": "Strategy not found"}

        stats = calculate_session_statistics(session)
        confidence, grade = calculate_confidence_and_grade(stats)

        # 创建股票-策略匹配记录
        match = create_stock_strategy_match(
            db=db,
            user_id=session.user_id,
            ticker=session.ticker,
            strategy_id=session.strategy_id,
            strategy_name=strategy.name,
            strategy_type=strategy.strategy_type,
            simulation_id=session.id,
            simulation_win_rate=stats["win_rate"],
            simulation_return=stats["total_return"],
            confidence_score=confidence,
            match_grade=grade
        )

        update_ai_simulation_session(
            db, session,
            qualified_for_library=True,
            qualification_date=datetime.utcnow()
        )

        return {
            "success": True,
            "match_id": match.id,
            "confidence_score": confidence,
            "grade": grade
        }


def check_all_active_simulations() -> List[Dict]:
    """检查所有活跃的模拟交易会话（供定时任务调用）"""
    results = []

    with get_db() as db:
        active_sessions = get_active_ai_simulation_sessions(db)

        for session in active_sessions:
            try:
                result = check_signal_and_trade(session.id)
                results.append({
                    "session_id": session.id,
                    "ticker": session.ticker,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "session_id": session.id,
                    "ticker": session.ticker,
                    "error": str(e)
                })

    return results
