import math
from typing import Dict, List


def _safe_pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value / base) * 100


def _calc_drawdowns(equity_curve: List[float]) -> Dict[str, List[float]]:
    if not equity_curve:
        return {"drawdown": [], "running_max": []}

    running_max = []
    drawdown = []
    current_max = equity_curve[0]
    for value in equity_curve:
        current_max = max(current_max, value)
        running_max.append(current_max)
        if current_max == 0:
            drawdown.append(0.0)
        else:
            drawdown.append((value - current_max) / current_max)
    return {"drawdown": drawdown, "running_max": running_max}


def generate_backtest_report(backtest: Dict) -> Dict:
    """Generate a richer report from backtest results."""
    equity_curve = backtest.get("equity_curve") or []
    daily_returns = backtest.get("daily_returns") or []

    drawdown_info = _calc_drawdowns(equity_curve)
    volatility = 0.0
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        volatility = math.sqrt(variance) * math.sqrt(252)

    cumulative_return = 0.0
    if equity_curve:
        cumulative_return = _safe_pct(equity_curve[-1] - equity_curve[0], equity_curve[0])

    return {
        "summary": {
            "final_value": backtest.get("final_value"),
            "total_return": backtest.get("total_return"),
            "annualized_return": backtest.get("annualized_return"),
            "sharpe_ratio": backtest.get("sharpe_ratio"),
            "max_drawdown": backtest.get("max_drawdown"),
            "volatility": round(volatility * 100, 4),
            "cumulative_return": round(cumulative_return, 4),
            "total_trades": backtest.get("total_trades"),
            "win_rate": backtest.get("win_rate"),
            "profit_factor": backtest.get("profit_factor"),
        },
        "series": {
            "equity_curve": equity_curve,
            "drawdown": drawdown_info["drawdown"],
            "running_max": drawdown_info["running_max"],
            "daily_returns": daily_returns,
        },
        "trade_history": backtest.get("trade_history") or [],
    }
