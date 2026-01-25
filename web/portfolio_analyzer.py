from typing import Dict, Any


def analyze_portfolio(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generate portfolio-level analytics from existing stats."""
    total_trades = stats.get("total_trades", 0)
    win_rate = stats.get("win_rate", 0)
    total_pnl = stats.get("total_pnl", 0)
    pnl_percentage = stats.get("pnl_percentage", 0)

    risk_score = 0
    if total_trades > 0:
        risk_score += min(50, total_trades)
    if abs(pnl_percentage) > 20:
        risk_score += 20
    if win_rate < 40:
        risk_score += 15

    risk_level = "low"
    if risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"

    return {
        "performance": {
            "total_pnl": total_pnl,
            "pnl_percentage": pnl_percentage,
            "win_rate": win_rate,
            "average_trade": stats.get("average_trade"),
            "best_trade": stats.get("best_trade"),
            "worst_trade": stats.get("worst_trade"),
        },
        "risk": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "open_positions": stats.get("open_positions"),
            "total_trades": total_trades,
        },
        "strategy_breakdown": stats.get("strategy_stats", {}),
        "open_trades": stats.get("open_trades", []),
    }
