from typing import Dict, Any
import yfinance as yf

from .backtest_engine import calculate_technical_indicators, ensure_moving_average


def _latest_signal_from_df(df, signal_column: str) -> int:
    if df.empty:
        return 0
    value = df.iloc[-1][signal_column]
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def generate_signal(ticker: str, strategy_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single latest signal for a ticker."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1d")
    if df.empty:
        return {"error": "No data available for the ticker."}

    df = calculate_technical_indicators(df)
    params = params or {}

    if strategy_type in ("moving_average_crossover", "trend_following"):
        short_window = params.get("short_window", 20)
        long_window = params.get("long_window", 50)
        short_col = ensure_moving_average(df, short_window)
        long_col = ensure_moving_average(df, long_window)
        df["Signal"] = 0
        df.loc[df[short_col] > df[long_col], "Signal"] = 1
        df.loc[df[short_col] < df[long_col], "Signal"] = -1
        reason = f"短均线({short_window}) vs 长均线({long_window})"
    elif strategy_type in ("rsi", "mean_reversion"):
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        df["Signal"] = 0
        df.loc[df["RSI"] < oversold, "Signal"] = 1
        df.loc[df["RSI"] > overbought, "Signal"] = -1
        reason = f"RSI({oversold}/{overbought})"
    elif strategy_type == "macd":
        df["Signal"] = 0
        df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1
        df.loc[df["MACD"] < df["MACD_Signal"], "Signal"] = -1
        reason = "MACD 交叉"
    elif strategy_type in ("bollinger_bands", "breakout"):
        df["Signal"] = 0
        df.loc[df["Close"] > df["BB_Upper"], "Signal"] = 1
        df.loc[df["Close"] < df["BB_Lower"], "Signal"] = -1
        reason = "布林带突破"
    elif strategy_type in ("momentum", "growth"):
        threshold = params.get("momentum_threshold", 0.05)
        df["Signal"] = 0
        df.loc[df["Momentum"] > threshold, "Signal"] = 1
        df.loc[df["Momentum"] < -threshold, "Signal"] = -1
        reason = "动量阈值"
    elif strategy_type == "value":
        ma_period = params.get("ma_period", 50)
        deviation_threshold = params.get("deviation_threshold", 0.03)
        ma_col = ensure_moving_average(df, ma_period)
        deviation = (df["Close"] - df[ma_col]) / df[ma_col]
        df["Signal"] = 0
        df.loc[deviation < -deviation_threshold, "Signal"] = 1
        df.loc[deviation > deviation_threshold, "Signal"] = -1
        reason = "估值型均值回归"
    else:
        return {"error": f"Unknown strategy type: {strategy_type}"}

    latest_signal = _latest_signal_from_df(df, "Signal")
    signal_map = {1: "buy", -1: "sell", 0: "hold"}
    return {
        "ticker": ticker.upper(),
        "strategy_type": strategy_type,
        "signal": signal_map[latest_signal],
        "reason": reason,
        "latest_price": float(df.iloc[-1]["Close"]),
        "timestamp": df.index[-1].strftime("%Y-%m-%d"),
    }
