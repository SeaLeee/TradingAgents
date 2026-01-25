"""
Backtest Engine for Trading Strategies
Implements various trading strategies and runs backtests on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
import yfinance as yf


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators"""
    # Moving averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Momentum
    df['Momentum'] = df['Close'].pct_change(periods=20)
    
    return df


def ensure_moving_average(df: pd.DataFrame, period: int) -> str:
    """Ensure a moving average column exists and return its name."""
    column = f"MA_{period}"
    if column not in df.columns:
        df[column] = df["Close"].rolling(window=period).mean()
    return column


def apply_default_params(params: Optional[Dict], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge provided params with defaults."""
    merged = dict(defaults)
    if params:
        merged.update(params)
    return merged


def run_moving_average_crossover(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    short_window: int = 20,
    long_window: int = 50,
    commission: float = 0.001
) -> Dict:
    """Moving Average Crossover Strategy"""
    df = calculate_technical_indicators(df.copy())
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MA_20'] > df['MA_50'], 'Signal'] = 1  # Buy
    df.loc[df['MA_20'] < df['MA_50'], 'Signal'] = -1  # Sell
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_rsi_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    commission: float = 0.001
) -> Dict:
    """RSI Overbought/Oversold Strategy"""
    df = calculate_technical_indicators(df.copy())
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['RSI'] < oversold, 'Signal'] = 1  # Buy when oversold
    df.loc[df['RSI'] > overbought, 'Signal'] = -1  # Sell when overbought
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_macd_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    commission: float = 0.001
) -> Dict:
    """MACD Crossover Strategy"""
    df = calculate_technical_indicators(df.copy())
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1  # Buy
    df.loc[df['MACD'] < df['MACD_Signal'], 'Signal'] = -1  # Sell
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_bollinger_bands_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    period: int = 20,
    std_dev: float = 2,
    commission: float = 0.001
) -> Dict:
    """Bollinger Bands Breakout Strategy"""
    df = calculate_technical_indicators(df.copy())
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['Close'] > df['BB_Upper'], 'Signal'] = 1  # Buy on upper breakout
    df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = -1  # Sell on lower breakout
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_momentum_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    lookback_period: int = 20,
    momentum_threshold: float = 0.05,
    commission: float = 0.001
) -> Dict:
    """Momentum Strategy"""
    df = calculate_technical_indicators(df.copy())
    
    # Generate signals based on momentum
    df['Signal'] = 0
    df.loc[df['Momentum'] > momentum_threshold, 'Signal'] = 1  # Buy on positive momentum
    df.loc[df['Momentum'] < -momentum_threshold, 'Signal'] = -1  # Sell on negative momentum
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_mean_reversion_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    ma_period: int = 20,
    deviation_threshold: float = 0.02,
    commission: float = 0.001
) -> Dict:
    """Mean Reversion Strategy"""
    df = calculate_technical_indicators(df.copy())

    ma_column = ensure_moving_average(df, ma_period)
    # Calculate deviation from moving average
    df['Deviation'] = (df['Close'] - df[ma_column]) / df[ma_column]
    
    # Generate signals
    df['Signal'] = 0
    df.loc[df['Deviation'] < -deviation_threshold, 'Signal'] = 1  # Buy when below MA
    df.loc[df['Deviation'] > deviation_threshold, 'Signal'] = -1  # Sell when above MA
    
    # Calculate positions
    df['Position'] = df['Signal'].diff()
    
    return run_backtest(df, initial_capital, commission)


def run_custom_strategy(
    df: pd.DataFrame,
    initial_capital: float = 100000,
    base_strategy: str = "moving_average_crossover",
    commission: float = 0.001,
    **params: Any
) -> Dict:
    """Custom strategy wrapper that reuses a base strategy."""
    if not base_strategy:
        base_strategy = "moving_average_crossover"
    resolved_type = STRATEGY_ALIASES.get(base_strategy, base_strategy)
    strategy = STRATEGY_REGISTRY.get(resolved_type)
    if not strategy:
        raise ValueError(f"Unknown base strategy: {base_strategy}")

    merged_params = apply_default_params(params, strategy["defaults"])
    handler: Callable[..., Dict] = strategy["handler"]
    return handler(df, initial_capital, commission=commission, **merged_params)


STRATEGY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "moving_average_crossover": {
        "handler": run_moving_average_crossover,
        "defaults": {"short_window": 20, "long_window": 50},
    },
    "rsi": {
        "handler": run_rsi_strategy,
        "defaults": {"rsi_period": 14, "oversold": 30, "overbought": 70},
    },
    "macd": {
        "handler": run_macd_strategy,
        "defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    },
    "bollinger_bands": {
        "handler": run_bollinger_bands_strategy,
        "defaults": {"period": 20, "std_dev": 2},
    },
    "momentum": {
        "handler": run_momentum_strategy,
        "defaults": {"lookback_period": 20, "momentum_threshold": 0.05},
    },
    "mean_reversion": {
        "handler": run_mean_reversion_strategy,
        "defaults": {"ma_period": 20, "deviation_threshold": 0.02},
    },
    "value": {
        "handler": run_mean_reversion_strategy,
        "defaults": {"ma_period": 50, "deviation_threshold": 0.03},
    },
    "growth": {
        "handler": run_momentum_strategy,
        "defaults": {"lookback_period": 20, "momentum_threshold": 0.05},
    },
    "custom": {
        "handler": run_custom_strategy,
        "defaults": {"base_strategy": "moving_average_crossover"},
    },
}

STRATEGY_ALIASES: Dict[str, str] = {
    "trend_following": "moving_average_crossover",
    "breakout": "bollinger_bands",
}


def run_backtest(df: pd.DataFrame, initial_capital: float, commission: float = 0.001) -> Dict:
    """Run backtest on dataframe with signals"""
    capital = initial_capital
    shares = 0
    equity_curve = [initial_capital]
    trades = []
    daily_returns = [0]
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]['Close']
        position_change = df.iloc[i]['Position']
        
        # Execute trades
        if position_change > 0:  # Buy
            if shares == 0:  # Only buy if we don't have position
                shares_to_buy = int(capital / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + commission)
                    if cost <= capital:
                        shares = shares_to_buy
                        capital -= cost
                        trades.append({
                            'date': df.index[i].strftime('%Y-%m-%d'),
                            'action': 'buy',
                            'price': float(current_price),
                            'shares': shares_to_buy,
                            'value': float(cost)
                        })
        elif position_change < 0:  # Sell
            if shares > 0:  # Only sell if we have position
                proceeds = shares * current_price * (1 - commission)
                capital += proceeds
                
                # Calculate trade P&L
                entry_price = trades[-1]['price'] if trades else current_price
                pnl = (current_price - entry_price) * shares
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                
                trades.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'action': 'sell',
                    'price': float(current_price),
                    'shares': shares,
                    'value': float(proceeds),
                    'pnl': float(pnl),
                    'pnl_pct': float(pnl_pct)
                })
                
                shares = 0
        
        # Calculate current portfolio value
        current_value = capital + (shares * current_price)
        equity_curve.append(current_value)
        
        # Calculate daily return
        prev_value = equity_curve[-2] if len(equity_curve) > 1 else initial_capital
        daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        daily_returns.append(daily_return)
    
    # Calculate final value (close any open positions)
    if shares > 0:
        final_price = df.iloc[-1]['Close']
        final_value = capital + (shares * final_price)
    else:
        final_value = capital
    
    # Calculate metrics
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Annualized return
    days = len(df)
    years = days / 252  # Trading days per year
    annualized_return = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Sharpe ratio (assuming risk-free rate = 0)
    returns_array = np.array(daily_returns)
    if len(returns_array) > 1 and returns_array.std() > 0:
        sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100
    
    # Win rate
    closed_trades = [t for t in trades if t.get('pnl') is not None]
    if closed_trades:
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        losing_trades_list = [t for t in closed_trades if t['pnl'] <= 0]
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades_list]) if losing_trades_list else 0
        
        # Profit factor
        total_wins = sum([t['pnl'] for t in winning_trades])
        total_losses = abs(sum([t['pnl'] for t in losing_trades_list]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
    
    return {
        'final_value': float(final_value),
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades) if closed_trades else 0,
        'losing_trades': len(losing_trades_list) if closed_trades else 0,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'equity_curve': [float(v) for v in equity_curve],
        'trade_history': trades,
        'daily_returns': [float(r) for r in daily_returns]
    }


def run_strategy_backtest(
    strategy_type: str,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    params: Dict = None
) -> Dict:
    """Run backtest for a specific strategy"""
    try:
        # Download historical data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data available for {ticker} in the specified date range")
        
        # Ensure we have required columns
        if 'Close' not in df.columns:
            raise ValueError("Historical data missing 'Close' column")
        
        # Run strategy
        resolved_type = STRATEGY_ALIASES.get(strategy_type, strategy_type)
        strategy = STRATEGY_REGISTRY.get(resolved_type)
        if not strategy:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        merged_params = apply_default_params(params, strategy["defaults"])
        handler: Callable[..., Dict] = strategy["handler"]
        return handler(df, initial_capital, **merged_params)
            
    except Exception as e:
        return {
            'error': str(e),
            'final_value': initial_capital,
            'total_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'equity_curve': [],
            'trade_history': [],
            'daily_returns': []
        }
