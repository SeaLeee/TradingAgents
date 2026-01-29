"""
增强版回测引擎
包含更多统计指标和可视化数据
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import json

@dataclass
class BacktestMetrics:
    """回测指标数据类"""
    # 基础指标
    total_return: float  # 累计收益率
    annual_return: float  # 年化收益率
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤
    volatility: float  # 波动率

    # 扩展指标
    calmar_ratio: float  # 卡玛比率（年化收益/最大回撤）
    sortino_ratio: float  # 索提诺比率
    win_rate: float  # 胜率
    profit_loss_ratio: float  # 盈亏比
    max_consecutive_losses: int  # 最大连续亏损次数
    max_consecutive_wins: int  # 最大连续盈利次数
    recovery_factor: float  # 恢复因子
    beta: float  # 贝塔
    alpha: float  # 阿尔法
    information_ratio: float  # 信息比率
    downside_deviation: float  # 下行标准差
    var_95: float  # 95%置信度的VaR
    cvar_95: float  # 95%置信度的CVaR

    # 时间相关指标
    best_day: float  # 最佳单日收益
    worst_day: float  # 最差单日收益
    best_month: float  # 最佳月度收益
    worst_month: float  # 最差月度收益
    positive_months: int  # 盈利月份数
    negative_months: int  # 亏损月份数
    longest_drawdown_duration: int  # 最长回撤持续天数

    # 交易相关指标
    total_trades: int  # 总交易次数
    winning_trades: int  # 盈利交易次数
    losing_trades: int  # 亏损交易次数
    avg_trade_return: float  # 平均交易收益
    avg_winning_trade: float  # 平均盈利交易收益
    avg_losing_trade: float  # 平均亏损交易损失
    largest_winning_trade: float  # 最大单笔盈利
    largest_losing_trade: float  # 最大单笔亏损
    avg_holding_period: float  # 平均持仓时间（天）

    # 基准对比
    excess_return: float  # 超额收益
    tracking_error: float  # 跟踪误差
    active_return: float  # 主动收益


class EnhancedBacktest:
    """增强版回测引擎"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.portfolio_value = []
        self.trades = []
        self.daily_returns = []
        self.benchmark_returns = []
        self.cash = initial_capital
        self.transaction_cost = 0.001  # 千分之一的手续费

    def run_backtest(self, strategy_func, data: pd.DataFrame,
                    benchmark_data: Optional[pd.DataFrame] = None,
                    **strategy_params) -> Dict[str, Any]:
        """运行回测"""
        results = {
            'portfolio_value': [],
            'positions': [],
            'trades': [],
            'signals': [],
            'daily_returns': [],
            'benchmark_returns': [],
            'drawdown': [],
            'monthly_returns': {},
            'metrics': None
        }

        # 逐日回测
        for i, date in enumerate(data.index):
            # 生成交易信号
            signals = strategy_func(data.iloc[:i+1], **strategy_params)

            # 执行交易
            if signals:
                self._execute_trades(signals, data.iloc[i])
                results['signals'].append({
                    'date': date,
                    'signals': signals
                })

            # 计算当日组合价值
            portfolio_val = self._calculate_portfolio_value(data.iloc[i])
            results['portfolio_value'].append({
                'date': date,
                'value': portfolio_val,
                'cash': self.cash,
                'positions_value': portfolio_val - self.cash
            })

            # 记录持仓
            results['positions'].append({
                'date': date,
                'positions': self.positions.copy()
            })

            # 计算收益率
            if i > 0:
                daily_return = (portfolio_val / results['portfolio_value'][i-1]['value'] - 1)
                results['daily_returns'].append(daily_return)

                if benchmark_data is not None and date in benchmark_data.index:
                    bench_return = (benchmark_data.loc[date] / benchmark_data.iloc[i-1] - 1).values[0]
                    results['benchmark_returns'].append(bench_return)

        # 计算回测指标
        results['metrics'] = self._calculate_metrics(
            results,
            benchmark_returns=results['benchmark_returns'] if benchmark_data is not None else None
        )

        # 计算回撤序列
        results['drawdown'] = self._calculate_drawdown_series(results['portfolio_value'])

        # 计算月度收益
        results['monthly_returns'] = self._calculate_monthly_returns(results)

        return results

    def _execute_trades(self, signals: List[Dict], current_prices: pd.Series):
        """执行交易"""
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            quantity = signal.get('quantity', 0)
            weight = signal.get('weight', 0)

            if action == 'BUY':
                if weight > 0:
                    # 基于权重计算数量
                    invest_amount = self.capital * weight
                    quantity = int(invest_amount / current_prices[symbol])

                if quantity > 0:
                    cost = quantity * current_prices[symbol] * (1 + self.transaction_cost)
                    if cost <= self.cash:
                        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                        self.cash -= cost
                        self.trades.append({
                            'date': current_prices.name,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': current_prices[symbol],
                            'cost': cost
                        })

            elif action == 'SELL':
                if symbol in self.positions and self.positions[symbol] > 0:
                    if quantity == 0 or quantity > self.positions[symbol]:
                        quantity = self.positions[symbol]

                    proceeds = quantity * current_prices[symbol] * (1 - self.transaction_cost)
                    self.positions[symbol] -= quantity
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    self.cash += proceeds
                    self.trades.append({
                        'date': current_prices.name,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': current_prices[symbol],
                        'proceeds': proceeds
                    })

    def _calculate_portfolio_value(self, current_prices: pd.Series) -> float:
        """计算组合总价值"""
        positions_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                positions_value += quantity * current_prices[symbol]
        return self.cash + positions_value

    def _calculate_metrics(self, results: Dict,
                          risk_free_rate: float = 0.02,
                          benchmark_returns: Optional[List[float]] = None) -> BacktestMetrics:
        """计算所有回测指标"""
        portfolio_values = [r['value'] for r in results['portfolio_value']]
        returns = np.array(results['daily_returns'])

        if len(returns) == 0:
            returns = np.array([0])

        # 基础指标
        total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100
        days = len(portfolio_values)
        annual_return = ((portfolio_values[-1] / self.initial_capital) ** (252 / days) - 1) * 100 if days > 0 else 0

        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
        sharpe_ratio = (annual_return - risk_free_rate * 100) / volatility if volatility > 0 else 0

        # 计算最大回撤
        drawdown_series = self._calculate_drawdown_series(results['portfolio_value'])
        max_drawdown = min(drawdown_series) if drawdown_series else 0

        # 卡玛比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 索提诺比率（只考虑下行波动）
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate * 100) / downside_deviation if downside_deviation > 0 else 0

        # 交易统计
        winning_trades = [t for t in self.trades if t.get('proceeds', 0) > t.get('cost', 0)]
        losing_trades = [t for t in self.trades if t.get('proceeds', 0) <= t.get('cost', 0)]

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t.get('proceeds', 0) - t.get('cost', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('proceeds', 0) - t.get('cost', 0) for t in losing_trades]) if losing_trades else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # 连续盈亏统计
        max_consecutive_wins = self._calculate_max_consecutive(returns > 0)
        max_consecutive_losses = self._calculate_max_consecutive(returns < 0)

        # VaR和CVaR
        var_95 = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * 100 if len(returns) > 0 else 0

        # 最佳/最差表现
        best_day = np.max(returns) * 100 if len(returns) > 0 else 0
        worst_day = np.min(returns) * 100 if len(returns) > 0 else 0

        # 月度统计
        monthly_returns = list(results.get('monthly_returns', {}).values())
        best_month = max(monthly_returns) if monthly_returns else 0
        worst_month = min(monthly_returns) if monthly_returns else 0
        positive_months = sum(1 for r in monthly_returns if r > 0)
        negative_months = sum(1 for r in monthly_returns if r < 0)

        # 基准对比
        excess_return = 0
        tracking_error = 0
        beta = 0
        alpha = 0
        information_ratio = 0

        if benchmark_returns:
            bench_returns = np.array(benchmark_returns)
            excess_returns = returns - bench_returns[:len(returns)]
            excess_return = np.mean(excess_returns) * 252 * 100
            tracking_error = np.std(excess_returns) * np.sqrt(252) * 100
            information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

            # 计算Beta和Alpha
            if len(returns) > 1 and len(bench_returns) > 1:
                covariance = np.cov(returns, bench_returns[:len(returns)])[0, 1]
                variance = np.var(bench_returns[:len(returns)])
                beta = covariance / variance if variance > 0 else 0
                alpha = (annual_return - risk_free_rate * 100) - beta * (np.mean(bench_returns) * 252 * 100 - risk_free_rate * 100)

        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            recovery_factor=total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95,
            best_day=best_day,
            worst_day=worst_day,
            best_month=best_month,
            worst_month=worst_month,
            positive_months=positive_months,
            negative_months=negative_months,
            longest_drawdown_duration=self._calculate_longest_drawdown_duration(drawdown_series),
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_return=(total_return / len(self.trades)) if self.trades else 0,
            avg_winning_trade=avg_win,
            avg_losing_trade=avg_loss,
            largest_winning_trade=max([t.get('proceeds', 0) - t.get('cost', 0) for t in winning_trades]) if winning_trades else 0,
            largest_losing_trade=min([t.get('proceeds', 0) - t.get('cost', 0) for t in losing_trades]) if losing_trades else 0,
            avg_holding_period=self._calculate_avg_holding_period(),
            excess_return=excess_return,
            tracking_error=tracking_error,
            active_return=excess_return
        )

    def _calculate_drawdown_series(self, portfolio_values: List[Dict]) -> List[float]:
        """计算回撤序列"""
        values = [p['value'] for p in portfolio_values]
        drawdowns = []
        peak = values[0] if values else 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(drawdown)

        return drawdowns

    def _calculate_max_consecutive(self, condition: np.ndarray) -> int:
        """计算最大连续次数"""
        if len(condition) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for c in condition:
            if c:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_longest_drawdown_duration(self, drawdown_series: List[float]) -> int:
        """计算最长回撤持续时间"""
        if not drawdown_series:
            return 0

        max_duration = 0
        current_duration = 0

        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_avg_holding_period(self) -> float:
        """计算平均持仓时间"""
        # 这里简化处理，实际需要根据具体买卖记录计算
        return len(self.portfolio_value) / max(len(self.trades), 1)

    def _calculate_monthly_returns(self, results: Dict) -> Dict[str, float]:
        """计算月度收益率"""
        if not results['portfolio_value']:
            return {}

        df = pd.DataFrame(results['portfolio_value'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        monthly = df['value'].resample('ME').last()
        monthly_returns = monthly.pct_change() * 100
        monthly_returns = monthly_returns.dropna()

        return monthly_returns.to_dict()