"""
ETF板块因子分析模块
包含动量、波动率、风险调整动量等因子计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf

class ETFFactorAnalyzer:
    """ETF因子分析器"""

    # 行业ETF列表
    SECTOR_ETFS = {
        # 科技板块
        '512480.SS': '半导体ETF',
        '515070.SS': '人工智能ETF',
        '515880.SS': '通信ETF',
        '512720.SS': '计算机ETF',
        '512760.SS': '芯片ETF',
        '512980.SS': '传媒ETF',
        '515230.SS': '软件ETF',

        # 消费板块
        '159928.SZ': '消费ETF',
        '512600.SS': '家电ETF',
        '515170.SS': '食品饮料ETF',
        '159825.SZ': '农业ETF',
        '516110.SS': '汽车ETF',
        '159707.SZ': '家居ETF',

        # 医药板块
        '512010.SS': '医药ETF',
        '516820.SS': '创新药ETF',
        '159883.SZ': '医疗器械ETF',
        '512290.SS': '生物医药ETF',
        '512170.SS': '医疗ETF',
        '159828.SZ': '中证医药ETF',

        # 金融板块
        '512880.SS': '证券ETF',
        '512800.SS': '银行ETF',
        '510230.SS': '非银ETF',
        '159931.SZ': '金融ETF',
        '512070.SS': '保险ETF',
        '512000.SS': '券商ETF',

        # 周期板块
        '159880.SZ': '有色ETF',
        '515220.SS': '煤炭ETF',
        '515210.SS': '钢铁ETF',
        '159870.SZ': '化工ETF',
        '159787.SZ': '建材ETF',
        '516950.SS': '基建ETF',

        # 新能源板块
        '515700.SS': '新能源车ETF',
        '515790.SS': '光伏ETF',
        '159840.SZ': '电池ETF',
        '159615.SZ': '风电ETF',
        '159885.SZ': '碳中和ETF',
        '516660.SS': '新能源ETF',
        '159806.SZ': '新能源汽车ETF',

        # 军工板块
        '512660.SS': '军工ETF',
    }

    # 宽基指数ETF
    BROAD_ETFS = {
        '510300.SS': '沪深300ETF',
        '510500.SS': '中证500ETF',
        '159915.SZ': '创业板ETF',
        '588000.SS': '科创50ETF',
        '510050.SS': '上证50ETF',
        '512100.SS': '中证1000ETF',
    }

    def __init__(self):
        self.all_etfs = {**self.SECTOR_ETFS, **self.BROAD_ETFS}

    def fetch_etf_data(self, symbols: List[str],
                       start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """获取ETF历史数据"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        all_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    all_data[symbol] = df['Close']
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue

        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()

    def calculate_returns(self, prices: pd.DataFrame,
                         periods: List[int] = [1, 5, 10, 20, 60, 120, 250]) -> pd.DataFrame:
        """计算不同周期的收益率"""
        returns = pd.DataFrame(index=prices.columns)

        for period in periods:
            period_return = (prices.iloc[-1] / prices.iloc[-min(period+1, len(prices))] - 1) * 100
            returns[f'{period}d'] = period_return

        return returns

    def calculate_momentum(self, prices: pd.DataFrame,
                          lookback_periods: List[int] = [20, 60]) -> pd.DataFrame:
        """计算动量因子"""
        momentum = pd.DataFrame(index=prices.columns)

        for period in lookback_periods:
            if len(prices) >= period:
                returns = prices.pct_change(period).iloc[-1] * 100
                momentum[f'momentum_{period}d'] = returns
            else:
                momentum[f'momentum_{period}d'] = np.nan

        return momentum

    def calculate_volatility(self, prices: pd.DataFrame,
                           periods: List[int] = [20, 60]) -> pd.DataFrame:
        """计算波动率因子"""
        volatility = pd.DataFrame(index=prices.columns)

        for period in periods:
            if len(prices) >= period:
                daily_returns = prices.pct_change()
                vol = daily_returns.rolling(window=period).std().iloc[-1] * np.sqrt(252) * 100
                volatility[f'volatility_{period}d'] = vol
            else:
                volatility[f'volatility_{period}d'] = np.nan

        return volatility

    def calculate_risk_adjusted_momentum(self, prices: pd.DataFrame,
                                       periods: List[int] = [20, 60]) -> pd.DataFrame:
        """计算风险调整后的动量（夏普比率）"""
        risk_adjusted = pd.DataFrame(index=prices.columns)

        for period in periods:
            if len(prices) >= period:
                returns = prices.pct_change()
                rolling_return = returns.rolling(window=period).mean() * 252
                rolling_std = returns.rolling(window=period).std() * np.sqrt(252)
                sharpe = (rolling_return / rolling_std).iloc[-1] * 100
                risk_adjusted[f'risk_adj_momentum_{period}d'] = sharpe
            else:
                risk_adjusted[f'risk_adj_momentum_{period}d'] = np.nan

        return risk_adjusted

    def calculate_relative_strength(self, prices: pd.DataFrame,
                                   benchmark_col: str = '510300.SS',
                                   period: int = 20) -> pd.Series:
        """计算相对强弱指标"""
        if benchmark_col not in prices.columns:
            return pd.Series(index=prices.columns)

        benchmark_return = (prices[benchmark_col].iloc[-1] / prices[benchmark_col].iloc[-period-1] - 1)

        relative_strength = pd.Series(index=prices.columns)
        for col in prices.columns:
            etf_return = (prices[col].iloc[-1] / prices[col].iloc[-period-1] - 1)
            relative_strength[col] = (etf_return - benchmark_return) * 100

        return relative_strength

    def calculate_rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        rsi_values = pd.Series(index=prices.columns)

        for col in prices.columns:
            price_series = prices[col]
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values[col] = rsi.iloc[-1]

        return rsi_values

    def calculate_max_drawdown(self, prices: pd.DataFrame,
                              period: Optional[int] = None) -> pd.Series:
        """计算最大回撤"""
        max_drawdown = pd.Series(index=prices.columns)

        for col in prices.columns:
            if period:
                price_series = prices[col].iloc[-period:]
            else:
                price_series = prices[col]

            cummax = price_series.expanding().max()
            drawdown = (price_series - cummax) / cummax * 100
            max_drawdown[col] = drawdown.min()

        return max_drawdown

    def calculate_all_factors(self, symbols: Optional[List[str]] = None,
                             start_date: Optional[str] = None) -> pd.DataFrame:
        """计算所有因子"""
        if not symbols:
            symbols = list(self.all_etfs.keys())

        # 获取价格数据
        prices = self.fetch_etf_data(symbols, start_date=start_date)
        if prices.empty:
            return pd.DataFrame()

        # 初始化结果DataFrame
        factors = pd.DataFrame()
        factors['symbol'] = prices.columns
        factors['name'] = [self.all_etfs.get(s, s) for s in prices.columns]

        # 计算收益率
        returns = self.calculate_returns(prices)
        for col in returns.columns:
            factors[col] = returns[col].values

        # 计算动量
        momentum = self.calculate_momentum(prices)
        for col in momentum.columns:
            factors[col] = momentum[col].values

        # 计算波动率
        volatility = self.calculate_volatility(prices)
        for col in volatility.columns:
            factors[col] = volatility[col].values

        # 计算风险调整动量
        risk_adj = self.calculate_risk_adjusted_momentum(prices)
        for col in risk_adj.columns:
            factors[col] = risk_adj[col].values

        # 计算相对强弱
        factors['relative_strength_20d'] = self.calculate_relative_strength(prices).values

        # 计算RSI
        factors['rsi_14d'] = self.calculate_rsi(prices).values

        # 计算最大回撤
        factors['max_drawdown'] = self.calculate_max_drawdown(prices).values

        # 添加更新时间
        factors['update_time'] = datetime.now()

        return factors

    def rank_by_factor(self, factors: pd.DataFrame,
                      factor_name: str,
                      ascending: bool = False) -> pd.DataFrame:
        """根据指定因子排序"""
        if factor_name not in factors.columns:
            return factors

        return factors.sort_values(by=factor_name, ascending=ascending)

    def get_top_performers(self, factors: pd.DataFrame,
                          factor_name: str = 'momentum_20d',
                          top_n: int = 10) -> pd.DataFrame:
        """获取表现最好的ETF"""
        sorted_df = self.rank_by_factor(factors, factor_name, ascending=False)
        return sorted_df.head(top_n)

    def get_sector_rotation_signals(self, factors: pd.DataFrame,
                                   strategy: str = 'momentum') -> List[Dict]:
        """生成板块轮动信号"""
        signals = []

        if strategy == 'momentum':
            # 基于20日动量选择
            top_etfs = self.get_top_performers(factors, 'momentum_20d', 5)

        elif strategy == 'low_volatility':
            # 选择低波动率ETF
            sorted_df = factors.sort_values(by='volatility_20d', ascending=True)
            top_etfs = sorted_df.head(5)

        elif strategy == 'risk_adjusted':
            # 基于风险调整后动量
            top_etfs = self.get_top_performers(factors, 'risk_adj_momentum_20d', 5)

        else:
            return signals

        for _, row in top_etfs.iterrows():
            signals.append({
                'symbol': row['symbol'],
                'name': row['name'],
                'score': row.get(f'{strategy}_20d', 0),
                'weight': 0.2,  # 均等权重
                'signal_time': datetime.now()
            })

        return signals

    def calculate_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """计算ETF之间的相关性矩阵"""
        returns = prices.pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix