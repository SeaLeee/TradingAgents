"""
增强版回测结果展示React组件
展示详细的回测统计指标和图表
"""

import React from 'react';
import {
  Card, CardContent, Typography, Grid, Box, Paper,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, LinearProgress, Divider, Alert
} from '@mui/material';
import { Line } from '@nivo/line';
import { Bar } from '@nivo/bar';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import ShowChartIcon from '@mui/icons-material/ShowChart';

const EnhancedBacktestResults = ({ results }) => {
  if (!results || !results.metrics) {
    return <Alert severity="info">暂无回测结果</Alert>;
  }

  const { metrics, portfolio_values, drawdown, monthly_returns, trades } = results;

  // 准备累计收益曲线数据
  const cumulativeReturnsData = [{
    id: '策略',
    data: portfolio_values.map(pv => ({
      x: pv.date,
      y: ((pv.value / results.initial_capital - 1) * 100).toFixed(2)
    }))
  }];

  // 准备回撤曲线数据
  const drawdownData = [{
    id: '回撤',
    data: drawdown.map(dd => ({
      x: dd.date,
      y: dd.drawdown
    }))
  }];

  // 准备月度收益柱状图数据
  const monthlyReturnsData = monthly_returns.map(mr => ({
    month: mr.month.substring(0, 7),
    return: mr.return
  }));

  const MetricCard = ({ title, value, unit = '', color = 'primary', icon }) => (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Box display="flex" justifyContent="space-between" alignItems="start">
        <Box>
          <Typography variant="caption" color="textSecondary">
            {title}
          </Typography>
          <Typography variant="h5" sx={{
            color: color === 'positive' ? 'success.main' :
                   color === 'negative' ? 'error.main' : 'primary.main',
            fontWeight: 'bold',
            mt: 0.5
          }}>
            {value}{unit}
          </Typography>
        </Box>
        {icon && <Box>{icon}</Box>}
      </Box>
    </Paper>
  );

  return (
    <Box>
      {/* 核心指标卡片 */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="累计收益率"
            value={metrics.basic_metrics.total_return}
            unit="%"
            color={metrics.basic_metrics.total_return > 0 ? 'positive' : 'negative'}
            icon={metrics.basic_metrics.total_return > 0 ? <TrendingUpIcon color="success" /> : <TrendingDownIcon color="error" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="年化收益率"
            value={metrics.basic_metrics.annual_return}
            unit="%"
            color={metrics.basic_metrics.annual_return > 0 ? 'positive' : 'negative'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="夏普比率"
            value={metrics.basic_metrics.sharpe_ratio}
            color={metrics.basic_metrics.sharpe_ratio > 1 ? 'positive' : 'neutral'}
            icon={<ShowChartIcon color="primary" />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="最大回撤"
            value={metrics.basic_metrics.max_drawdown}
            unit="%"
            color="negative"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* 累计收益曲线 */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                累计收益曲线
              </Typography>
              <Box sx={{ height: 400 }}>
                <Line
                  data={cumulativeReturnsData}
                  margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                  xScale={{ type: 'point' }}
                  yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
                  yFormat=" >+.2f"
                  axisTop={null}
                  axisRight={null}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: -45,
                    legend: '日期',
                    legendOffset: 40,
                    legendPosition: 'middle'
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: '累计收益 (%)',
                    legendOffset: -50,
                    legendPosition: 'middle'
                  }}
                  enablePoints={false}
                  enableGridX={false}
                  enableArea={true}
                  areaOpacity={0.1}
                  colors={{ scheme: 'category10' }}
                  lineWidth={2}
                  useMesh={true}
                  tooltip={({ point }) => (
                    <Box sx={{
                      background: 'white',
                      padding: '5px 10px',
                      border: '1px solid #ccc',
                      borderRadius: '4px'
                    }}>
                      <Typography variant="caption">
                        {point.data.x}: {point.data.y}%
                      </Typography>
                    </Box>
                  )}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 风险指标 */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                风险指标
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell>波动率</TableCell>
                      <TableCell align="right">
                        {metrics.basic_metrics.volatility}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>索提诺比率</TableCell>
                      <TableCell align="right">
                        {metrics.risk_metrics.sortino_ratio}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>卡玛比率</TableCell>
                      <TableCell align="right">
                        {metrics.basic_metrics.calmar_ratio}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>VaR (95%)</TableCell>
                      <TableCell align="right">
                        {metrics.risk_metrics.var_95}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>CVaR (95%)</TableCell>
                      <TableCell align="right">
                        {metrics.risk_metrics.cvar_95}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Beta</TableCell>
                      <TableCell align="right">
                        {metrics.risk_metrics.beta}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Alpha</TableCell>
                      <TableCell align="right">
                        {metrics.risk_metrics.alpha}%
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* 回撤图 */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                动态回撤
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line
                  data={drawdownData}
                  margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                  xScale={{ type: 'point' }}
                  yScale={{ type: 'linear', min: 'auto', max: 0 }}
                  yFormat=" >+.2f"
                  axisTop={null}
                  axisRight={null}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: -45,
                    legend: '日期',
                    legendOffset: 40,
                    legendPosition: 'middle'
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: '回撤 (%)',
                    legendOffset: -50,
                    legendPosition: 'middle'
                  }}
                  enablePoints={false}
                  enableGridX={false}
                  enableArea={true}
                  areaOpacity={0.3}
                  colors={['#ff5252']}
                  lineWidth={2}
                  useMesh={true}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 月度收益 */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                月度收益分布
              </Typography>
              <Box sx={{ height: 300 }}>
                <Bar
                  data={monthlyReturnsData}
                  keys={['return']}
                  indexBy="month"
                  margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
                  padding={0.3}
                  valueScale={{ type: 'linear' }}
                  indexScale={{ type: 'band', round: true }}
                  colors={(bar) => bar.data.return > 0 ? '#4caf50' : '#f44336'}
                  borderColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
                  axisTop={null}
                  axisRight={null}
                  axisBottom={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: -45,
                    legend: '月份',
                    legendPosition: 'middle',
                    legendOffset: 50
                  }}
                  axisLeft={{
                    tickSize: 5,
                    tickPadding: 5,
                    tickRotation: 0,
                    legend: '收益率 (%)',
                    legendPosition: 'middle',
                    legendOffset: -50
                  }}
                  labelSkipWidth={12}
                  labelSkipHeight={12}
                  labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* 交易统计 */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                交易统计
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">总交易次数</Typography>
                  <Typography variant="h6">{metrics.trading_metrics.total_trades}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">胜率</Typography>
                  <Typography variant="h6" color={metrics.trading_metrics.win_rate > 50 ? 'success.main' : 'error.main'}>
                    {metrics.trading_metrics.win_rate}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">盈亏比</Typography>
                  <Typography variant="h6">{metrics.trading_metrics.profit_loss_ratio}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">平均交易收益</Typography>
                  <Typography variant="h6">{metrics.trading_metrics.avg_trade_return}%</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最大连续盈利</Typography>
                  <Typography variant="h6" color="success.main">
                    {metrics.trading_metrics.max_consecutive_wins}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最大连续亏损</Typography>
                  <Typography variant="h6" color="error.main">
                    {metrics.trading_metrics.max_consecutive_losses}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 业绩极值 */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                业绩极值
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最佳单日</Typography>
                  <Typography variant="h6" color="success.main">
                    +{metrics.performance_metrics.best_day}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最差单日</Typography>
                  <Typography variant="h6" color="error.main">
                    {metrics.performance_metrics.worst_day}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最佳月度</Typography>
                  <Typography variant="h6" color="success.main">
                    +{metrics.performance_metrics.best_month}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">最差月度</Typography>
                  <Typography variant="h6" color="error.main">
                    {metrics.performance_metrics.worst_month}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">盈利月份</Typography>
                  <Typography variant="h6">{metrics.performance_metrics.positive_months}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="textSecondary">亏损月份</Typography>
                  <Typography variant="h6">{metrics.performance_metrics.negative_months}</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* 最近交易记录 */}
        {trades && trades.length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  交易记录 (最近50条)
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>日期</TableCell>
                        <TableCell>代码</TableCell>
                        <TableCell>操作</TableCell>
                        <TableCell align="right">数量</TableCell>
                        <TableCell align="right">价格</TableCell>
                        <TableCell align="right">金额</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trades.slice(0, 10).map((trade, index) => (
                        <TableRow key={index}>
                          <TableCell>{trade.date}</TableCell>
                          <TableCell>{trade.symbol}</TableCell>
                          <TableCell>
                            <Chip
                              label={trade.action}
                              size="small"
                              color={trade.action === 'BUY' ? 'success' : 'error'}
                            />
                          </TableCell>
                          <TableCell align="right">{trade.quantity}</TableCell>
                          <TableCell align="right">${trade.price}</TableCell>
                          <TableCell align="right">${trade.amount}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default EnhancedBacktestResults;