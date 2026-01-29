"""
ETF市场热力图React组件
展示板块ETF的收益率和因子数据
"""

import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Grid, Box, Tabs, Tab, Button, Select, MenuItem, FormControl, InputLabel, Chip } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import { Heatmap } from '@nivo/heatmap';
import { Line } from '@nivo/line';
import { Bar } from '@nivo/bar';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001/api';

const MarketHeatmap = () => {
  const [tabValue, setTabValue] = useState(0);
  const [etfData, setEtfData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [etfType, setEtfType] = useState('sector');
  const [sortBy, setSortBy] = useState('momentum_20d');

  useEffect(() => {
    fetchETFData();
  }, [etfType]);

  const fetchETFData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/market/etf-heatmap?sort_by=${sortBy}`);
      setEtfData(response.data.data);
    } catch (error) {
      console.error('Error fetching ETF data:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatHeatmapData = () => {
    const periods = ['1d', '5d', '10d', '20d', '60d', '120d', '250d'];
    return etfData.slice(0, 20).map(etf => ({
      id: etf.name,
      data: periods.map(period => ({
        x: period,
        y: etf.returns[period] || 0
      }))
    }));
  };

  const getColorForReturn = (value) => {
    if (value > 5) return '#1b5e20';
    if (value > 2) return '#388e3c';
    if (value > 0) return '#66bb6a';
    if (value > -2) return '#ffcdd2';
    if (value > -5) return '#e53935';
    return '#b71c1c';
  };

  const columns = [
    {
      field: 'name',
      headerName: 'ETF名称',
      width: 150,
      fixed: true
    },
    {
      field: '1d',
      headerName: '1日',
      width: 80,
      renderCell: (params) => (
        <Box sx={{
          color: params.value > 0 ? '#388e3c' : '#e53935',
          fontWeight: 'bold'
        }}>
          {params.value > 0 ? '+' : ''}{params.value?.toFixed(2)}%
        </Box>
      ),
    },
    {
      field: '5d',
      headerName: '5日',
      width: 80,
      renderCell: (params) => (
        <Box sx={{
          color: params.value > 0 ? '#388e3c' : '#e53935',
          fontWeight: 'bold'
        }}>
          {params.value > 0 ? '+' : ''}{params.value?.toFixed(2)}%
        </Box>
      ),
    },
    {
      field: '20d',
      headerName: '20日',
      width: 80,
      renderCell: (params) => (
        <Box sx={{
          color: params.value > 0 ? '#388e3c' : '#e53935',
          fontWeight: 'bold'
        }}>
          {params.value > 0 ? '+' : ''}{params.value?.toFixed(2)}%
        </Box>
      ),
    },
    {
      field: '60d',
      headerName: '60日',
      width: 80,
      renderCell: (params) => (
        <Box sx={{
          color: params.value > 0 ? '#388e3c' : '#e53935',
          fontWeight: 'bold'
        }}>
          {params.value > 0 ? '+' : ''}{params.value?.toFixed(2)}%
        </Box>
      ),
    },
    {
      field: 'momentum_20d',
      headerName: '动量20d',
      width: 100,
      renderCell: (params) => (
        <Box sx={{ fontWeight: 'bold' }}>
          {params.value?.toFixed(2)}
        </Box>
      ),
    },
    {
      field: 'volatility_20d',
      headerName: '波动率20d',
      width: 100,
      renderCell: (params) => (
        <Box>
          {params.value?.toFixed(2)}%
        </Box>
      ),
    },
    {
      field: 'risk_adj_momentum_20d',
      headerName: '风险调整动量',
      width: 120,
      renderCell: (params) => (
        <Box sx={{ fontWeight: 'bold' }}>
          {params.value?.toFixed(2)}
        </Box>
      ),
    },
  ];

  const rows = etfData.map((etf, index) => ({
    id: index,
    name: etf.name,
    '1d': etf.returns['1d'],
    '5d': etf.returns['5d'],
    '20d': etf.returns['20d'],
    '60d': etf.returns['60d'],
    momentum_20d: etf.factors.momentum_20d,
    volatility_20d: etf.factors.volatility_20d,
    risk_adj_momentum_20d: etf.factors.risk_adj_momentum_20d,
  }));

  return (
    <Card>
      <CardContent>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Typography variant="h5" gutterBottom>
            ETF市场热力图
          </Typography>
          <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
            <Tab label="数据表格" />
            <Tab label="热力图" />
            <Tab label="因子分析" />
          </Tabs>
        </Box>

        <Box sx={{ mb: 2, display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>ETF类型</InputLabel>
            <Select
              value={etfType}
              label="ETF类型"
              onChange={(e) => setEtfType(e.target.value)}
            >
              <MenuItem value="sector">行业ETF</MenuItem>
              <MenuItem value="broad">宽基指数</MenuItem>
              <MenuItem value="all">全部</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>排序</InputLabel>
            <Select
              value={sortBy}
              label="排序"
              onChange={(e) => {
                setSortBy(e.target.value);
                fetchETFData();
              }}
            >
              <MenuItem value="1d">1日涨跌</MenuItem>
              <MenuItem value="20d">20日涨跌</MenuItem>
              <MenuItem value="momentum_20d">20日动量</MenuItem>
              <MenuItem value="volatility_20d">波动率</MenuItem>
              <MenuItem value="risk_adj_momentum_20d">风险调整动量</MenuItem>
            </Select>
          </FormControl>

          <Button variant="outlined" onClick={fetchETFData}>
            刷新数据
          </Button>
        </Box>

        {tabValue === 0 && (
          <Box sx={{ height: 600 }}>
            <DataGrid
              rows={rows}
              columns={columns}
              pageSize={20}
              loading={loading}
              density="compact"
              disableSelectionOnClick
            />
          </Box>
        )}

        {tabValue === 1 && (
          <Box sx={{ height: 600 }}>
            <Heatmap
              data={formatHeatmapData()}
              margin={{ top: 60, right: 90, bottom: 60, left: 120 }}
              valueFormat=">+.2f"
              axisTop={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: -90,
                legend: '',
                legendOffset: 46
              }}
              axisRight={null}
              axisBottom={null}
              axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'ETF',
                legendPosition: 'middle',
                legendOffset: -100
              }}
              colors={{
                type: 'diverging',
                scheme: 'red_yellow_green',
                divergeAt: 0.5,
                min: -10,
                max: 10
              }}
              emptyColor="#555555"
              borderColor={{
                from: 'color',
                modifiers: [['darker', 0.6]]
              }}
              labelTextColor={{
                from: 'color',
                modifiers: [['darker', 2]]
              }}
              legends={[
                {
                  anchor: 'bottom',
                  translateX: 0,
                  translateY: 30,
                  length: 400,
                  thickness: 8,
                  direction: 'row',
                  tickPosition: 'after',
                  tickSize: 3,
                  tickSpacing: 4,
                  tickOverlap: false,
                  tickFormat: '>+.1f',
                  title: '收益率 (%)',
                  titleAlign: 'start',
                  titleOffset: 4
                }
              ]}
            />
          </Box>
        )}

        {tabValue === 2 && (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6">动量vs波动率</Typography>
                  <Box sx={{ height: 300 }}>
                    {/* 散点图展示动量与波动率关系 */}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6">风险调整后收益排名</Typography>
                  <Box sx={{ height: 300 }}>
                    <Bar
                      data={etfData.slice(0, 10).map(etf => ({
                        etf: etf.name.substring(0, 10),
                        value: etf.factors.risk_adj_momentum_20d
                      }))}
                      keys={['value']}
                      indexBy="etf"
                      margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
                      padding={0.3}
                      valueScale={{ type: 'linear' }}
                      indexScale={{ type: 'band', round: true }}
                      colors={{ scheme: 'nivo' }}
                      borderColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
                      axisTop={null}
                      axisRight={null}
                      axisBottom={{
                        tickSize: 5,
                        tickPadding: 5,
                        tickRotation: -45,
                        legend: 'ETF',
                        legendPosition: 'middle',
                        legendOffset: 50
                      }}
                      axisLeft={{
                        tickSize: 5,
                        tickPadding: 5,
                        tickRotation: 0,
                        legend: '风险调整动量',
                        legendPosition: 'middle',
                        legendOffset: -40
                      }}
                      labelSkipWidth={12}
                      labelSkipHeight={12}
                      labelTextColor={{ from: 'color', modifiers: [['darker', 1.6]] }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </CardContent>
    </Card>
  );
};

export default MarketHeatmap;