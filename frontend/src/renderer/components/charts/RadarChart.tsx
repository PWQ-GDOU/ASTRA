import React, { useMemo } from 'react';
import ChartContainer, { chartTooltip, CHART_COLORS, CHART_PALETTE } from './ChartContainer';

export interface RadarIndicator {
  name: string;
  max: number;
}

export interface RadarSeriesItem {
  name: string;
  data: number[];
}

interface RadarChartProps {
  indicators: RadarIndicator[];
  series: RadarSeriesItem[];
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
  /** 多边形形状: 'polygon' | 'circle', 默认 'polygon' */
  shape?: 'polygon' | 'circle';
}

const RadarChart: React.FC<RadarChartProps> = ({
  indicators,
  series,
  height,
  loading,
  emptyText,
  className,
  shape = 'polygon',
}) => {
  const option = useMemo(() => {
    if (indicators.length === 0 || series.length === 0) return null;

    return {
      color: CHART_PALETTE,
      tooltip: {
        ...chartTooltip(),
        trigger: 'item',
      } as unknown as Record<string, unknown>,
      legend: {
        show: series.length > 1,
        bottom: 0,
        textStyle: { color: CHART_COLORS.foregroundMuted, fontSize: 11 },
        icon: 'roundRect',
        itemWidth: 10,
        itemHeight: 10,
      },
      radar: {
        center: ['50%', '48%'],
        radius: '65%',
        indicator: indicators,
        shape,
        axisName: {
          color: CHART_COLORS.foregroundMuted,
          fontSize: 11,
        },
        axisLine: { lineStyle: { color: CHART_COLORS.border } },
        splitLine: { lineStyle: { color: CHART_COLORS.borderLight } },
        splitArea: {
          areaStyle: {
            color: ['rgba(99, 102, 241, 0.02)', 'rgba(99, 102, 241, 0.04)'],
          },
        },
      },
      series: [
        {
          type: 'radar' as const,
          data: series.map((s) => ({
            name: s.name,
            value: s.data,
            lineStyle: { width: 2 },
            areaStyle: { opacity: 0.15 },
            symbol: 'circle',
            symbolSize: 4,
          })),
        },
      ],
    };
  }, [indicators, series, shape]);

  return (
    <ChartContainer
      option={option}
      height={height}
      loading={loading}
      emptyText={emptyText}
      className={className}
    />
  );
};

export default RadarChart;
