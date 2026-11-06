import React, { useMemo } from 'react';
import ChartContainer, { chartTooltip, CHART_COLORS, CHART_PALETTE } from './ChartContainer';

export interface PieDataItem {
  name: string;
  value: number;
}

interface PieChartProps {
  data: PieDataItem[];
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
  /** 饼图半径, 默认 '70%' */
  radius?: string | [string, string];
  /** 是否显示为环形图 */
  doughnut?: boolean;
  /** 是否显示标签 */
  showLabel?: boolean;
  /** 是否显示图例 */
  showLegend?: boolean;
  /** 圆心, 默认 ['50%', '50%'] */
  center?: [string, string];
}

const PieChart: React.FC<PieChartProps> = ({
  data,
  height,
  loading,
  emptyText,
  className,
  radius,
  doughnut = false,
  showLabel = true,
  showLegend = true,
  center,
}) => {
  const option = useMemo(() => {
    if (data.length === 0) return null;

    const resolvedRadius: string | [string, string] = radius ?? (doughnut ? ['50%', '70%'] : '70%');

    return {
      color: CHART_PALETTE,
      tooltip: {
        ...chartTooltip(),
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
      } as unknown as Record<string, unknown>,
      legend: showLegend
        ? {
            bottom: 0,
            textStyle: { color: CHART_COLORS.foregroundMuted, fontSize: 11 },
            itemWidth: 8,
            itemHeight: 8,
            itemGap: 16,
          }
        : undefined,
      series: [
        {
          type: 'pie' as const,
          radius: resolvedRadius,
          center: center ?? ['50%', '50%'],
          avoidLabelOverlap: false,
          itemStyle: {
            borderColor: 'rgba(12, 12, 20, 0.8)',
            borderWidth: 2,
            borderRadius: 2,
          },
          label: showLabel
            ? {
                show: true,
                position: 'outside',
                formatter: '{b}\n{d}%',
                color: CHART_COLORS.foregroundMuted,
                fontSize: 11,
              }
            : { show: false },
          emphasis: {
            label: { show: true, fontSize: 14, fontWeight: 'bold' },
            itemStyle: { shadowBlur: 20, shadowColor: 'rgba(0, 0, 0, 0.5)' },
          },
          data,
        },
      ],
    };
  }, [data, radius, doughnut, showLabel, showLegend, center]);

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

export default PieChart;
