import React, { useMemo } from 'react';
import * as echarts from 'echarts';
import ChartContainer, {
  chartTooltip,
  chartGrid,
  chartCategoryAxis,
  chartValueAxis,
  CHART_PALETTE,
} from './ChartContainer';

interface AreaChartProps {
  /** X 轴类目标签 (e.g., 日期) */
  categories: (string | number)[];
  /** Y 轴数据 */
  data: number[];
  /** 系列名称 */
  name?: string;
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
  /** 是否平滑曲线 */
  smooth?: boolean;
}

const AreaChart: React.FC<AreaChartProps> = ({
  categories,
  data,
  name = '数值',
  height,
  loading,
  emptyText,
  className,
  smooth = true,
}) => {
  const option = useMemo(() => {
    if (categories.length === 0 || data.length === 0) return null;

    return {
      color: [CHART_PALETTE[0]],
      tooltip: {
        ...chartTooltip(),
        trigger: 'axis',
        axisPointer: { type: 'cross' },
      } as unknown as Record<string, unknown>,
      grid: chartGrid(16, 24, 28, 48),
      xAxis: chartCategoryAxis(categories),
      yAxis: chartValueAxis(name),
      series: [
        {
          name,
          type: 'line' as const,
          data,
          smooth,
          symbol: 'circle',
          symbolSize: 4,
          lineStyle: { width: 2, color: CHART_PALETTE[0] },
          itemStyle: { color: CHART_PALETTE[0] },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(99, 102, 241, 0.35)' },
              { offset: 0.6, color: 'rgba(99, 102, 241, 0.10)' },
              { offset: 1, color: 'rgba(99, 102, 241, 0.02)' },
            ]),
          },
        },
      ],
    };
  }, [categories, data, name, smooth]);

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

export default AreaChart;
