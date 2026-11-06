import React, { useMemo } from 'react';
import * as echarts from 'echarts';
import ChartContainer, {
  chartTooltip,
  chartGrid,
  chartCategoryAxis,
  chartValueAxis,
  CHART_PALETTE,
} from './ChartContainer';

interface BarChartProps {
  /** X 轴类目标签 */
  categories: (string | number)[];
  /** Y 轴数据 */
  data: number[];
  /** 系列名称 */
  name?: string;
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
  /** 柱状图方向: 默认 'vertical' */
  direction?: 'vertical' | 'horizontal';
  /** 是否使用渐变填充 */
  gradient?: boolean;
}

const BarChart: React.FC<BarChartProps> = ({
  categories,
  data,
  name = '数值',
  height,
  loading,
  emptyText,
  className,
  direction = 'vertical',
  gradient = true,
}) => {
  const option = useMemo(() => {
    if (categories.length === 0 || data.length === 0) return null;

    const isHorizontal = direction === 'horizontal';

    const barColors = data.map((_v, i) => {
      if (!gradient) return CHART_PALETTE[i % CHART_PALETTE.length];
      return new echarts.graphic.LinearGradient(
        isHorizontal ? 0 : 0,
        isHorizontal ? 0 : 0,
        isHorizontal ? 1 : 0,
        isHorizontal ? 0 : 1,
        [
          { offset: 0, color: CHART_PALETTE[i % CHART_PALETTE.length] },
          { offset: 1, color: CHART_PALETTE[(i + 1) % CHART_PALETTE.length] },
        ],
      );
    });

    const xConfig = isHorizontal ? chartValueAxis(name) : chartCategoryAxis(categories);
    const yConfig = isHorizontal ? chartCategoryAxis(categories) : chartValueAxis(name);

    return {
      color: CHART_PALETTE,
      tooltip: {
        ...chartTooltip(),
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      } as unknown as Record<string, unknown>,
      grid: chartGrid(16, 32, 28, isHorizontal ? 80 : 48),
      xAxis: xConfig,
      yAxis: yConfig,
      series: [
        {
          name,
          type: 'bar' as const,
          data: data.map((v, i) => ({
            value: v,
            itemStyle: {
              color: barColors[i],
              borderRadius: direction === 'vertical' ? [4, 4, 0, 0] : [0, 4, 4, 0],
            },
          })),
          barWidth: direction === 'vertical' ? '50%' : '60%',
          emphasis: {
            itemStyle: { shadowBlur: 10, shadowColor: 'rgba(99, 102, 241, 0.5)' },
          },
        },
      ],
    };
  }, [categories, data, name, direction, gradient]);

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

export default BarChart;
