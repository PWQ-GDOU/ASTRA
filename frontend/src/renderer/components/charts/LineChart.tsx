import React, { useMemo } from 'react';
import * as echarts from 'echarts';
import ChartContainer, {
  chartTooltip,
  chartGrid,
  chartCategoryAxis,
  chartValueAxis,
  CHART_COLORS,
} from './ChartContainer';

export interface LineChartSeries {
  name: string;
  data: number[];
  color?: string;
  areaStyle?: boolean;
}

interface LineChartProps {
  /** X 轴类目标签 (e.g., epoch 编号) */
  categories: (string | number)[];
  /** Y 轴数据序列（支持多条线） */
  series: LineChartSeries[];
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
  /** 是否平滑曲线 */
  smooth?: boolean;
  /** 是否显示面积填充 */
  showArea?: boolean;
}

const LineChart: React.FC<LineChartProps> = ({
  categories,
  series,
  height,
  loading,
  emptyText,
  className,
  smooth = false,
  showArea = false,
}) => {
  const isEmpty = categories.length === 0 || series.length === 0;

  const option = useMemo(() => {
    if (isEmpty) return null;

    const resolvedSeries = series.map((s, i) => ({
      ...s,
      color:
        s.color ??
        [
          CHART_COLORS.accent,
          CHART_COLORS.accentSecondary,
          CHART_COLORS.success,
          CHART_COLORS.warning,
        ][i % 8],
    }));

    const colors = resolvedSeries.map((s) => s.color!);

    return {
      color: colors,
      tooltip: {
        ...chartTooltip(),
        trigger: 'axis',
      } as unknown as Record<string, unknown>,
      legend: {
        show: resolvedSeries.length > 1,
        bottom: 0,
        textStyle: { color: CHART_COLORS.foregroundMuted, fontSize: 11 },
        icon: 'roundRect',
        itemWidth: 10,
        itemHeight: 10,
      },
      grid: chartGrid(16, 24, resolvedSeries.length > 1 ? 40 : 28, 48),
      xAxis: chartCategoryAxis(categories),
      yAxis: chartValueAxis(),
      series: resolvedSeries.map((s) => ({
        name: s.name,
        type: 'line' as const,
        data: s.data,
        smooth,
        symbol: 'circle',
        symbolSize: 5,
        lineStyle: { width: 2, color: s.color },
        itemStyle: { color: s.color },
        areaStyle: showArea
          ? {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: s.color + '40' },
                { offset: 1, color: s.color + '05' },
              ]),
            }
          : undefined,
      })),
    };
  }, [categories, series, smooth, showArea, isEmpty]);

  if (isEmpty) {
    return (
      <div
        className="flex items-center justify-center"
        style={{ height: height ?? 260, color: CHART_COLORS.foregroundDim, fontSize: 14 }}
      >
        <span>{emptyText ?? '暂无数据'}</span>
      </div>
    );
  }

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

export default LineChart;
