import React, { useRef, useEffect } from 'react';
import * as echarts from 'echarts';

/* ===== Spacecraft 图表主题常量 ===== */
export const CHART_COLORS = {
  accent: '#6366f1',
  accentHover: '#818cf8',
  accentGlow: 'rgba(99, 102, 241, 0.4)',
  accentSecondary: '#06b6d4',
  accentSecondaryGlow: 'rgba(6, 182, 212, 0.4)',
  success: '#22c55e',
  successGlow: 'rgba(34, 197, 94, 0.4)',
  warning: '#f59e0b',
  warningGlow: 'rgba(245, 158, 11, 0.4)',
  danger: '#ef4444',
  dangerGlow: 'rgba(239, 68, 68, 0.4)',
  foreground: '#1a1a24',
  foregroundMuted: '#5a5a6e',
  foregroundDim: '#8a8a9e',
  border: 'rgba(0, 0, 0, 0.08)',
  borderLight: 'rgba(0, 0, 0, 0.05)',
  surface: 'rgba(255, 255, 255, 0.75)',
} as const;

export const CHART_TEXT_STYLE: Record<string, React.CSSProperties[keyof React.CSSProperties]> = {
  color: '#5a5a6e',
  fontSize: 12,
  fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
};

const CHART_PALETTE = [
  CHART_COLORS.accent,
  CHART_COLORS.accentSecondary,
  CHART_COLORS.success,
  CHART_COLORS.warning,
  CHART_COLORS.danger,
  '#a855f7',
  '#ec4899',
  '#f97316',
];

/* ===== 工具：创建双色渐变 ===== */
export function createVerticalGradient(
  topColor: string,
  bottomColor: string,
): echarts.graphic.LinearGradient {
  return new echarts.graphic.LinearGradient(0, 0, 0, 1, [
    { offset: 0, color: topColor },
    { offset: 1, color: bottomColor },
  ]);
}

/* ===== 基础 ChartContainer ===== */
interface ChartContainerProps {
  option: Record<string, unknown> | null;
  height?: number;
  loading?: boolean;
  emptyText?: string;
  className?: string;
}

const ChartContainer: React.FC<ChartContainerProps> = ({
  option,
  height = 260,
  loading = false,
  emptyText = '暂无数据',
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  /* init + dispose */
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    let instance = echarts.getInstanceByDom(node);
    if (!instance) {
      instance = echarts.init(node, undefined, {
        renderer: 'canvas',
      });
    }
    chartRef.current = instance;

    return () => {
      instance.dispose();
      chartRef.current = null;
    };
  }, []);

  /* setOption */
  useEffect(() => {
    if (chartRef.current && option) {
      chartRef.current.setOption(option, true);
    }
  }, [option]);

  /* resize via ResizeObserver */
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;

    const ro = new ResizeObserver(() => {
      chartRef.current?.resize();
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  /* loading */
  useEffect(() => {
    if (!chartRef.current) return;
    if (loading) {
      chartRef.current.showLoading('default', {
        text: '',
        color: CHART_COLORS.accent,
        maskColor: 'rgba(12, 12, 20, 0.6)',
        fontSize: 14,
        fontWeight: 'normal',
      });
    } else {
      chartRef.current.hideLoading();
    }
  }, [loading]);

  /* empty state */
  if (!option) {
    return (
      <div
        className={`flex items-center justify-center ${className}`}
        style={{ height, color: CHART_COLORS.foregroundDim, fontSize: 14 }}
      >
        <span>{emptyText}</span>
      </div>
    );
  }

  return <div ref={containerRef} className={className} style={{ width: '100%', height }} />;
};

export default ChartContainer;

/* ===== 共享工具函数 ===== */

/** 构建 ECharts tooltip 主题化配置 */
export function chartTooltip(): Record<string, unknown> {
  return {
    backgroundColor: 'rgba(20, 20, 40, 0.95)',
    borderColor: CHART_COLORS.border,
    borderWidth: 1,
    textStyle: { ...CHART_TEXT_STYLE, color: CHART_COLORS.foreground },
    extraCssText:
      'border-radius: 10px; backdrop-filter: blur(10px); box-shadow: 0 4px 24px rgba(0,0,0,0.5);',
  };
}

/** 构建 ECharts grid 主题化配置 */
export function chartGrid(top?: number, right?: number, bottom?: number, left?: number) {
  return {
    top: top ?? 20,
    right: right ?? 24,
    bottom: bottom ?? 32,
    left: left ?? 48,
    containLabel: false,
  };
}

/** Category 轴 (X 轴) 主题化 */
export function chartCategoryAxis(data: (string | number)[]): Record<string, unknown> {
  return {
    type: 'category',
    data,
    axisLine: { lineStyle: { color: CHART_COLORS.border } },
    axisTick: { show: false },
    axisLabel: { ...CHART_TEXT_STYLE, color: CHART_COLORS.foregroundMuted },
    splitLine: { show: false },
  };
}

/** Value 轴 (Y 轴) 主题化 */
export function chartValueAxis(name?: string): Record<string, unknown> {
  return {
    type: 'value',
    name,
    nameTextStyle: { ...CHART_TEXT_STYLE, color: CHART_COLORS.foregroundDim },
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { ...CHART_TEXT_STYLE, color: CHART_COLORS.foregroundMuted },
    splitLine: { lineStyle: { color: CHART_COLORS.borderLight } },
  };
}

/** 图表调色盘 */
export { CHART_PALETTE };
