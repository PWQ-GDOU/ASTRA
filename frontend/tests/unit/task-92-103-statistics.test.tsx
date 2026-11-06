/**
 * Task 92-103 测试：统计分析模块
 *
 * 覆盖：
 * - Task 92: statistics.store
 * - Task 101-102: StatisticsPage 渲染各类图表 + 空状态 + 骨架
 * - Task 94-99: 6 类图表组件（LineChart/AreaChart/PieChart/RadarChart/BarChart）
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { useStatisticsStore } from '../../src/renderer/stores/statistics.store';

// Mock echarts — jsdom 无法初始化真实图表
vi.mock('echarts', () => ({
  init: vi.fn(() => ({
    setOption: vi.fn(),
    resize: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
  })),
  getInstanceByDom: vi.fn(() => null),
  graphic: {
    LinearGradient: class {
      constructor() {}
    },
  },
}));

// jsdom 没有 ResizeObserver
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverMock);

describe('Task 92: statistics.store', () => {
  beforeEach(() => {
    useStatisticsStore.setState({ data: null, dateRange: null, isLoading: false });
  });

  it('test_stats_store_initial_null', () => {
    const s = useStatisticsStore.getState();
    expect(s.data).toBeNull();
    expect(s.dateRange).toBeNull();
    expect(s.isLoading).toBe(false);
  });

  it('test_stats_store_set_data', () => {
    useStatisticsStore.getState().setData({
      trainingTrend: [{ epoch: 1, loss: 2.0 }],
      inferenceAccuracyTrend: [{ date: 'd', accuracy: 0.9 }],
      datasetDistribution: [{ name: 'train', count: 10, type: 'train' }],
      modelPerformance: [{ name: 'm', accuracy: 0.9, speed: 10, size: 1, params: 100 }],
      taskStats: { completed: 1, failed: 2, running: 3, pending: 4 },
      modelStatusDist: [{ status: 'available', count: 5 }],
    });
    expect(useStatisticsStore.getState().data?.trainingTrend).toHaveLength(1);
  });

  it('test_stats_store_set_date_range', () => {
    useStatisticsStore.getState().setDateRange(['2026-01-01', '2026-08-01']);
    expect(useStatisticsStore.getState().dateRange).toEqual(['2026-01-01', '2026-08-01']);
  });

  it('test_stats_store_set_loading', () => {
    useStatisticsStore.getState().setLoading(true);
    expect(useStatisticsStore.getState().isLoading).toBe(true);
  });

  it('test_stats_store_clear_date_range', () => {
    useStatisticsStore.getState().setDateRange(['a', 'b']);
    useStatisticsStore.getState().setDateRange(null);
    expect(useStatisticsStore.getState().dateRange).toBeNull();
  });
});

describe('Task 101-102: 统计分析页面', () => {
  beforeEach(() => {
    useStatisticsStore.setState({
      data: {
        trainingTrend: [
          { epoch: 1, loss: 2.0, valLoss: 2.1 },
          { epoch: 2, loss: 1.5, valLoss: 1.6 },
        ],
        inferenceAccuracyTrend: [
          { date: 'd1', accuracy: 0.9 },
          { date: 'd2', accuracy: 0.92 },
        ],
        datasetDistribution: [
          { name: '训练集', count: 80, type: 'train' },
          { name: '测试集', count: 20, type: 'test' },
        ],
        modelPerformance: [{ name: 'multiscale', accuracy: 0.92, speed: 10, size: 1, params: 100 }],
        taskStats: { completed: 10, failed: 2, running: 3, pending: 1 },
        modelStatusDist: [
          { status: 'available', count: 5 },
          { status: 'training', count: 2 },
        ],
      },
      isLoading: false,
    });
  });

  async function renderPage() {
    const { default: Statistics } = await import('../../src/renderer/pages/Statistics');
    return render(React.createElement(Statistics));
  }

  it('test_stats_page_renders_header', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('统计分析')).toBeDefined();
  });

  it('test_stats_page_renders_training_trend_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText(/训练趋势/)).toBeDefined();
  });

  it('test_stats_page_renders_accuracy_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('推理准确率趋势')).toBeDefined();
  });

  it('test_stats_page_renders_dataset_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('数据集分布')).toBeDefined();
  });

  it('test_stats_page_renders_model_perf_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('模型性能对比')).toBeDefined();
  });

  it('test_stats_page_renders_task_stats_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('推理任务统计')).toBeDefined();
  });

  it('test_stats_page_renders_model_status_panel', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('模型状态分布')).toBeDefined();
  });

  it('test_stats_page_loading_shows_skeleton', async () => {
    useStatisticsStore.setState({ isLoading: true });
    await renderPage();
    // 骨架屏有动画元素
    expect(document.querySelectorAll('.animate-pulse, [class*="pulse"]').length).toBeGreaterThan(0);
  });

  it('test_stats_page_empty_shows_placeholder', async () => {
    useStatisticsStore.setState({ data: null, isLoading: false });
    await renderPage();
    expect(screen.getAllByText(/暂无数据|加载统计数据/).length).toBeGreaterThan(0);
  });
});
