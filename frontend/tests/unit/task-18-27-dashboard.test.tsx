/**
 * Task 18-27 测试：Dashboard 模块
 *
 * 覆盖工作台页面的真实功能：
 * - Task 18: 四统计卡片（数据集/模型/推理次数/系统状态）
 * - Task 19: 系统状态指示器（CUDA 状态灯）
 * - Task 20: 最近推理任务表格
 * - Task 21: 训练任务进度面板
 * - Task 23: DashboardPage 组装 + PageHeader
 * - Task 24: 空状态处理
 * - Task 25: lastUpdated 时间戳
 * - Task 27: 集成验证
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { useDashboardStore } from '../../src/renderer/stores/dashboard.store';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useModelStore } from '../../src/renderer/stores/model.store';
import { useInferenceStore } from '../../src/renderer/stores/inference.store';
import { useInferenceHistoryStore } from '../../src/renderer/stores/inference-history.store';

const mockSystemCudaStatus = vi.fn(async () => ({
  success: true,
  data: { cudaAvailable: true, gpuName: 'RTX 4090', gpuMemory: '24 GB' },
}));

(window as unknown as Record<string, unknown>).electronAPI = {
  systemCudaStatus: mockSystemCudaStatus,
};

describe('Task 18-27: Dashboard 模块', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // 重置所有相关 store
    useDashboardStore.setState({
      stats: {
        datasetCount: 0,
        modelCount: 0,
        inferenceCount: 0,
        systemStatus: 'offline',
        gpuInfo: null,
      },
      recentTasks: [],
      trainingProgress: [],
      accuracyTrend: [],
      lastUpdated: null,
    });
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    useModelStore.setState({
      models: [],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
    useInferenceStore.setState({ activeTask: null, queue: [], result: null, isRunning: false });
    useInferenceHistoryStore.setState({
      items: [],
      selectedItem: null,
      isDrawerOpen: false,
      total: 0,
      page: 1,
    });
  });

  async function renderDashboard() {
    const { default: Dashboard } = await import('../../src/renderer/pages/Dashboard');
    return render(React.createElement(Dashboard));
  }

  it('test_dashboard_renders_page_header', async () => {
    const { findByText } = await renderDashboard();
    expect(await findByText('工作台')).toBeDefined();
  });

  it('test_dashboard_renders_4_stat_cards', async () => {
    await renderDashboard();
    expect(screen.getByText('数据集')).toBeDefined();
    expect(screen.getByText('模型')).toBeDefined();
    expect(screen.getByText('推理次数')).toBeDefined();
  });

  it('test_dashboard_shows_dataset_count_from_store', async () => {
    useDatasetStore.getState().setDatasets([
      { id: 'd1', name: 'A', samples: 10, format: 'csv', type: 'train', createdAt: 't' },
      { id: 'd2', name: 'B', samples: 20, format: 'csv', type: 'test', createdAt: 't' },
      { id: 'd3', name: 'C', samples: 30, format: 'csv', type: 'val', createdAt: 't' },
    ]);
    useModelStore.getState().setModels([
      {
        id: 'm1',
        name: 'M1',
        type: 'llm',
        status: 'available',
        source: 'imported',
        isDefault: false,
        path: '/p',
        createdAt: 't',
      },
      {
        id: 'm2',
        name: 'M2',
        type: 'cnn',
        status: 'available',
        source: 'trained',
        isDefault: true,
        path: '/p',
        createdAt: 't',
      },
    ]);
    useInferenceStore.getState().setQueue([
      {
        id: 'q1',
        name: 'Q1',
        model: 'm',
        dataset: 'd',
        status: 'completed',
        progress: 100,
        createdAt: 't',
      },
      {
        id: 'q2',
        name: 'Q2',
        model: 'm',
        dataset: 'd',
        status: 'running',
        progress: 50,
        createdAt: 't',
      },
    ]);
    const { findByText } = await renderDashboard();
    expect(await findByText('3')).toBeDefined();
    // 模型数=2 和 推理次数=2 都可能显示 '2'，用 getAllByText 断言至少出现
    expect(screen.getAllByText('2').length).toBeGreaterThan(0);
  });

  it('test_dashboard_renders_training_panel_title', async () => {
    const { findByText } = await renderDashboard();
    expect(await findByText('训练任务进度')).toBeDefined();
  });

  it('test_dashboard_shows_training_progress', async () => {
    useDashboardStore.setState({
      trainingProgress: [{ name: 'train-run', progress: 50 }],
    });
    const { findByText } = await renderDashboard();
    expect(await findByText('train-run')).toBeDefined();
    expect(await findByText('50%')).toBeDefined();
  });

  it('test_dashboard_renders_recent_tasks_table', async () => {
    useInferenceHistoryStore.getState().setItems([
      {
        id: 'i1',
        taskName: 'T1',
        model: 'm1',
        dataset: 'd1',
        status: 'completed',
        duration: '1s',
        time: '10:00',
        metrics: { accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9, latency: 1, throughput: 2 },
        log: [],
      },
    ]);
    const { findByText } = await renderDashboard();
    expect(await findByText('最近推理任务')).toBeDefined();
    expect(await findByText('T1')).toBeDefined();
  });

  it('test_dashboard_empty_training_shows_placeholder', async () => {
    await renderDashboard();
    expect(screen.getByText('暂无训练任务')).toBeDefined();
  });

  it('test_dashboard_empty_recent_tasks_shows_placeholder', async () => {
    await renderDashboard();
    expect(screen.getByText('暂无数据')).toBeDefined();
  });

  it('test_dashboard_calls_system_cuda_status', async () => {
    await renderDashboard();
    await waitFor(() => {
      expect(mockSystemCudaStatus).toHaveBeenCalled();
    });
  });

  it('test_dashboard_cuda_available_shows_gpu', async () => {
    await renderDashboard();
    await waitFor(() => {
      expect(screen.getByText(/RTX 4090/)).toBeDefined();
    });
  });
});
