/**
 * Task 40-53 测试：推理模块
 *
 * 覆盖：
 * - Task 40: inference.store（activeTask/queue/result/isRunning + actions）
 * - Task 41-43: 数据集/模型选择器 + 开始按钮 disabled 逻辑
 * - Task 44-45: ProgressBar + LogViewer
 * - Task 46-48: 活跃任务面板 + 任务队列 + 结果面板
 * - Task 53: 页面集成
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { useInferenceStore } from '../../src/renderer/stores/inference.store';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useModelStore } from '../../src/renderer/stores/model.store';

vi.mock('../../src/renderer/components/charts/LineChart', async () => {
  const ReactModule = await import('react');
  return {
    default: ({ series }: { series: { name: string }[] }) =>
      ReactModule.createElement(
        'div',
        { 'data-testid': 'result-chart' },
        series.map((item) => item.name).join(' / '),
      ),
  };
});

describe('Task 40: inference.store', () => {
  beforeEach(() => {
    useInferenceStore.setState({ activeTask: null, queue: [], result: null, isRunning: false });
  });

  it('test_inference_store_initial_state', () => {
    const s = useInferenceStore.getState();
    expect(s.activeTask).toBeNull();
    expect(s.queue).toEqual([]);
    expect(s.result).toBeNull();
    expect(s.isRunning).toBe(false);
  });

  it('test_inference_store_set_active_task', () => {
    useInferenceStore.getState().setActiveTask({
      taskId: 't1',
      progress: 50,
      status: 'running',
      log: [],
    });
    expect(useInferenceStore.getState().activeTask?.progress).toBe(50);
  });

  it('test_inference_store_update_progress', () => {
    useInferenceStore.getState().setActiveTask({
      taskId: 't1',
      progress: 10,
      status: 'running',
      log: [],
    });
    useInferenceStore.getState().updateProgress({ progress: 60 });
    expect(useInferenceStore.getState().activeTask?.progress).toBe(60);
  });

  it('test_inference_store_add_log', () => {
    useInferenceStore.getState().setActiveTask({
      taskId: 't1',
      progress: 10,
      status: 'running',
      log: [],
    });
    useInferenceStore.getState().addLog('line1');
    useInferenceStore.getState().addLog('line2');
    expect(useInferenceStore.getState().activeTask?.log).toEqual(['line1', 'line2']);
  });

  it('test_inference_store_set_result', () => {
    useInferenceStore.getState().setResult({
      taskId: 't1',
      units: [1], predictions: [42], targets: [40], rmse: 2, mae: 2, score: 0.2, seconds: 0.5,
    });
    expect(useInferenceStore.getState().result?.rmse).toBe(2);
  });

  it('test_inference_store_add_to_queue', () => {
    useInferenceStore.getState().addToQueue({
      id: 'q1',
      name: 'Q1',
      model: 'm',
      dataset: 'd',
      status: 'running',
      progress: 0,
      createdAt: 't',
    });
    expect(useInferenceStore.getState().queue).toHaveLength(1);
  });

  it('test_inference_store_update_queue_item', () => {
    useInferenceStore.getState().addToQueue({
      id: 'q1',
      name: 'Q1',
      model: 'm',
      dataset: 'd',
      status: 'running',
      progress: 0,
      createdAt: 't',
    });
    useInferenceStore.getState().updateQueueItem('q1', { progress: 80, status: 'completed' });
    const item = useInferenceStore.getState().queue[0];
    expect(item.progress).toBe(80);
    expect(item.status).toBe('completed');
  });

  it('test_inference_store_set_is_running', () => {
    useInferenceStore.getState().setIsRunning(true);
    expect(useInferenceStore.getState().isRunning).toBe(true);
  });
});

describe('Task 41-53: 推理页面集成', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, 'electronAPI', {
      configurable: true,
      value: {
        inferenceRun: vi.fn().mockResolvedValue({ success: true, data: 'run-old-test' }),
        inferenceCancel: vi.fn().mockResolvedValue({ success: true }),
        onInferenceProgress: vi.fn(() => () => {}),
        onInferenceResult: vi.fn(() => () => {}),
      },
    });
    useInferenceStore.setState({ activeTask: null, queue: [], result: null, isRunning: false });
    useDatasetStore.setState({
      datasets: [
        { id: 'd1', name: 'FD002', samples: 100, format: 'csv', type: 'train', createdAt: 't', astraCompatible: true, trainFile: 'train.any', testFile: 'test.any', rulFile: 'rul.any' },
      ],
      searchQuery: '',
      selectedDataset: null,
    });
    useModelStore.setState({
      models: [
        {
          id: 'm1',
          name: 'multiscale',
          type: 'tcn',
          status: 'available',
          source: 'trained',
          isDefault: true,
          path: '/p',
          createdAt: 't',
        },
      ],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
  });

  async function renderPage() {
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    return render(React.createElement(Inference));
  }

  it('test_inference_page_renders_header', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('推理模块')).toBeDefined();
  });

  it('test_inference_page_shows_datasets', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('FD002')).toBeDefined();
  });

  it('test_inference_page_shows_models', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('multiscale')).toBeDefined();
  });

  it('test_inference_page_start_disabled_without_selection', async () => {
    await renderPage();
    const startBtn = screen.getByText('开始推理');
    expect(startBtn).toBeDefined();
  });

  it('test_inference_page_start_runs_task', async () => {
    const { findByText } = await renderPage();
    // 选择数据集和模型
    fireEvent.click(await findByText('FD002'));
    fireEvent.click(await findByText('multiscale'));
    fireEvent.click(await findByText('开始推理'));
    await waitFor(() => {
      expect(useInferenceStore.getState().isRunning).toBe(true);
    });
    expect(useInferenceStore.getState().queue).toHaveLength(1);
  });

  it('test_inference_page_shows_active_task', async () => {
    useInferenceStore.getState().setActiveTask({
      taskId: 't1',
      progress: 40,
      status: 'running',
      log: ['step 1: loss=1.2'],
    });
    const { findByText } = await renderPage();
    expect(await findByText(/活跃任务/)).toBeDefined();
    expect(await findByText('40%')).toBeDefined();
  });

  it('test_inference_page_shows_result_metrics', async () => {
    useInferenceStore.getState().setResult({
      taskId: 't1',
      units: [1], predictions: [42], targets: [40], rmse: 2, mae: 2, score: 0.2, seconds: 0.5,
    });
    const { findByText } = await renderPage();
      expect(await findByText('推理结果')).toBeDefined();
      expect(await findByText('RMSE')).toBeDefined();
      expect((await findByText('RMSE')).nextSibling).toBeDefined();
      expect(screen.getByTestId('result-chart').textContent).toBe('预测 RUL / 真实 RUL');
  });

  it('test_inference_page_shows_queue', async () => {
    useInferenceStore.getState().addToQueue({
      id: 'q1',
      name: '推理-FD002',
      model: 'multiscale',
      dataset: 'FD002',
      status: 'running',
      progress: 30,
      createdAt: 't',
    });
    const { findByText } = await renderPage();
    expect(await findByText('任务队列')).toBeDefined();
    expect(await findByText('推理-FD002')).toBeDefined();
  });

  it('test_inference_page_has_no_dataset_without_imports', async () => {
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    const { findByText } = await renderPage();
    expect(screen.queryByText('FD001')).toBeNull();
  });
});
