import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useInferenceStore } from '../../src/renderer/stores/inference.store';
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

describe('ASTRA inference integration', () => {
  const inferenceRun = vi.fn();
  let progressCallback: ((event: unknown) => void) | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    inferenceRun.mockResolvedValue({ success: true, data: 'run-1' });
    Object.defineProperty(window, 'electronAPI', {
      configurable: true,
      value: {
        inferenceRun,
        inferenceCancel: vi.fn().mockResolvedValue({ success: true }),
        onInferenceProgress: vi.fn((callback: (event: unknown) => void) => {
          progressCallback = callback;
          return () => {};
        }),
        onInferenceResult: vi.fn(() => () => {}),
      },
    });
    useInferenceStore.setState({ activeTask: null, queue: [], result: null, isRunning: false });
    useDatasetStore.setState({
      datasets: [{ id: 'fd2', name: 'FD002', samples: 100, format: 'txt', type: 'test', createdAt: 't', astraCompatible: true, trainFile: 'train.any', testFile: 'test.any', rulFile: 'rul.any' }],
      searchQuery: '',
      selectedDataset: null,
    });
    useModelStore.setState({
      models: [{ id: 'm1', name: 'checkpoint.pt', status: 'available', source: 'trained', isDefault: true, path: 'E:/models/checkpoint.pt', createdAt: 't' }],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
  });

  it('sends selected checkpoint and inference parameters to Electron', async () => {
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    render(<Inference />);
    fireEvent.click(screen.getByText('FD002'));
    fireEvent.click(screen.getByText('checkpoint.pt'));
    fireEvent.change(screen.getByLabelText('批大小'), { target: { value: '128' } });
    fireEvent.change(screen.getByLabelText('计算设备'), { target: { value: 'cuda:1' } });
    fireEvent.click(screen.getByText('开始推理'));

    await waitFor(() => expect(inferenceRun).toHaveBeenCalledWith({
      datasetId: 'fd2',
      modelId: 'm1',
      batchSize: 128,
      device: 'cuda:1',
      applyRulCap: true,
      rulCap: 125,
    }));
  });

  it('renders RUL regression metrics instead of classification metrics', async () => {
    useInferenceStore.setState({
      result: {
        taskId: 'r1',
        units: [101, 102, 103],
        predictions: [42, 50, 60],
        targets: [40, 55, 58],
        rmse: 3.2,
        mae: 3,
        score: 0.2,
        seconds: 0.5,
      },
    });
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    render(<Inference />);
    expect(screen.getByText('RMSE')).toBeDefined();
    expect(screen.getByText('MAE')).toBeDefined();
    expect(screen.getAllByText('预测 RUL').length).toBeGreaterThan(0);
    expect(screen.queryByText('准确率')).toBeNull();
    expect(screen.getByText('预测与真实 RUL')).toBeDefined();
    expect(screen.getByText('样本误差明细')).toBeDefined();
    expect(screen.getByText('最大绝对误差')).toBeDefined();
    expect(screen.getByTestId('result-chart').textContent).toBe('预测 RUL / 真实 RUL');
    expect(screen.getByRole('cell', { name: '102' })).toBeDefined();
    expect(screen.getByRole('cell', { name: '-5.000' })).toBeDefined();
  });

  it('renders prediction-only results without pretending target errors are available', async () => {
    useInferenceStore.setState({
      result: { taskId: 'r2', units: [1, 2], predictions: [20, 21], seconds: 0.2 },
    });
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    render(<Inference />);

    expect(screen.getByText('预测与真实 RUL')).toBeDefined();
    expect(screen.getByTestId('result-chart').textContent).toBe('预测 RUL');
    expect(screen.getByText('当前结果没有真实标签，无法计算逐样本误差。')).toBeDefined();
  });

  it('shows the backend error and releases the running state', async () => {
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    render(<Inference />);
    fireEvent.click(screen.getByText('FD002'));
    fireEvent.click(screen.getByText('checkpoint.pt'));
    fireEvent.click(screen.getByText('开始推理'));
    await waitFor(() => expect(useInferenceStore.getState().isRunning).toBe(true));

    progressCallback?.({
      taskId: 'run-1', progress: 0, status: 'failed', elapsed: 0.1,
      message: 'checkpoint incompatible',
    });

    await waitFor(() => {
      expect(useInferenceStore.getState().isRunning).toBe(false);
      expect(screen.getByText(/checkpoint incompatible/)).toBeDefined();
    });
  });
});
