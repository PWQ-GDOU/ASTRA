/**
 * Task 158 测试：Training 页面接线真实后端
 *
 * 验证 Training.tsx 的 handleStartTraining 调用 electronAPI.trainingStartLocal，
 * 并监听 onTrainingProgress / onTrainingResult。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

// Mock electronAPI
const mockTrainingStartLocal = vi.fn();
const mockTrainingStop = vi.fn();
const mockOnTrainingProgress = vi.fn(() => () => {});
const mockOnTrainingResult = vi.fn(() => () => {});
const mockSystemCudaStatus = vi.fn(async () => ({
  success: true,
  data: { cudaAvailable: true, gpuName: 'RTX 4090' },
}));

(window as unknown as Record<string, unknown>).electronAPI = {
  trainingStartLocal: mockTrainingStartLocal,
  trainingStop: mockTrainingStop,
  onTrainingProgress: mockOnTrainingProgress,
  onTrainingResult: mockOnTrainingResult,
  systemCudaStatus: mockSystemCudaStatus,
};

// Reset stores
import { useTrainingStore } from '../../src/renderer/stores/training.store';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useModelStore } from '../../src/renderer/stores/model.store';

describe('Task 158: Training 页面接线真实后端', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockTrainingStartLocal.mockReset();
    mockTrainingStartLocal.mockResolvedValue({ success: true, data: 'run-1' });
    mockTrainingStop.mockResolvedValue({ success: true, data: undefined });

    useTrainingStore.setState({
      mode: 'local',
      config: {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'my-run',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
      progress: null,
      isRunning: false,
    });
    useDatasetStore.setState({
      datasets: [
        { id: 'd1', name: 'FD002', samples: 100, format: 'csv', type: 'train', createdAt: 't' },
      ],
    });
    useModelStore.setState({ models: [] });
  });

  async function renderTraining() {
    const { default: Training } = await import('../../src/renderer/pages/Training');
    return render(React.createElement(Training));
  }

  it('test_training_page_calls_ipc_on_start', async () => {
    const { findByText } = await renderTraining();
    // 等待页面渲染完成
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    await waitFor(() => {
      expect(mockTrainingStartLocal).toHaveBeenCalled();
    });
  });

  it('test_training_page_listens_progress', async () => {
    await renderTraining();
    expect(mockOnTrainingProgress).toHaveBeenCalled();
  });

  it('test_training_page_listens_result', async () => {
    await renderTraining();
    expect(mockOnTrainingResult).toHaveBeenCalled();
  });

  it('test_training_page_passes_config_to_ipc', async () => {
    const { findByText } = await renderTraining();
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    await waitFor(() => {
      const cfg = mockTrainingStartLocal.mock.calls[0][0];
      expect(cfg.dataset).toBe('FD002');
      expect(cfg.hyperParams.epochs).toBe(10);
    });
  });

  it('test_training_page_progress_updates_ui', async () => {
    const { findByText } = await renderTraining();
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    // 模拟 progress 回调
    const progressCb = mockOnTrainingProgress.mock.calls[0][0];
    progressCb({ epoch: 3, step: 300, loss: 1.5, valLoss: 1.6 });
    await waitFor(() => {
      expect(screen.getByText('3 / 10')).toBeDefined();
    });
  });

  it('stops the running state and shows backend failures', async () => {
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('开始训练'));
    const progressCb = mockOnTrainingProgress.mock.calls[0][0];
    progressCb({
      epoch: 0,
      step: 0,
      loss: 0,
      status: 'failed',
      log: 'ASTRA 训练失败（退出码 2）',
    });

    await waitFor(() => {
      expect(useTrainingStore.getState().isRunning).toBe(false);
      expect(screen.getByText(/退出码 2/)).toBeDefined();
    });
  });

  it('test_training_page_stop_calls_ipc', async () => {
    useTrainingStore.setState({ isRunning: true });
    const { findByText } = await renderTraining();
    const stopBtn = await findByText('停止');
    fireEvent.click(stopBtn);
    await waitFor(() => {
      expect(mockTrainingStop).toHaveBeenCalled();
    });
  });

  it('test_training_page_start_validates', async () => {
    useTrainingStore.setState({
      config: {
        dataset: '',
        modelArch: 'multiscale',
        outputName: '',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    });
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    const { findByText } = await renderTraining();
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    expect(alertSpy).toHaveBeenCalled();
    expect(mockTrainingStartLocal).not.toHaveBeenCalled();
    alertSpy.mockRestore();
  });

  it('test_training_page_pure_training_no_output_name', async () => {
    // 纯训练：无输出模型名也能启动
    useTrainingStore.setState({
      config: {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: '',
        sensorMode: 'all24',
        seqLen: 60,
        rulCap: 125,
        seeds: '42',
        patience: 60,
        device: 'cuda',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    });
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    const { findByText } = await renderTraining();
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    await waitFor(() => {
      expect(mockTrainingStartLocal).toHaveBeenCalled();
    });
    expect(alertSpy).not.toHaveBeenCalled();
    alertSpy.mockRestore();
  });

  it('test_training_page_passes_custom_config', async () => {
    useTrainingStore.setState({
      config: {
        dataset: 'FD004',
        modelArch: 'condition',
        outputName: '',
        sensorMode: 'all24',
        seqLen: 30,
        rulCap: 150,
        seeds: '42,123',
        patience: 20,
        device: 'cpu',
        hyperParams: { epochs: 5, batchSize: 16, learningRate: 0.0001, optimizer: 'adam' },
      },
    });
    const { findByText } = await renderTraining();
    const startBtn = await findByText('开始训练');
    fireEvent.click(startBtn);
    await waitFor(() => {
      const cfg = mockTrainingStartLocal.mock.calls[0][0];
      expect(cfg.dataset).toBe('FD004');
      expect(cfg.modelArch).toBe('condition');
      expect(cfg.seqLen).toBe(30);
      expect(cfg.rulCap).toBe(150);
      expect(cfg.seeds).toBe('42,123');
      expect(cfg.patience).toBe(20);
      expect(cfg.device).toBe('cpu');
      expect(cfg.hyperParams.epochs).toBe(5);
    });
  });

  it('allows typing decimal parameters before committing them', async () => {
    const { default: Training } = await import('../../src/renderer/pages/Training');
    render(React.createElement(Training));
    const input = screen.getByDisplayValue('0.001');
    fireEvent.change(input, { target: { value: '' } });
    expect((input as HTMLInputElement).value).toBe('');
    fireEvent.change(input, { target: { value: '0.0005' } });
    fireEvent.blur(input);
    expect(useTrainingStore.getState().config.hyperParams.learningRate).toBe(0.0005);
  });
});
