/**
 * Task 158b 测试：应用加载数据集到 store，训练/推理整单位可选
 *
 * 覆盖：
 * - 应用启动时调用 datasetList 加载数据集
 * - 训练页面数据集下拉显示已加载的数据集
 * - 推理页面数据集选择器显示已加载的数据集
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useTrainingStore } from '../../src/renderer/stores/training.store';
import { useModelStore } from '../../src/renderer/stores/model.store';
import { useInferenceStore } from '../../src/renderer/stores/inference.store';

const mockDatasetList = vi.fn();
let modelStatusCallback: ((model: unknown) => void) | undefined;

function setElectronApi() {
  (window as unknown as Record<string, unknown>).electronAPI = {
    datasetList: mockDatasetList,
    datasetImport: vi.fn(),
    modelList: vi.fn(async () => ({ success: true, data: [] })),
    onModelStatus: vi.fn((callback: (model: unknown) => void) => {
      modelStatusCallback = callback;
      return () => {};
    }),
    trainingStartLocal: vi.fn(async () => ({ success: true, data: 'run-1' })),
    trainingStop: vi.fn(async () => ({ success: true, data: undefined })),
    onTrainingProgress: vi.fn(() => () => {}),
    onTrainingResult: vi.fn(() => () => {}),
    systemCudaStatus: vi.fn(async () => ({ success: true, data: { cudaAvailable: false } })),
    inferenceRun: vi.fn(),
  };
}

describe('数据集加载 + 训练/推理整单位可选', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setElectronApi();
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    useTrainingStore.setState({
      mode: 'local',
      config: {
        dataset: '',
        modelArch: 'multiscale',
        outputName: '',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
      progress: null,
      isRunning: false,
    });
    useModelStore.setState({
      models: [],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
    useInferenceStore.setState({ activeTask: null, queue: [], result: null, isRunning: false });
  });

  it('test_app_loads_datasets_on_mount', async () => {
    mockDatasetList.mockResolvedValue({
      success: true,
      data: [
        { id: 'd1', name: 'FD002', samples: 100, format: 'txt', type: 'train', createdAt: 't', astraCompatible: true, trainFile: 'a', testFile: 'b', rulFile: 'c' },
      ],
    });
    const { default: App } = await import('../../src/renderer/App');
    render(React.createElement(App));
    await waitFor(() => {
      expect(mockDatasetList).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(useDatasetStore.getState().datasets.some((d) => d.name === 'FD002')).toBe(true);
    });
  });

  it('test_training_page_shows_loaded_datasets', async () => {
    useDatasetStore
      .getState()
      .setDatasets([
        { id: 'd1', name: 'FD002', samples: 100, format: 'txt', type: 'train', createdAt: 't', astraCompatible: true, trainFile: 'a', testFile: 'b', rulFile: 'c' },
      ]);
    const { default: Training } = await import('../../src/renderer/pages/Training');
    render(React.createElement(Training));
    // 数据集下拉应包含 FD002
    const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
    await waitFor(() => {
      expect(Array.from(select.options).map((o) => o.value)).toContain('d1');
    });
  });

  it('adds a newly trained checkpoint to the shared model store', async () => {
    mockDatasetList.mockResolvedValue({ success: true, data: [] });
    const { default: App } = await import('../../src/renderer/App');
    render(React.createElement(App));
    await waitFor(() => expect(modelStatusCallback).toBeTypeOf('function'));
    modelStatusCallback?.({
      id: 'trained-1', name: 'rocket-rul', status: 'available', source: 'trained',
      isDefault: true, path: 'E:/backend/outputs/rocket.pt', createdAt: 't',
    });
    await waitFor(() => {
      expect(useModelStore.getState().models.some((model) => model.id === 'trained-1')).toBe(true);
    });
  });

  it('test_training_selects_dataset_whole_unit', async () => {
    useDatasetStore
      .getState()
      .setDatasets([
        { id: 'd1', name: 'FD002', samples: 100, format: 'txt', type: 'train', createdAt: 't', astraCompatible: true, trainFile: 'a', testFile: 'b', rulFile: 'c' },
      ]);
    const { default: Training } = await import('../../src/renderer/pages/Training');
    render(React.createElement(Training));
    const select = screen.getAllByRole('combobox')[0] as HTMLSelectElement;
    fireEvent.change(select, { target: { value: 'd1' } });
    await waitFor(() => {
      expect(useTrainingStore.getState().config.dataset).toBe('d1');
    });
  });

  it('test_inference_page_shows_loaded_datasets', async () => {
    useDatasetStore
      .getState()
      .setDatasets([
        { id: 'd1', name: 'FD002', samples: 100, format: 'txt', type: 'train', createdAt: 't', astraCompatible: true, trainFile: 'a', testFile: 'b', rulFile: 'c' },
      ]);
    const { default: Inference } = await import('../../src/renderer/pages/Inference');
    render(React.createElement(Inference));
    // 推理页面应显示数据集 FD002 供选择
    expect(await screen.findByText('FD002')).toBeDefined();
  });
});
