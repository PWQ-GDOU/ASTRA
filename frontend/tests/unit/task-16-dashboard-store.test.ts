/**
 * Task 16 测试：dashboard.store.ts — Zustand store 定义
 *
 * 按 02-dashboard.md 任务 16 的功能点要求验证：
 * - DashboardState: stats { datasetCount, modelCount, inferenceCount, systemStatus, gpuInfo },
 *   recentTasks, trainingProgress, accuracyTrend, lastUpdated
 * - DashboardActions: refreshStats, refreshRecentTasks, refreshTrainingProgress
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { useDashboardStore } from '../../src/renderer/stores/dashboard.store';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useModelStore } from '../../src/renderer/stores/model.store';
import { useInferenceStore } from '../../src/renderer/stores/inference.store';
import { useInferenceHistoryStore } from '../../src/renderer/stores/inference-history.store';
import { useTrainingStore } from '../../src/renderer/stores/training.store';

describe('Task 16: dashboard.store.ts Zustand store', () => {
  beforeEach(() => {
    // 重置所有 store 到初始状态，避免测试间相互污染
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
      savedConfigs: [],
    });
  });

  it('test_dashboard_store_initial_stats_zero', () => {
    const s = useDashboardStore.getState();
    expect(s.stats.datasetCount).toBe(0);
    expect(s.stats.modelCount).toBe(0);
    expect(s.stats.inferenceCount).toBe(0);
  });

  it('test_dashboard_store_refresh_stats_calls_other_stores', () => {
    useDatasetStore.getState().setDatasets([
      { id: 'd1', name: 'A', samples: 10, format: 'csv', type: 'train', createdAt: 't' },
      { id: 'd2', name: 'B', samples: 20, format: 'csv', type: 'test', createdAt: 't' },
    ]);
    useModelStore.getState().setModels([
      {
        id: 'm1',
        name: 'M',
        type: 'llm',
        status: 'available',
        source: 'imported',
        isDefault: false,
        path: '/p',
        createdAt: 't',
      },
    ]);

    useDashboardStore.getState().refreshStats();

    const s = useDashboardStore.getState();
    expect(s.stats.datasetCount).toBe(2);
    expect(s.stats.modelCount).toBe(1);
  });

  it('test_dashboard_store_refresh_recent_tasks_calls_ipc', async () => {
    useInferenceHistoryStore.getState().setItems([
      {
        id: 'i1',
        taskName: 'T1',
        model: 'm',
        dataset: 'd',
        status: 'completed',
        duration: '1s',
        time: 't',
        metrics: { accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9, latency: 1, throughput: 2 },
        log: [],
      },
    ]);

    await useDashboardStore.getState().refreshRecentTasks();

    expect(useDashboardStore.getState().recentTasks).toHaveLength(1);
    expect(useDashboardStore.getState().recentTasks[0].name).toBe('T1');
  });

  it('test_dashboard_store_refresh_training_progress_calls_ipc', () => {
    useTrainingStore.getState().setConfig({
      outputName: 'my-run',
      hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
    });
    useTrainingStore.getState().setProgress({ epoch: 5, step: 10, loss: 0.5, log: [] });
    useTrainingStore.getState().setIsRunning(true);

    useDashboardStore.getState().refreshTrainingProgress();

    expect(useDashboardStore.getState().trainingProgress).toHaveLength(1);
    expect(useDashboardStore.getState().trainingProgress[0].name).toBe('my-run');
    expect(useDashboardStore.getState().trainingProgress[0].progress).toBe(50);
  });

  it('test_dashboard_store_initial_recent_tasks_empty', () => {
    expect(useDashboardStore.getState().recentTasks).toEqual([]);
  });

  it('test_dashboard_store_initial_training_progress_empty', () => {
    expect(useDashboardStore.getState().trainingProgress).toEqual([]);
  });

  it('test_dashboard_store_refresh_stats_updates_counts', () => {
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
    ]);

    useDashboardStore.getState().refreshStats();

    expect(useDashboardStore.getState().stats.inferenceCount).toBe(1);
  });

  it('test_dashboard_store_last_updated_set_on_refresh', () => {
    expect(useDashboardStore.getState().lastUpdated).toBeNull();
    useDashboardStore.getState().refreshStats();
    expect(useDashboardStore.getState().lastUpdated).not.toBeNull();
  });

  it('test_dashboard_store_accuracy_trend_initial_empty', () => {
    expect(useDashboardStore.getState().accuracyTrend).toEqual([]);
  });

  it('test_dashboard_store_system_status_default_offline', () => {
    expect(useDashboardStore.getState().stats.systemStatus).toBe('offline');
  });
});
