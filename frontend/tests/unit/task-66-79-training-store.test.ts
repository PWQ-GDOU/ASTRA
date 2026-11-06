/**
 * Task 66-79 测试：训练模块
 *
 * 覆盖：
 * - Task 66: training.store（mode/config/progress/systemStatus/isRunning）
 * - Task 67-71: 模式切换 + 配置面板
 * - Task 72: 系统状态
 * - Task 73: 训练进度
 * - Task 78-79: 页面集成（部分在 task-158 已测）
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { useTrainingStore } from '../../src/renderer/stores/training.store';

describe('Task 66: training.store', () => {
  beforeEach(() => {
    useTrainingStore.setState({
      mode: 'local',
      config: {
        dataset: '',
        baseModel: '',
        outputName: '',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
      progress: null,
      isRunning: false,
      savedConfigs: [],
      systemStatus: {
        cuda: { available: false },
        cpu: { usage: 0, cores: 0 },
        ram: { used: '...', total: '...', percent: 0 },
      },
    });
  });

  it('test_training_store_default_mode_local', () => {
    expect(useTrainingStore.getState().mode).toBe('local');
  });

  it('test_training_store_set_mode', () => {
    useTrainingStore.getState().setMode('remote');
    expect(useTrainingStore.getState().mode).toBe('remote');
  });

  it('test_training_store_default_config', () => {
    const c = useTrainingStore.getState().config;
    expect(c.hyperParams.epochs).toBe(10);
    expect(c.hyperParams.learningRate).toBe(0.001);
  });

  it('test_training_store_set_config_merges', () => {
    useTrainingStore.getState().setConfig({ dataset: 'FD002', outputName: 'run1' });
    const c = useTrainingStore.getState().config;
    expect(c.dataset).toBe('FD002');
    expect(c.outputName).toBe('run1');
  });

  it('test_training_store_set_config_hyperparams', () => {
    useTrainingStore.getState().setConfig({
      hyperParams: { ...useTrainingStore.getState().config.hyperParams, epochs: 50 },
    });
    expect(useTrainingStore.getState().config.hyperParams.epochs).toBe(50);
  });

  it('test_training_store_set_progress', () => {
    useTrainingStore.getState().setProgress({ epoch: 3, step: 300, loss: 1.5, log: [] });
    expect(useTrainingStore.getState().progress?.epoch).toBe(3);
  });

  it('test_training_store_add_log', () => {
    useTrainingStore.getState().setProgress({ epoch: 1, step: 100, loss: 2.0, log: [] });
    useTrainingStore.getState().addLog('E1 loss=2.0');
    expect(useTrainingStore.getState().progress?.log).toContain('E1 loss=2.0');
  });

  it('test_training_store_set_system_status', () => {
    useTrainingStore.getState().setSystemStatus({
      cuda: { available: true, gpuName: 'RTX 4090', memory: '24GB' },
      cpu: { usage: 30, cores: 16 },
      ram: { used: '8GB', total: '32GB', percent: 25 },
    });
    expect(useTrainingStore.getState().systemStatus.cuda.available).toBe(true);
  });

  it('test_training_store_set_is_running', () => {
    useTrainingStore.getState().setIsRunning(true);
    expect(useTrainingStore.getState().isRunning).toBe(true);
  });

  it('test_training_store_save_config', () => {
    const cfg = {
      dataset: 'FD002',
      baseModel: 'multiscale',
      outputName: 'run1',
      hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
    };
    useTrainingStore.getState().saveConfig(cfg);
    expect(useTrainingStore.getState().savedConfigs).toHaveLength(1);
  });
});
