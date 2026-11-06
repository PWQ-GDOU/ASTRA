/**
 * Task 104-116 测试：IPC 层
 *
 * 覆盖：
 * - Task 104: IPC_CHANNELS 常量
 * - Task 105: IPCResult 类型
 * - Task 106: 共享类型
 * - Task 107: ipc-client.ts ipcCall
 * - Task 108: register-all 注册入口
 * - Task 112: model IPC handlers
 * - Task 113: system IPC handlers（window controls）
 * - Task 114: statistics IPC handlers
 * - Task 115: preload electronAPI
 * - Task 116: 数据持久化层 storage
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { IPC_CHANNELS } from '../../src/shared/types/ipc';
import { resolve } from 'path';
import type { IPCResult } from '../../src/shared/types/ipc-result';

describe('Task 104: IPC_CHANNELS 常量', () => {
  it('test_ipc_channels_window_defined', () => {
    expect(IPC_CHANNELS.WINDOW_MINIMIZE).toBe('window:minimize');
    expect(IPC_CHANNELS.WINDOW_MAXIMIZE).toBe('window:maximize');
    expect(IPC_CHANNELS.WINDOW_CLOSE).toBe('window:close');
  });

  it('test_ipc_channels_dataset_defined', () => {
    expect(IPC_CHANNELS.DATASET_LIST).toBe('dataset:list');
    expect(IPC_CHANNELS.DATASET_IMPORT).toBe('dataset:import');
    expect(IPC_CHANNELS.DATASET_UPDATE_TYPE).toBe('dataset:update-type');
    expect(IPC_CHANNELS.DATASET_DELETE).toBe('dataset:delete');
  });

  it('test_ipc_channels_inference_defined', () => {
    expect(IPC_CHANNELS.INFERENCE_RUN).toBe('inference:run');
    expect(IPC_CHANNELS.INFERENCE_PROGRESS).toBe('inference:progress');
    expect(IPC_CHANNELS.INFERENCE_RESULT).toBe('inference:result');
  });

  it('test_ipc_channels_training_defined', () => {
    expect(IPC_CHANNELS.TRAINING_START_LOCAL).toBe('training:start-local');
    expect(IPC_CHANNELS.TRAINING_PROGRESS).toBe('training:progress');
    expect(IPC_CHANNELS.TRAINING_RESULT).toBe('training:result');
  });

  it('test_ipc_channels_model_defined', () => {
    expect(IPC_CHANNELS.MODEL_LIST).toBe('model:list');
    expect(IPC_CHANNELS.MODEL_SET_DEFAULT).toBe('model:set-default');
  });

  it('test_ipc_channels_system_statistics', () => {
    expect(IPC_CHANNELS.SYSTEM_CUDA_STATUS).toBe('system:cuda-status');
    expect(IPC_CHANNELS.STATISTICS_QUERY).toBe('statistics:query');
  });

  it('test_ipc_channels_all_namespaced', () => {
    for (const key of Object.keys(IPC_CHANNELS)) {
      const val = IPC_CHANNELS[key as keyof typeof IPC_CHANNELS];
      expect(val).toMatch(/^[a-z]+:[a-z-]+$/);
    }
  });
});

describe('Task 105-106: IPCResult 类型 + 共享类型', () => {
  it('test_ipc_result_success_shape', () => {
    const r: IPCResult<number> = { success: true, data: 42 };
    expect(r.success).toBe(true);
    expect(r.data).toBe(42);
  });

  it('test_ipc_result_error_shape', () => {
    const r: IPCResult<number> = { success: false, error: 'boom' };
    expect(r.success).toBe(false);
    expect(r.error).toBe('boom');
  });

  it('test_shared_inference_types', async () => {
    // interface 类型无运行时值，验证模块存在且可导入即可
    const mod = await import('../../src/shared/types/inference');
    expect(mod).toBeDefined();
  });

  it('test_shared_types_index_exports', async () => {
    const idx = await import('../../src/shared/types/index');
    expect(idx).toBeDefined();
  });
});

describe('Task 108: register-all 注册入口', () => {
  it('test_register_all_exports_function', async () => {
    const { readFileSync } = await import('fs');
    const { resolve } = await import('path');
    const content = readFileSync(
      resolve(__dirname, '..', '..', 'src', 'main', 'ipc', 'register-all.ts'),
      'utf-8',
    );
    expect(content).toContain('export function registerAllHandlers');
    expect(content).toContain('registerSystemHandlers');
    expect(content).toContain('registerDatasetHandlers');
    expect(content).toContain('registerTrainingHandlers');
  });
});

describe('Task 112: model IPC handlers', () => {
  const mockHandle = vi.fn();
  const mockShowOpenDialog = vi.fn();
  const mockReadJSON = vi.fn();
  const mockWriteJSON = vi.fn();
  const mockAppendJSON = vi.fn();
  const mockDeleteJSON = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockReadJSON.mockReset();
    mockWriteJSON.mockReset();
    mockAppendJSON.mockReset();
    mockDeleteJSON.mockReset();
  });

  async function loadModel() {
    vi.resetModules();
    vi.doMock('electron', () => ({
      ipcMain: { handle: (...args: unknown[]) => mockHandle(...args) },
      dialog: { showOpenDialog: (...args: unknown[]) => mockShowOpenDialog(...args) },
    }));
    vi.doMock('../../src/main/lib/storage', () => ({
      readJSON: (...args: unknown[]) => mockReadJSON(...args),
      writeJSON: (...args: unknown[]) => mockWriteJSON(...args),
      appendJSON: (...args: unknown[]) => mockAppendJSON(...args),
      deleteJSON: (...args: unknown[]) => mockDeleteJSON(...args),
      MODELS_FILE: 'models.json',
    }));
    const mod = await import('../../src/main/ipc/model.ipc');
    mod.registerModelHandlers();
  }

  function findHandler(channel: string) {
    return mockHandle.mock.calls.find((c: unknown[]) => c[0] === channel)?.[1];
  }

  it('test_model_list_handler_registered', async () => {
    mockReadJSON.mockReturnValue([]);
    await loadModel();
    expect(findHandler('model:list')).toBeDefined();
  });

  it('test_model_list_returns_models', async () => {
    mockReadJSON.mockReturnValue([
      {
        id: 'm1',
        name: 'A',
        status: 'available',
        source: 'imported',
        isDefault: false,
        path: '/p',
        createdAt: 't',
      },
    ]);
    await loadModel();
    const result = findHandler('model:list')();
    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(1);
  });

  it('test_model_set_default_writes', async () => {
    mockReadJSON.mockReturnValue([
      {
        id: 'm1',
        name: 'A',
        status: 'available',
        source: 'imported',
        isDefault: false,
        path: '/p',
        createdAt: 't',
      },
      {
        id: 'm2',
        name: 'B',
        status: 'available',
        source: 'imported',
        isDefault: true,
        path: '/p',
        createdAt: 't',
      },
    ]);
    await loadModel();
    const result = findHandler('model:set-default')({}, 'm1');
    expect(result.success).toBe(true);
    expect(mockWriteJSON).toHaveBeenCalled();
  });

  it('makes the first imported checkpoint the default model', async () => {
    mockReadJSON.mockReturnValue([]);
    mockShowOpenDialog.mockResolvedValue({ canceled: false, filePaths: ['E:/models/astra.pt'] });
    await loadModel();
    const result = await findHandler('model:import')();
    expect(result.success).toBe(true);
    expect(result.data.isDefault).toBe(true);
    expect(mockAppendJSON).toHaveBeenCalledWith(
      'models.json',
      expect.objectContaining({ isDefault: true, path: 'E:/models/astra.pt' }),
    );
  });

  it('test_model_delete_calls_delete', async () => {
    await loadModel();
    const result = findHandler('model:delete')({}, 'm1');
    expect(mockDeleteJSON).toHaveBeenCalledWith('models.json', 'm1');
    expect(result.success).toBe(true);
  });
});

describe('Task 113-114: system/statistics IPC handlers', () => {
  const mockHandle = vi.fn();
  const mockIpcMain = { handle: (...args: unknown[]) => mockHandle(...args) };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('test_statistics_handler_returns_data', async () => {
    vi.resetModules();
    vi.doMock('electron', () => ({ ipcMain: mockIpcMain }));
    vi.doMock('../../src/main/lib/storage', () => ({
      readJSON: () => [],
      DATASETS_FILE: 'datasets.json',
      MODELS_FILE: 'models.json',
      INFERENCE_HISTORY_FILE: 'inference-history.json',
    }));
    const mod = await import('../../src/main/ipc/statistics.ipc');
    mod.registerStatisticsHandlers();
    const handler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'statistics:query')?.[1];
    expect(handler).toBeDefined();
    const result = handler();
    expect(result.success).toBe(true);
    expect(result.data.trainingTrend).toEqual([]);
    expect(result.data.taskStats.completed).toBe(0);
  });
});

describe('Task 115: preload electronAPI', () => {
  it('test_preload_exposes_api', async () => {
    const content = (await import('fs')).readFileSync(
      resolve(__dirname, '..', '..', 'src', 'preload', 'index.ts'),
      'utf-8',
    );
    expect(content).toContain('contextBridge');
    expect(content).toContain('exposeInMainWorld');
    expect(content).toContain('electronAPI');
  });

  it('test_preload_training_result_listener', async () => {
    const content = (await import('fs')).readFileSync(
      resolve(__dirname, '..', '..', 'src', 'preload', 'index.ts'),
      'utf-8',
    );
    expect(content).toContain('onTrainingResult');
  });
});
