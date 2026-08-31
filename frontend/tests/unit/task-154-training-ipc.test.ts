/**
 * Task 154 测试：training:start-local 真实化
 *
 * 验证 training.ipc.ts 的 start-local 从 mock interval 改为调用 spawnTraining，
 * 并在后端不可用时回退 mock。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mocks
const mockSpawnTraining = vi.fn();
const mockSend = vi.fn();
const mockHandle = vi.fn();
const mockShowOpenDialog = vi.fn(async () => ({ canceled: true, filePaths: [] }));
const mockReadTrainingResultFile = vi.fn();
const mockAppendJSON = vi.fn();

const mockBrowserWindow = {
  webContents: { send: (...args: unknown[]) => mockSend(...args) },
};

vi.mock('child_process', () => ({
  spawn: () => ({
    on: vi.fn(),
    stdout: { on: vi.fn() },
    stderr: { on: vi.fn() },
    kill: vi.fn(),
  }),
}));

vi.mock('electron', () => ({
  ipcMain: { handle: (...args: unknown[]) => mockHandle(...args) },
  dialog: {
    showOpenDialog: (...args: unknown[]) =>
      mockShowOpenDialog(...(args as [])) as ReturnType<typeof mockShowOpenDialog>,
  },
  BrowserWindow: {},
  app: {
    getPath: vi.fn(() => '/mock/userdata'),
  },
}));

vi.mock('../../src/main/lib/python-bridge', () => ({
  spawnTraining: (...args: unknown[]) => mockSpawnTraining(...args),
  isAstraAvailable: vi.fn(() => true),
  ASTRA_DIR: '/mock/astra',
  OUTPUT_DIR: 'outputs/clean_benchmark',
}));
vi.mock('../../src/main/lib/storage', () => ({
  readJSON: (filename: string) => filename === 'datasets.json'
    ? [{ id: 'FD002', name: 'Custom', astraCompatible: true, trainFile: 'train.any', testFile: 'test.any', rulFile: 'rul.any' }, { id: 'FD003', name: 'Custom 3', astraCompatible: true, trainFile: 'train3.any', testFile: 'test3.any', rulFile: 'rul3.any' }]
    : [],
  appendJSON: (...args: unknown[]) => mockAppendJSON(...args),
  TRAINING_CONFIG_FILE: 'training.json', DATASETS_FILE: 'datasets.json', MODELS_FILE: 'models.json',
}));
vi.mock('../../src/main/lib/training-progress', () => ({
  parseEpochLine: vi.fn(() => null),
  parseResultLine: vi.fn(() => null),
  isSeedStartLine: vi.fn(() => false),
  parsePrimarySeed: vi.fn(() => 42),
  readTrainingResultFile: (...args: unknown[]) => mockReadTrainingResultFile(...args),
}));

describe('Task 154: training:start-local 真实化', () => {
  let registerTrainingHandlers: (win: unknown) => void;

  beforeEach(() => {
    vi.clearAllMocks();
    mockSpawnTraining.mockReset();
    mockSpawnTraining.mockReturnValue({
      runId: 'run-abc',
      kill: vi.fn(),
      promise: Promise.resolve({ exitCode: 0 }),
    });
    mockReadTrainingResultFile.mockReturnValue({
      runId: '', seed: 42, bestEpoch: 1, valRmse: 1, testRmse: 2,
      testScore: 3, testRmseCap: 2, testScoreCap: 3, nParams: 10,
      seconds: 0.1, history: [],
    });
  });

  it('test_training_start_local_returns_run_id', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    // Find the start-local handler
    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    expect(startLocalHandler).toBeDefined();
    const handler = startLocalHandler[1] as (e: unknown, config: unknown) => Promise<unknown>;

    const result = await handler(
      {},
      {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'run1',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    );
    expect(result).toMatchObject({ success: true });
    const data = result as { data: string };
    expect(data.data).toBeDefined();
    expect(data.data.length).toBeGreaterThan(0);
  });

  it('test_training_start_local_spawns_python', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    const handler = startLocalHandler[1] as (e: unknown, config: unknown) => Promise<unknown>;

    await handler(
      {},
      {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'run1',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    );
    expect(mockSpawnTraining).toHaveBeenCalled();
  });

  it('test_training_start_local_config_passed', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    const handler = startLocalHandler[1] as (e: unknown, config: unknown) => Promise<unknown>;

    const cfg = {
      dataset: 'FD003',
      modelArch: 'v2',
      outputName: 'run2',
      hyperParams: { epochs: 5, batchSize: 16, learningRate: 0.0001, optimizer: 'adam' },
    };
    await handler({}, cfg);
    const spawnArg = mockSpawnTraining.mock.calls[0][0];
    expect(spawnArg).toMatchObject({ ...cfg, trainFile: 'train3.any', testFile: 'test3.any', rulFile: 'rul3.any' });
  });

  it('test_training_start_remote_reports_not_integrated', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const startRemoteHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-remote',
    )!;
    expect(startRemoteHandler).toBeDefined();
    const handler = startRemoteHandler[1] as (e: unknown) => Promise<unknown>;
    const result = await handler({});
    expect(result).toMatchObject({ success: false });
    expect(mockSpawnTraining).not.toHaveBeenCalled();
  });

  it('test_training_start_local_error_handled', async () => {
    mockSpawnTraining.mockImplementation(() => {
      throw new Error('spawn failed');
    });
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    const handler = startLocalHandler[1] as (e: unknown, config: unknown) => Promise<unknown>;

    const result = await handler(
      {},
      {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'run1',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    );
    expect(result).toMatchObject({ success: false });
  });

  it('test_training_stop_clears_process', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    // start a run
    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    const started = (await (startLocalHandler[1] as (e: unknown, cfg: unknown) => Promise<unknown>)(
      {},
      {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'r',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    )) as { data: string };

    // stop it with the actual returned runId
    const stopHandler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:stop')!;
    expect(stopHandler).toBeDefined();
    const result = await (stopHandler[1] as (e: unknown, runId: string) => Promise<unknown>)(
      {},
      started.data,
    );
    expect(result).toMatchObject({ success: true });
  });

  it('test_training_stop_missing_run_id_error', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const stopHandler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:stop')!;
    const result = await (stopHandler[1] as (e: unknown, runId: string) => Promise<unknown>)(
      {},
      'nonexistent',
    );
    expect(result).toMatchObject({ success: true });
  });

  it('test_training_start_local_registers_process', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    registerTrainingHandlers = mod.registerTrainingHandlers as (win: unknown) => void;
    registerTrainingHandlers(mockBrowserWindow);

    const startLocalHandler = mockHandle.mock.calls.find(
      (c: unknown[]) => c[0] === 'training:start-local',
    )!;
    const started = (await (startLocalHandler[1] as (e: unknown, cfg: unknown) => Promise<unknown>)(
      {},
      {
        dataset: 'FD002',
        modelArch: 'multiscale',
        outputName: 'r',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
    )) as { data: string };

    // stop should find it (already tested), and stop again should be graceful
    const stopHandler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:stop')!;
    await (stopHandler[1] as (e: unknown, runId: string) => Promise<unknown>)({}, started.data);
    const second = await (stopHandler[1] as (e: unknown, runId: string) => Promise<unknown>)(
      {},
      started.data,
    );
    expect(second).toMatchObject({ success: true });
  });

  it('reports a failed training process to the renderer', async () => {
    mockSpawnTraining.mockReturnValue({
      runId: 'run-failed',
      kill: vi.fn(),
      promise: Promise.resolve({ exitCode: 2 }),
    });
    const mod = await import('../../src/main/ipc/training.ipc');
    mod.registerTrainingHandlers(mockBrowserWindow as never);
    const handler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:start-local')![1] as (e: unknown, config: unknown) => Promise<unknown>;

    await handler({}, {
      dataset: 'FD002', modelArch: 'multiscale', sensorMode: 'all24', seeds: '42',
      hyperParams: { epochs: 1, batchSize: 8, learningRate: 0.001, optimizer: 'adam' },
    });

    await vi.waitFor(() => expect(mockSend).toHaveBeenCalledWith(
      'training:progress',
      expect.objectContaining({ status: 'failed', log: expect.stringContaining('退出码 2') }),
    ));
  });

  it('reports a missing result file instead of leaving the UI running', async () => {
    mockReadTrainingResultFile.mockReturnValue(null);
    const mod = await import('../../src/main/ipc/training.ipc');
    mod.registerTrainingHandlers(mockBrowserWindow as never);
    const handler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:start-local')![1] as (e: unknown, config: unknown) => Promise<unknown>;

    await handler({}, {
      dataset: 'FD002', modelArch: 'multiscale', sensorMode: 'all24', seeds: '42',
      hyperParams: { epochs: 1, batchSize: 8, learningRate: 0.001, optimizer: 'adam' },
    });

    await vi.waitFor(() => expect(mockSend).toHaveBeenCalledWith(
      'training:progress',
      expect.objectContaining({ status: 'failed', log: expect.stringContaining('结果文件') }),
    ));
  });

  it('registers the generated checkpoint as an immediately usable model', async () => {
    const mod = await import('../../src/main/ipc/training.ipc');
    mod.registerTrainingHandlers(mockBrowserWindow as never);
    const handler = mockHandle.mock.calls.find((c: unknown[]) => c[0] === 'training:start-local')![1] as (e: unknown, config: unknown) => Promise<unknown>;

    await handler({}, {
      dataset: 'FD002', modelArch: 'multiscale', sensorMode: 'all24', seeds: '42',
      outputName: 'rocket-rul', outputDir: 'outputs/custom',
      hyperParams: { epochs: 1, batchSize: 8, learningRate: 0.001, optimizer: 'adam' },
    });

    await vi.waitFor(() => expect(mockAppendJSON).toHaveBeenCalledWith(
      'models.json',
      expect.objectContaining({
        name: 'rocket-rul', source: 'trained', isDefault: true,
        path: expect.stringMatching(/outputs[\\/]custom[\\/]FD002_multiscale_all24_s42\.pt$/),
      }),
    ));
    expect(mockSend).toHaveBeenCalledWith('model:status', expect.objectContaining({ name: 'rocket-rul' }));
  });
});
