import { EventEmitter } from 'events';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  handle: vi.fn(),
  send: vi.fn(),
  spawn: vi.fn(),
  readFile: vi.fn(),
  appendJSON: vi.fn(),
  updateJSON: vi.fn(),
  deleteJSON: vi.fn(),
  unlink: vi.fn(),
}));

class FakeProcess extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  killed = false;
  kill = vi.fn(() => {
    this.killed = true;
  });
}

vi.mock('electron', () => ({
  app: { getPath: vi.fn(() => 'E:/temp') },
  BrowserWindow: {},
  ipcMain: { handle: (...args: unknown[]) => mocks.handle(...args) },
}));

vi.mock('child_process', () => {
  const childProcess = { spawn: (...args: unknown[]) => mocks.spawn(...args) };
  return { ...childProcess, default: childProcess };
});

vi.mock('fs', () => {
  const fs = {
    existsSync: vi.fn(() => true),
    readFileSync: (...args: unknown[]) => mocks.readFile(...args),
    unlinkSync: (...args: unknown[]) => mocks.unlink(...args),
  };
  return { ...fs, default: fs };
});

vi.mock('../../src/main/lib/python-bridge', () => ({ ASTRA_DIR: 'E:/repo/backend' }));
vi.mock('../../src/main/lib/storage', () => ({
  MODELS_FILE: 'models.json',
  DATASETS_FILE: 'datasets.json',
  INFERENCE_HISTORY_FILE: 'inference-history.json',
  readJSON: (filename: string) =>
    filename === 'models.json'
      ? [
          {
            id: 'm1',
            name: 'model.pt',
            path: 'E:/model.pt',
            status: 'available',
            source: 'imported',
            isDefault: true,
            createdAt: 't',
          },
        ]
      : filename === 'datasets.json'
        ? [
            {
              id: 'd1',
              name: 'FD002',
              astraCompatible: true,
              trainFile: 'train.txt',
              testFile: 'test.txt',
              rulFile: 'rul.txt',
            },
          ]
        : [
            {
              id: 'h1',
              taskName: '推理-FD002',
              status: 'completed',
              model: 'model.pt',
              dataset: 'FD002',
            },
          ],
  appendJSON: (...args: unknown[]) => mocks.appendJSON(...args),
  updateJSON: (...args: unknown[]) => mocks.updateJSON(...args),
  deleteJSON: (...args: unknown[]) => mocks.deleteJSON(...args),
}));

describe('ASTRA inference process lifecycle', () => {
  let child: FakeProcess;
  let run: (event: unknown, request: unknown) => { success: boolean; data?: string };

  beforeEach(async () => {
    vi.clearAllMocks();
    vi.resetModules();
    child = new FakeProcess();
    mocks.spawn.mockReturnValue(child);
    mocks.readFile.mockReturnValue(
      JSON.stringify({
        units: [1],
        predictions: [42],
        targets: [40],
        metrics: { rmse: 2, mae: 2, score: 0.2 },
        seconds: 0.5,
      }),
    );
    const mod = await import('../../src/main/ipc/inference.ipc');
    mod.registerInferenceHandlers({
      webContents: { send: (...args: unknown[]) => mocks.send(...args) },
    } as never);
    run = mocks.handle.mock.calls.find((call: unknown[]) => call[0] === 'inference:run')![1];
  });

  const request = {
    datasetId: 'd1',
    modelId: 'm1',
    batchSize: 32,
    device: 'cpu',
    applyRulCap: true,
    rulCap: 125,
  };

  it('rejects non-finite inference parameters before starting Python', () => {
    const result = run({}, { ...request, rulCap: Number.NaN });

    expect(result.success).toBe(false);
    expect(mocks.spawn).not.toHaveBeenCalled();
  });

  it('reports Python stderr when inference exits unsuccessfully', async () => {
    const started = run({}, request);
    expect(started.success).toBe(true);
    child.stderr.emit('data', Buffer.from('checkpoint incompatible'));
    child.emit('close', 2);

    expect(mocks.send).toHaveBeenCalledWith(
      'inference:progress',
      expect.objectContaining({
        status: 'failed',
        message: expect.stringContaining('checkpoint incompatible'),
      }),
    );
    expect(mocks.unlink).toHaveBeenCalledWith(expect.stringContaining('astra-inference-'));
  });

  it('turns malformed backend output into a failed task', () => {
    run({}, request);
    mocks.readFile.mockImplementation(() => {
      throw new Error('missing output');
    });

    expect(() => child.emit('close', 0)).not.toThrow();
    expect(mocks.send).toHaveBeenCalledWith(
      'inference:progress',
      expect.objectContaining({
        status: 'failed',
        message: expect.stringContaining('missing output'),
      }),
    );
  });

  it('rejects structurally invalid backend JSON instead of emitting a broken result', () => {
    run({}, request);
    mocks.readFile.mockReturnValue(
      JSON.stringify({ units: 'not-an-array', metrics: {}, seconds: 1 }),
    );

    child.emit('close', 0);

    expect(mocks.send).toHaveBeenCalledWith(
      'inference:progress',
      expect.objectContaining({ status: 'failed', message: expect.stringContaining('格式无效') }),
    );
    expect(mocks.send).not.toHaveBeenCalledWith('inference:result', expect.anything());
  });

  it('persists a successful inference for the history and statistics pages', () => {
    run({}, request);
    child.emit('close', 0);

    expect(mocks.appendJSON).toHaveBeenCalledWith(
      'inference-history.json',
      expect.objectContaining({ status: 'completed', model: 'model.pt', dataset: 'FD002' }),
    );
    expect(mocks.send).toHaveBeenCalledWith(
      'inference:result',
      expect.objectContaining({ rmse: 2 }),
    );
  });

  it('writes the latest measured performance back to the selected model', () => {
    run({}, request);
    child.emit('close', 0);

    expect(mocks.updateJSON).toHaveBeenCalledWith(
      'models.json',
      'm1',
      expect.any(Function),
    );
    const update = mocks.updateJSON.mock.calls[0][2] as (model: Record<string, unknown>) => {
      performance: Record<string, unknown>;
    };
    const updated = update({ id: 'm1', name: 'model.pt' });
    expect(updated.performance).toMatchObject({
      taskId: expect.any(String),
      datasetId: 'd1',
      datasetName: 'FD002',
      rmse: 2,
      mae: 2,
      score: 0.2,
      sampleCount: 1,
      seconds: 0.5,
      evaluatedAt: expect.any(String),
    });
    expect(mocks.send).toHaveBeenCalledWith(
      'model:status',
      expect.objectContaining({
        id: 'm1',
        performance: expect.objectContaining({ datasetName: 'FD002', rmse: 2 }),
      }),
    );
  });

  it('registers history list and delete handlers backed by persistent storage', () => {
    const list = mocks.handle.mock.calls.find(
      (call: unknown[]) => call[0] === 'inference:history-list',
    )![1];
    const remove = mocks.handle.mock.calls.find(
      (call: unknown[]) => call[0] === 'inference:history-delete',
    )![1];

    expect(list({}, { status: 'completed' })).toEqual({
      success: true,
      data: [expect.objectContaining({ id: 'h1' })],
    });
    expect(remove({}, 'h1')).toEqual({ success: true, data: undefined });
    expect(mocks.deleteJSON).toHaveBeenCalledWith('inference-history.json', 'h1');
  });
});
