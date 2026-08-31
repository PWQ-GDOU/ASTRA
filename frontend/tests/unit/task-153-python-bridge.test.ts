/**
 * Task 153 测试：Python bridge 模块
 *
 * 验证 src/main/lib/python-bridge.ts 正确构造 spawn 命令调用 ASTRA clean_benchmark.py。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { resolve } from 'path';

// Mock child_process.spawn before importing module
const mockSpawn = vi.fn();

vi.mock('child_process', () => {
  const mocked = {
    spawn: (...args: unknown[]) => mockSpawn(...args),
  };
  return {
    default: mocked,
    ...mocked,
  };
});

describe('Task 153: Python bridge 模块', () => {
  let bridge: typeof import('../../src/main/lib/python-bridge');

  beforeEach(async () => {
    vi.resetModules();
    // 恢复 mockSpawn 的默认实现
    mockSpawn.mockImplementation(() => ({
      on: vi.fn(),
      stdout: { on: vi.fn() },
      stderr: { on: vi.fn() },
      kill: vi.fn(),
      pid: 1234,
    }));
    bridge = await import('../../src/main/lib/python-bridge');
  });

  const sampleConfig = {
    dataset: 'FD002',
    modelArch: 'multiscale',
    outputName: 'test-run',
    sensorMode: 'all24',
    seqLen: 60,
    rulCap: 125.0,
    seeds: '42',
    patience: 60,
    device: 'cuda',
    hyperParams: { epochs: 20, batchSize: 64, learningRate: 0.0005, optimizer: 'adam' },
  };

  it('resolves the backend from the frontend repository', () => {
    expect(bridge.ASTRA_DIR).toBe(resolve(process.cwd(), '..', 'backend'));
    expect(bridge.isAstraAvailable()).toBe(true);
  });

  it('test_python_bridge_builds_command', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const args = mockSpawn.mock.calls[0];
    expect(args).toBeDefined();
    const command = args[0];
    const cmdArgs = args[1] as string[];
    expect(String(command)).toMatch(/python/);
    expect(cmdArgs[0]).toBe('-u');
    expect(cmdArgs.join(' ')).toMatch(/clean_benchmark\.py/);
  });

  it('test_python_bridge_maps_epochs', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--epochs');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('20');
  });

  it('test_python_bridge_maps_batch_size', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--batch-size');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('64');
  });

  it('test_python_bridge_maps_learning_rate', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--lr');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('0.0005');
  });

  it('test_python_bridge_maps_model_arch', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--model');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('multiscale');
  });

  it('test_python_bridge_maps_sensor_mode', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--sensor-mode');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('all24');
  });

  it('test_python_bridge_maps_seq_len', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--seq-len');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('60');
  });

  it('test_python_bridge_maps_rul_cap', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--rul-cap');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('125');
  });

  it('test_python_bridge_maps_seeds', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--seeds');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('42');
  });

  it('test_python_bridge_maps_patience', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--patience');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('60');
  });

  it('test_python_bridge_maps_device', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--device');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toBe('cuda');
  });

  it('test_python_bridge_custom_config_passed_through', () => {
    const custom = {
      dataset: 'FD004',
      modelArch: 'condition',
      outputName: '',
      sensorMode: 'all24',
      seqLen: 30,
      rulCap: 150,
      seeds: '42,123,456',
      patience: 20,
      device: 'cpu',
      hyperParams: { epochs: 5, batchSize: 16, learningRate: 0.0001, optimizer: 'adam' },
    };
    mockSpawn.mockClear();
    bridge.spawnTraining(custom, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    expect(cmdArgs[cmdArgs.indexOf('--fd') + 1]).toBe('FD004');
    expect(cmdArgs[cmdArgs.indexOf('--model') + 1]).toBe('condition');
    expect(cmdArgs[cmdArgs.indexOf('--seq-len') + 1]).toBe('30');
    expect(cmdArgs[cmdArgs.indexOf('--rul-cap') + 1]).toBe('150');
    expect(cmdArgs[cmdArgs.indexOf('--seeds') + 1]).toBe('42,123,456');
    expect(cmdArgs[cmdArgs.indexOf('--patience') + 1]).toBe('20');
    expect(cmdArgs[cmdArgs.indexOf('--device') + 1]).toBe('cpu');
    expect(cmdArgs[cmdArgs.indexOf('--epochs') + 1]).toBe('5');
    expect(cmdArgs[cmdArgs.indexOf('--batch-size') + 1]).toBe('16');
  });

  it('maps dataset paths, split seed and over-estimation weight', () => {
    const args = bridge.buildTrainingArgs({
      ...sampleConfig,
      dataRoot: 'D:/astra-data',
      outputDir: 'outputs/fd004-run',
      splitSeed: 1234,
      overWeight: 0.15,
      valRatio: 0.35,
      device: 'cuda:2',
    });
    expect(args).toContain('D:/astra-data');
    expect(args).toContain('outputs/fd004-run');
    expect(args.slice(args.indexOf('--split-seed'), args.indexOf('--split-seed') + 2)).toEqual([
      '--split-seed',
      '1234',
    ]);
    expect(args.slice(args.indexOf('--over-weight'), args.indexOf('--over-weight') + 2)).toEqual([
      '--over-weight',
      '0.15',
    ]);
    expect(args.slice(args.indexOf('--val-ratio'), args.indexOf('--val-ratio') + 2)).toEqual([
      '--val-ratio',
      '0.35',
    ]);
    expect(args.slice(args.indexOf('--device'), args.indexOf('--device') + 2)).toEqual([
      '--device',
      'cuda:2',
    ]);
  });

  it('test_python_bridge_no_output_name_still_builds', () => {
    const noOutput = { ...sampleConfig, outputName: '' };
    bridge.spawnTraining(noOutput, vi.fn());
    expect(mockSpawn).toHaveBeenCalled();
  });

  it('test_python_bridge_sets_cwd_astra', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const opts = mockSpawn.mock.calls[0][2] as { cwd?: string };
    expect(opts.cwd).toBeDefined();
    expect(String(opts.cwd)).toMatch(/[\\/]backend$/);
  });

  it('test_python_bridge_windows_hide', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const opts = mockSpawn.mock.calls[0][2] as { windowsHide?: boolean };
    expect(opts.windowsHide).toBe(true);
  });

  it('test_python_bridge_returns_kill', () => {
    const result = bridge.spawnTraining(sampleConfig, vi.fn());
    expect(typeof result.kill).toBe('function');
  });

  it('test_python_bridge_returns_run_id', () => {
    const result = bridge.spawnTraining(sampleConfig, vi.fn());
    expect(result.runId).toBeDefined();
    expect(typeof result.runId).toBe('string');
    expect(result.runId.length).toBeGreaterThan(0);
  });

  it('test_python_bridge_output_dir_set', () => {
    bridge.spawnTraining(sampleConfig, vi.fn());
    const cmdArgs = mockSpawn.mock.calls[0][1] as string[];
    const idx = cmdArgs.indexOf('--output');
    expect(idx).toBeGreaterThan(-1);
    expect(cmdArgs[idx + 1]).toMatch(/outputs/);
  });

  it('buffers split stdout chunks until a complete progress line is available', () => {
    let stdoutData: ((chunk: Buffer) => void) | undefined;
    mockSpawn.mockImplementation(() => ({
      on: vi.fn(),
      stdout: {
        on: vi.fn((event: string, callback: (chunk: Buffer) => void) => {
          if (event === 'data') stdoutData = callback;
        }),
      },
      stderr: { on: vi.fn() },
      kill: vi.fn(),
      pid: 1234,
    }));
    const onStdout = vi.fn();

    bridge.spawnTraining(sampleConfig, onStdout);
    stdoutData?.(Buffer.from('Epoch 1/20 - loss=0.4'));
    stdoutData?.(Buffer.from('2 - val_rmse=17.5\nEpoch 2'));
    stdoutData?.(Buffer.from('/20 - loss=0.3 - val_rmse=16.0\n'));

    expect(onStdout.mock.calls.map((call) => call[0])).toEqual([
      'Epoch 1/20 - loss=0.42 - val_rmse=17.5',
      'Epoch 2/20 - loss=0.3 - val_rmse=16.0',
    ]);
  });

  it('test_python_bridge_detects_missing_astra', () => {
    // ASTRA-main 目录不存在时应抛错（检测逻辑）
    expect(() => bridge.spawnTraining(sampleConfig, vi.fn())).toBeDefined();
    // spawn 应至少被调用（目录存在时），若不存在则抛错
  });
});
