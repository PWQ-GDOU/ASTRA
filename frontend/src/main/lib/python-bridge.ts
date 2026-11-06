import { spawn } from 'child_process';
import { existsSync } from 'fs';
import { resolve, join } from 'path';
import type { TrainingConfig } from '../../shared/types/training';

/** 训练脚本相对 ASTRA 根目录的路径 */
const TRAIN_SCRIPT = join('scripts', 'clean_benchmark.py');

/**
 * ASTRA 后端根目录。
 *
 * 开发目录已调整为 backend/ 与 frontend/ 同级；同时保留环境变量覆盖，
 * 方便安装版或自定义部署把后端放在其他位置。
 */
function resolveAstraDir(): string {
  if (process.env.ASTRA_DIR) return resolve(process.env.ASTRA_DIR);

  const ascendToBackend = (base: string) =>
    Array.from({ length: 7 }, (_, depth) =>
      resolve(base, ...Array.from({ length: depth }, () => '..'), 'backend'),
    );
  const resourcesPath = typeof process.resourcesPath === 'string' ? process.resourcesPath : '';
  const candidates = [
    resolve(process.cwd(), '..', 'backend'),
    resolve(process.cwd(), 'backend'),
    ...ascendToBackend(process.cwd()),
    ...ascendToBackend(__dirname),
    ...(resourcesPath ? ascendToBackend(resourcesPath) : []),
  ];
  return (
    candidates.find((candidate) => existsSync(resolve(candidate, TRAIN_SCRIPT))) || candidates[0]
  );
}

export const ASTRA_DIR = resolveAstraDir();

/** 结果输出相对 ASTRA 根目录的路径 */
export const OUTPUT_DIR = join('outputs', 'clean_benchmark');

export interface TrainingSpawn {
  /** 本次训练的 runId */
  runId: string;
  /** 终止子进程 */
  kill: () => void;
  /** 训练完成的 Promise（resolve 时训练已结束） */
  promise: Promise<{ exitCode: number | null }>;
}

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function createLineReceiver(onLine: (line: string) => void): {
  push: (chunk: Buffer) => void;
  flush: () => void;
} {
  let pending = '';
  const emit = (line: string) => {
    const normalized = line.replace(/\r$/, '');
    if (normalized.trim()) onLine(normalized);
  };
  return {
    push: (chunk) => {
      pending += chunk.toString();
      const lines = pending.split('\n');
      pending = lines.pop() || '';
      lines.forEach(emit);
    },
    flush: () => {
      emit(pending);
      pending = '';
    },
  };
}

/**
 * 检查 ASTRA 后端是否可运行（目录存在 + Python 可用由 spawn 失败时捕获）
 */
export function isAstraAvailable(): boolean {
  try {
    return existsSync(resolve(ASTRA_DIR, TRAIN_SCRIPT));
  } catch {
    return false;
  }
}

/**
 * 构造 clean_benchmark.py 的命令行参数。
 * 前端可自定义后端全部配置（不改后端，只是传参）。
 */
export function buildTrainingArgs(config: TrainingConfig): string[] {
  const h = config.hyperParams;
  return [
    TRAIN_SCRIPT,
    '--data-root',
    config.dataRoot || 'data/processed',
    '--fd',
    config.dataset || 'FD002',
    '--dataset-id',
    config.dataset,
    ...(config.trainFile && config.testFile && config.rulFile
      ? [
          '--train-file',
          config.trainFile,
          '--test-file',
          config.testFile,
          '--rul-file',
          config.rulFile,
        ]
      : []),
    '--model',
    config.modelArch || 'multiscale',
    '--sensor-mode',
    config.sensorMode || 'all24',
    '--seq-len',
    String(config.seqLen || 60),
    '--rul-cap',
    String(config.rulCap ?? 125.0),
    '--val-ratio',
    String(config.valRatio ?? 0.2),
    '--seeds',
    config.seeds || '42',
    '--split-seed',
    String(config.splitSeed ?? 2026),
    '--epochs',
    String(h.epochs || 10),
    '--patience',
    String(config.patience ?? 60),
    '--batch-size',
    String(h.batchSize || 64),
    '--lr',
    String(h.learningRate || 0.0005),
    '--over-weight',
    String(config.overWeight ?? 0.02),
    '--device',
    config.device || 'cuda',
    '--output',
    config.outputDir || OUTPUT_DIR,
  ];
}

/**
 * 启动 ASTRA 训练子进程。
 *
 * @param config 前端训练配置
 * @param onStdout 实时 stdout 回调（用于进度解析）
 * @param onStderr 实时 stderr 回调
 */
export function spawnTraining(
  config: TrainingConfig,
  onStdout: (line: string) => void,
  onStderr?: (line: string) => void,
): TrainingSpawn {
  const runId = genId();
  const args = buildTrainingArgs(config);

  if (!isAstraAvailable()) {
    const message = `ASTRA 后端不可用：${resolve(ASTRA_DIR, TRAIN_SCRIPT)}`;
    onStderr?.(message);
    return {
      runId,
      kill: () => {},
      promise: Promise.resolve({ exitCode: null }),
    };
  }

  const child = spawn(process.env.ASTRA_PYTHON || 'python', ['-u', ...args], {
    cwd: ASTRA_DIR,
    windowsHide: true,
  });

  const stdout = createLineReceiver(onStdout);
  const stderr = createLineReceiver((line) => onStderr?.(line));
  child.stdout?.on('data', stdout.push);
  child.stderr?.on('data', stderr.push);

  const promise = new Promise<{ exitCode: number | null }>((resolvePromise) => {
    child.on('close', (code) => {
      stdout.flush();
      stderr.flush();
      resolvePromise({ exitCode: code });
    });
    child.on('error', (err) => {
      onStderr?.(`[spawn error] ${err.message}`);
      resolvePromise({ exitCode: null });
    });
  });

  return {
    runId,
    kill: () => {
      if (!child.killed) {
        child.kill();
      }
    },
    promise,
  };
}
