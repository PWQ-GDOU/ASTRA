/**
 * 训练进度 stdout 解析（纯函数，无副作用）
 *
 * 解析 ASTRA clean_benchmark.py 的 stdout 输出，
 * 供 training.ipc.ts 实时推进前端训练进度。
 */

export interface ParsedProgress {
  epoch: number;
  loss: number;
  valLoss?: number;
}

/** 解析 "E{epoch:03d} loss=.. val_rmse=.. best=.." 行 */
export function parseEpochLine(line: string): ParsedProgress | null {
  const m = line.match(/E(\d+)\s+loss=([\d.]+)\s+val_rmse=([\d.]+)/);
  if (!m) return null;
  return {
    epoch: parseInt(m[1], 10),
    loss: parseFloat(m[2]),
    valLoss: parseFloat(m[3]),
  };
}

/** 解析 "RESULT seed=42 val_rmse=.. test_rmse=.." 行 */
export function parseResultLine(
  line: string,
): { seed: number; valRmse: number; testRmse: number } | null {
  const m = line.match(/RESULT seed=(\d+)\s+val_rmse=([\d.]+)\s+test_rmse=([\d.]+)/);
  if (!m) return null;
  return {
    seed: parseInt(m[1], 10),
    valRmse: parseFloat(m[2]),
    testRmse: parseFloat(m[3]),
  };
}

/** 判断是否训练开始行 "=== seed N ===" */
export function isSeedStartLine(line: string): boolean {
  return /^===\s*seed\s*\d+\s*===/.test(line);
}

/* ===== 训练结果读取（任务 156）===== */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import type { TrainingResult } from '../../shared/types/training';

/**
 * 读取 clean_benchmark.py 产出的单 seed 结果 JSON。
 *
 * @param seed     种子号
 * @param fd       数据集（如 FD002）
 * @param model    模型（如 multiscale）
 * @param astraDir ASTRA 根目录（绝对路径）
 * @param outDir   输出相对目录（默认 outputs/clean_benchmark）
 */
export function readTrainingResultFile(
  seed: number,
  fd: string,
  model: string,
  astraDir: string,
  outDir = 'outputs/clean_benchmark',
  sensorMode = 'all24',
): TrainingResult | null {
  try {
    const filename = `${fd}_${model}_${sensorMode}_s${seed}.json`;
    const raw = readFileSync(resolve(astraDir, outDir, filename), 'utf-8');
    const data = JSON.parse(raw) as {
      seed: number;
      best_epoch: number;
      val_rmse: number;
      test_rmse: number;
      test_score: number;
      test_rmse_cap: number;
      test_score_cap: number;
      n_params: number;
      seconds: number;
      history: Array<{
        epoch: number;
        loss: number;
        val_rmse: number;
        val_score: number;
      }>;
    };
    return {
      runId: '',
      seed: data.seed,
      bestEpoch: data.best_epoch,
      valRmse: data.val_rmse,
      testRmse: data.test_rmse,
      testScore: data.test_score,
      testRmseCap: data.test_rmse_cap,
      testScoreCap: data.test_score_cap,
      nParams: data.n_params,
      seconds: data.seconds,
      history: (data.history || []).map((h) => ({
        epoch: h.epoch,
        loss: h.loss,
        valRmse: h.val_rmse,
        valScore: h.val_score,
      })),
    };
  } catch {
    return null;
  }
}

/** 返回多 seed 配置中用于结果页展示的首个 seed。 */
export function parsePrimarySeed(seeds: string): number {
  const first = seeds.split(',')[0]?.trim();
  const parsed = Number.parseInt(first, 10);
  return Number.isFinite(parsed) ? parsed : 42;
}
