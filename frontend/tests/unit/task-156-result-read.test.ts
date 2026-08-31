/**
 * Task 156 测试：训练结果读取
 *
 * 验证 training-progress.ts 的 readTrainingResultFile 能正确读取
 * clean_benchmark.py 产出的单 seed 结果 JSON。
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { parsePrimarySeed, readTrainingResultFile } from '../../src/main/lib/training-progress';

const SAMPLE_RESULT = {
  seed: 42,
  best_epoch: 5,
  val_rmse: 13.363,
  test_rmse: 14.951,
  test_score: 308.66,
  test_rmse_cap: 13.785,
  test_score_cap: 277.61,
  n_params: 409921,
  seconds: 67.7,
  history: [
    { epoch: 1, loss: 4144.7, val_rmse: 26.85, val_score: 46658 },
    { epoch: 5, loss: 200.3, val_rmse: 13.36, val_score: 300 },
  ],
};

let tmpDir = '';

describe('Task 156: 训练结果读取', () => {
  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'task156-'));
    writeFileSync(
      join(tmpDir, 'FD002_multiscale_all24_s42.json'),
      JSON.stringify(SAMPLE_RESULT),
      'utf-8',
    );
  });

  afterEach(() => {
    if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
  });

  const read = (seed: number, fd: string, model: string) =>
    readTrainingResultFile(seed, fd, model, tmpDir, '');

  it('test_result_reads_json_file', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result).not.toBeNull();
  });

  it('test_result_parses_val_rmse', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result?.valRmse).toBeCloseTo(13.363);
  });

  it('test_result_parses_test_rmse', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result?.testRmse).toBeCloseTo(14.951);
  });

  it('test_result_parses_score', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result?.testScore).toBeCloseTo(308.66);
  });

  it('test_result_parses_history', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result?.history).toHaveLength(2);
    expect(result?.history[1].epoch).toBe(5);
    expect(result?.history[1].valRmse).toBeCloseTo(13.36);
  });

  it('test_result_parses_best_epoch', () => {
    const result = read(42, 'FD002', 'multiscale');
    expect(result?.bestEpoch).toBe(5);
  });

  it('test_result_missing_file_handled', () => {
    const result = read(99, 'FD999', 'v2');
    expect(result).toBeNull();
  });

  it('test_result_invalid_json_handled', () => {
    writeFileSync(join(tmpDir, 'FD002_v2_all24_s7.json'), 'not json{{', 'utf-8');
    const result = read(7, 'FD002', 'v2');
    expect(result).toBeNull();
  });

  it('test_result_uses_selected_sensor_mode', () => {
    writeFileSync(
      join(tmpDir, 'FD003_multiscale_selected14_s7.json'),
      JSON.stringify({ ...SAMPLE_RESULT, seed: 7 }),
      'utf-8',
    );
    const result = readTrainingResultFile(7, 'FD003', 'multiscale', tmpDir, '', 'selected14');
    expect(result?.seed).toBe(7);
  });

  it('test_primary_seed_uses_first_configured_seed', () => {
    expect(parsePrimarySeed('7, 42, 2026')).toBe(7);
  });
});
