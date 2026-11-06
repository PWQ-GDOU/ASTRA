/**
 * Task 155 测试：训练进度推送 — stdout 解析
 *
 * 验证 training.ipc.ts 的 parseEpochLine / parseResultLine / isSeedStartLine
 * 能从 clean_benchmark.py 的 stdout 正确解析进度。
 */
import { describe, it, expect } from 'vitest';
import {
  parseEpochLine,
  parseResultLine,
  isSeedStartLine,
} from '../../src/main/lib/training-progress';

describe('Task 155: 训练进度推送 — stdout 解析', () => {
  it('test_progress_parses_epoch_loss', () => {
    const parsed = parseEpochLine('E010 loss=1.2345 val_rmse=13.363 best=13.363');
    expect(parsed).not.toBeNull();
    expect(parsed?.epoch).toBe(10);
    expect(parsed?.loss).toBeCloseTo(1.2345);
  });

  it('test_progress_parses_val_loss', () => {
    const parsed = parseEpochLine('E001 loss=2.5000 val_rmse=26.850 best=26.850');
    expect(parsed?.valLoss).toBeCloseTo(26.85);
  });

  it('test_progress_parses_high_epoch', () => {
    const parsed = parseEpochLine('E125 loss=0.42 val_rmse=15.5 best=13.2');
    expect(parsed?.epoch).toBe(125);
  });

  it('test_progress_ignores_unrelated_lines', () => {
    expect(parseEpochLine('FD=FD002 model=v2 sensors=all24')).toBeNull();
    expect(parseEpochLine('train=(14168, 30, 24) val=(3563, 30, 24)')).toBeNull();
    expect(parseEpochLine('=== seed 42 ===')).toBeNull();
  });

  it('test_progress_ignores_empty_string', () => {
    expect(parseEpochLine('')).toBeNull();
  });

  it('test_progress_seed_start_line', () => {
    expect(isSeedStartLine('=== seed 42 ===')).toBe(true);
  });

  it('test_progress_seed_start_ignores_others', () => {
    expect(isSeedStartLine('E001 loss=1.0 val_rmse=2.0')).toBe(false);
    expect(isSeedStartLine('some random line')).toBe(false);
  });

  it('test_progress_parses_result_line', () => {
    const parsed = parseResultLine(
      'RESULT seed=42 val_rmse=13.363 test_rmse=14.951 test_score=308.7',
    );
    expect(parsed).not.toBeNull();
    expect(parsed?.seed).toBe(42);
    expect(parsed?.valRmse).toBeCloseTo(13.363);
    expect(parsed?.testRmse).toBeCloseTo(14.951);
  });

  it('test_progress_result_line_ignores_unrelated', () => {
    expect(parseResultLine('E010 loss=1.23 val_rmse=13.36')).toBeNull();
    expect(parseResultLine('RESULT something')).toBeNull();
  });
});
