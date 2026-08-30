/**
 * Task 6 测试：完整目录结构搭建
 */
import { describe, it, expect } from 'vitest';
import { existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');
const SRC = resolve(ROOT, 'src');

describe('Task 6: 完整目录结构搭建', () => {
  it('test_main_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'main'))).toBe(true);
  });

  it('test_preload_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'preload'))).toBe(true);
  });

  it('test_renderer_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer'))).toBe(true);
  });

  it('test_shared_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'shared'))).toBe(true);
  });

  it('test_pages_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer', 'pages'))).toBe(true);
  });

  it('test_stores_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer', 'stores'))).toBe(true);
  });

  it('test_layout_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer', 'components', 'layout'))).toBe(true);
  });

  it('test_common_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer', 'components', 'common'))).toBe(true);
  });

  it('test_styles_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'renderer', 'styles'))).toBe(true);
  });

  it('test_shared_types_dir_exists', () => {
    expect(existsSync(resolve(SRC, 'shared', 'types'))).toBe(true);
  });
});
