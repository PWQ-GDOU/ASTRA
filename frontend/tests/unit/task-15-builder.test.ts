/**
 * Task 15 测试：electron-builder 打包配置（NSIS + Portable）
 *
 * 验证 package.json 中 build 字段的打包配置。
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');
const pkg = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf-8'));
const build = pkg.build;

describe('Task 15: electron-builder 打包配置', () => {
  it('test_build_config_app_id_set', () => {
    expect(build.appId).toBe('com.spacecraft.platform');
  });

  it('test_build_config_product_name', () => {
    expect(build.productName).toBe('Spacecraft');
  });

  it('test_build_config_output_dir', () => {
    expect(build.directories.output).toBe('out');
  });

  it('test_build_config_nsis_target', () => {
    const targets = build.win.target.map((t: { target: string } | string) =>
      typeof t === 'string' ? t : t.target,
    );
    expect(targets).toContain('nsis');
  });

  it('test_build_config_portable_target', () => {
    const targets = build.win.target.map((t: { target: string } | string) =>
      typeof t === 'string' ? t : t.target,
    );
    expect(targets).toContain('portable');
  });

  it('test_build_config_nsis_oneclick_false', () => {
    expect(build.nsis.oneClick).toBe(false);
  });

  it('test_build_config_nsis_allow_change_dir', () => {
    expect(build.nsis.allowToChangeInstallationDirectory).toBe(true);
  });

  it('test_build_config_files_includes_dist', () => {
    expect(build.files).toContain('dist');
  });

  it('test_build_config_icon_set', () => {
    expect(build.win.icon).toBe('assets/icon.ico');
  });
});
