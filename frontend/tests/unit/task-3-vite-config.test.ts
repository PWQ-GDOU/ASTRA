/**
 * Task 3 测试：Vite 构建配置（dev + prod + electron-builder 联动）
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');

describe('Task 3: Vite 构建配置', () => {
  it('test_vite_config_has_react_plugin', () => {
    const config = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf-8');
    expect(config).toContain('@vitejs/plugin-react');
    expect(config).toContain('react(');
  });

  it('test_vite_dev_server_port_5173', () => {
    const config = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf-8');
    expect(config).toMatch(/port.*5173/);
  });

  it('test_vite_base_is_relative', () => {
    const config = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf-8');
    expect(config).toMatch(/base.*['"]\.\/['"]/);
  });

  it('test_vite_outdir_is_dist', () => {
    const config = readFileSync(resolve(ROOT, 'vite.config.ts'), 'utf-8');
    expect(config).toMatch(/outDir.*['"]dist['"]/);
  });

  it('test_electron_dev_script_exists', () => {
    expect(existsSync(resolve(ROOT, 'scripts', 'electron-dev.js'))).toBe(true);
  });

  it('test_tsconfig_electron_exists', () => {
    expect(existsSync(resolve(ROOT, 'tsconfig.electron.json'))).toBe(true);
  });

  it('test_tsconfig_electron_module_commonjs', () => {
    const config = JSON.parse(readFileSync(resolve(ROOT, 'tsconfig.electron.json'), 'utf-8'));
    expect(config.compilerOptions.module).toBe('commonjs');
  });

  it('test_tsconfig_electron_target_es2022', () => {
    const config = JSON.parse(readFileSync(resolve(ROOT, 'tsconfig.electron.json'), 'utf-8'));
    expect(config.compilerOptions.target).toBe('ES2022');
  });

  it('test_electron_dev_script_handles_vite_url', () => {
    const script = readFileSync(resolve(ROOT, 'scripts', 'electron-dev.js'), 'utf-8');
    expect(script).toContain('VITE_DEV_SERVER_URL');
  });

  it('test_tsconfig_electron_outdir', () => {
    const config = JSON.parse(readFileSync(resolve(ROOT, 'tsconfig.electron.json'), 'utf-8'));
    expect(config.compilerOptions.outDir).toBe('dist-electron');
  });
});
