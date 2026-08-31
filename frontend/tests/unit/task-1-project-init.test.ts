/**
 * Task 1 测试：项目骨架搭建 — Electron + React + TypeScript + Vite
 *
 * 这些测试验证 package.json 和核心文件配置。
 * 运行环境：Node.js (vitest)
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');

function readPkg(): Record<string, unknown> | null {
  const p = resolve(ROOT, 'package.json');
  if (!existsSync(p)) return null;
  return JSON.parse(readFileSync(p, 'utf-8'));
}

describe('Task 1: 项目骨架搭建', () => {
  it('test_vite_project_has_correct_name', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    expect(pkg!.name).toBe('spacecraft');
  });

  it('test_all_core_deps_installed', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const allDeps = {
      ...((pkg!.dependencies as Record<string, string>) || {}),
      ...((pkg!.devDependencies as Record<string, string>) || {}),
    };
    const required = ['electron', 'react', 'react-dom', 'vite', 'zustand', 'tailwindcss'];
    for (const dep of required) {
      expect(allDeps[dep], `Missing dependency: ${dep}`).toBeDefined();
    }
  });

  it('test_dev_script_exists', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const scripts = (pkg!.scripts as Record<string, string>) || {};
    expect(scripts.dev).toBeDefined();
    expect(scripts.build).toBeDefined();
    expect(scripts.test).toBeDefined();
    expect(scripts.lint).toBeDefined();
    expect(scripts['format:check']).toBeDefined();
    expect(scripts.typecheck).toBeDefined();
  });

  it('test_electron_version_matches_spec', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const allDeps = {
      ...((pkg!.dependencies as Record<string, string>) || {}),
      ...((pkg!.devDependencies as Record<string, string>) || {}),
    };
    const ver = allDeps['electron'];
    expect(ver).toBeDefined();
    const major = parseInt((ver.match(/\d+/) || ['0'])[0], 10);
    expect(major).toBeGreaterThanOrEqual(40);
  });

  it('test_react_version_matches_spec', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const allDeps = {
      ...((pkg!.dependencies as Record<string, string>) || {}),
      ...((pkg!.devDependencies as Record<string, string>) || {}),
    };
    const ver = allDeps['react'];
    expect(ver).toBeDefined();
    const major = parseInt((ver.match(/\d+/) || ['0'])[0], 10);
    expect(major).toBeGreaterThanOrEqual(18);
  });

  it('test_typescript_version_matches_spec', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const allDeps = {
      ...((pkg!.dependencies as Record<string, string>) || {}),
      ...((pkg!.devDependencies as Record<string, string>) || {}),
    };
    const ver = allDeps['typescript'];
    expect(ver).toBeDefined();
    const major = parseInt((ver.match(/\d+/) || ['0'])[0], 10);
    expect(major).toBeGreaterThanOrEqual(5);
  });

  it('test_concurrently_installed', () => {
    const pkg = readPkg();
    expect(pkg).not.toBeNull();
    const devDeps = (pkg!.devDependencies as Record<string, string>) || {};
    expect(devDeps['concurrently']).toBeDefined();
  });

  it('test_vite_config_exists', () => {
    const p = resolve(ROOT, 'vite.config.ts');
    expect(existsSync(p)).toBe(true);
  });
});
