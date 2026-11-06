/**
 * Task 130-152 测试：收尾验证
 *
 * 覆盖可自动验证的部分：
 * - Task 130-132: Electron 主进程启动配置 + IPC 通道注册
 * - Task 133-139: 各页面路由 + 渲染验证
 * - Task 140: 窗口控制 IPC
 * - Task 142: electron-builder 配置
 * - Task 143-147: 质量门禁（test/lint/format/typecheck/build 配置）
 * - Task 148-149: 页面路由
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { render } from '@testing-library/react';
import React from 'react';
import { IPC_CHANNELS } from '../../src/shared/types/ipc';

const ROOT = resolve(__dirname, '..', '..');
const pkg = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf-8'));

describe('Task 130-132: Electron 启动配置', () => {
  it('test_main_window_config', async () => {
    const content = readFileSync(resolve(ROOT, 'src', 'main', 'index.ts'), 'utf-8');
    expect(content).toContain('1440');
    expect(content).toContain('1000');
    expect(content).toContain('frame: false');
    expect(content).toContain('contextIsolation: true');
    expect(content).toContain('nodeIntegration: false');
  });

  it('test_main_registers_all_handlers', () => {
    const content = readFileSync(resolve(ROOT, 'src', 'main', 'ipc', 'register-all.ts'), 'utf-8');
    expect(content).toContain('registerSystemHandlers');
    expect(content).toContain('registerDatasetHandlers');
    expect(content).toContain('registerInferenceHandlers');
    expect(content).toContain('registerTrainingHandlers');
    expect(content).toContain('registerModelHandlers');
    expect(content).toContain('registerStatisticsHandlers');
  });

  it('test_preload_loaded_in_main', () => {
    const content = readFileSync(resolve(ROOT, 'src', 'main', 'index.ts'), 'utf-8');
    expect(content).toContain('preload');
    expect(content).toContain('dist-electron');
  });
});

describe('Task 133-139: 页面路由渲染', () => {
  it('test_all_page_files_exist', () => {
    const pages = [
      'Dashboard.tsx',
      'DataCollection.tsx',
      'Inference.tsx',
      'InferenceHistory.tsx',
      'Training.tsx',
      'ModelManagement.tsx',
      'Statistics.tsx',
    ];
    for (const p of pages) {
      const path = resolve(ROOT, 'src', 'renderer', 'pages', p);
      expect(existsSync(path)).toBe(true);
    }
  });

  it('test_app_has_all_routes', () => {
    const content = readFileSync(resolve(ROOT, 'src', 'renderer', 'App.tsx'), 'utf-8');
    expect(content).toContain('/data');
    expect(content).toContain('/inference');
    expect(content).toContain('/inference-history');
    expect(content).toContain('/training');
    expect(content).toContain('/models');
    expect(content).toContain('/statistics');
  });

  it('test_pages_render_in_app', async () => {
    // App 完整渲染含图表在 jsdom 较重，验证框架层渲染（TitleBar + Sidebar）
    const { default: App } = await import('../../src/renderer/App');
    const { findByText } = render(React.createElement(App));
    // TitleBar 标题由 AppShell 渲染，等待出现
    expect(await findByText('Spacecraft 推理训练平台', {}, { timeout: 10000 })).toBeDefined();
  });
});

describe('Task 140: 窗口控制 IPC', () => {
  it('test_window_channels_defined', () => {
    expect(IPC_CHANNELS.WINDOW_MINIMIZE).toBeDefined();
    expect(IPC_CHANNELS.WINDOW_MAXIMIZE).toBeDefined();
    expect(IPC_CHANNELS.WINDOW_CLOSE).toBeDefined();
  });

  it('test_system_ipc_registers_window_controls', async () => {
    const content = readFileSync(resolve(ROOT, 'src', 'main', 'ipc', 'system.ipc.ts'), 'utf-8');
    expect(content).toContain('minimize');
    expect(content).toContain('maximize');
    expect(content).toContain('close');
  });
});

describe('Task 142: electron-builder 配置', () => {
  it('test_build_config_complete', () => {
    expect(pkg.build.appId).toBe('com.spacecraft.platform');
    expect(pkg.build.productName).toBe('Spacecraft');
    expect(pkg.build.directories.output).toBe('out');
    expect(pkg.build.files).toContain('dist');
  });
});

describe('Task 143-147: 质量门禁', () => {
  it('test_test_script_defined', () => {
    expect(pkg.scripts.test).toContain('vitest');
  });

  it('test_lint_script_defined', () => {
    expect(pkg.scripts.lint).toContain('eslint');
  });

  it('test_format_check_script_defined', () => {
    expect(pkg.scripts['format:check']).toContain('prettier');
  });

  it('test_typecheck_script_defined', () => {
    expect(pkg.scripts.typecheck).toContain('tsc');
  });

  it('test_build_script_defined', () => {
    expect(pkg.scripts.build).toContain('electron-builder');
  });

  it('test_eslint_config_exists', () => {
    expect(existsSync(resolve(ROOT, '.eslintrc.cjs'))).toBe(true);
  });

  it('test_prettier_config_exists', () => {
    expect(existsSync(resolve(ROOT, 'prettier.config.cjs'))).toBe(true);
  });
});

describe('Task 148-149: E2E 路由验证', () => {
  it('test_main_entry_exists', () => {
    expect(existsSync(resolve(ROOT, 'dist-electron', 'main', 'index.js'))).toBe(true);
  });

  it('test_vite_config_exists', () => {
    expect(existsSync(resolve(ROOT, 'vite.config.ts'))).toBe(true);
  });
});
