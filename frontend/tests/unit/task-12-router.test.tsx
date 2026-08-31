/**
 * Task 12 测试：React Router v7 路由配置（7 个页面路由 + 导航切换）
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

(window as unknown as Record<string, unknown>).electronAPI = {
  minimizeWindow: vi.fn(),
  maximizeWindow: vi.fn(),
  closeWindow: vi.fn(),
  getPlatform: vi.fn(() => 'win32'),
  getElectronVersion: vi.fn(() => '40.0.0'),
};

describe('Task 12: React Router 路由配置', () => {
  it('test_app_renders_within_router', async () => {
    const { default: App } = await import('../../src/renderer/App');
    render(React.createElement(App));
    // 完整 App 渲染较重，异步等待标题出现
    expect(
      await screen.findByText('Spacecraft 推理训练平台', {}, { timeout: 15000 }),
    ).toBeDefined();
  });

  it('test_app_renders_titlebar_and_sidebar', async () => {
    const { default: App } = await import('../../src/renderer/App');
    render(React.createElement(App));
    expect(
      await screen.findByText('Spacecraft 推理训练平台', {}, { timeout: 15000 }),
    ).toBeDefined();
    // 版本号由 useEffect 异步设置，需等待渲染
    expect(await screen.findByText('v40.0.0', {}, { timeout: 15000 })).toBeDefined();
  });

  // Generic page render test
  for (const [route] of [
    ['/data', 'DataCollection'],
    ['/inference', 'Inference'],
    ['/inference-history', 'InferenceHistory'],
    ['/training', 'Training'],
    ['/models', 'ModelManagement'],
    ['/statistics', 'Statistics'],
  ]) {
    it(`test_page_placeholder_exists_${route.replace(/\//g, '_')}`, async () => {
      const { expect: e } = await import('vitest');
      // Pages exist as files
      const fs = await import('fs');
      const { resolve } = await import('path');
      const pageMap: Record<string, string> = {
        '/data': 'DataCollection.tsx',
        '/inference': 'Inference.tsx',
        '/inference-history': 'InferenceHistory.tsx',
        '/training': 'Training.tsx',
        '/models': 'ModelManagement.tsx',
        '/statistics': 'Statistics.tsx',
      };
      const pageFile = pageMap[route];
      const pagePath = resolve(__dirname, '..', '..', 'src', 'renderer', 'pages', pageFile);
      e(fs.existsSync(pagePath)).toBe(true);
    });
  }
});
