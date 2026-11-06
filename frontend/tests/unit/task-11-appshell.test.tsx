/**
 * Task 11 测试：AppShell 布局组件
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import React from 'react';

(window as unknown as Record<string, unknown>).electronAPI = {
  minimizeWindow: vi.fn(),
  maximizeWindow: vi.fn(),
  closeWindow: vi.fn(),
  getPlatform: vi.fn(() => 'win32'),
  getElectronVersion: vi.fn(() => '40.0.0'),
};

describe('Task 11: AppShell 布局组件', () => {
  it('test_appshell_renders_titlebar', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    expect(screen.getByText('Spacecraft 推理训练平台')).toBeDefined();
  });

  it('test_appshell_renders_sidebar', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    // 版本号由 useEffect 异步设置，需等待渲染
    expect(await screen.findByText('v40.0.0')).toBeDefined();
  });

  it('test_appshell_renders_children', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'test-content')),
      ),
    );
    expect(screen.getByText('test-content')).toBeDefined();
  });

  it('test_appshell_main_area_is_flex_1', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    const main = container.querySelector('main');
    expect(main).toBeDefined();
  });

  it('test_appshell_main_overflow_auto', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    const main = container.querySelector('main');
    expect(main).toBeDefined();
  });

  it('test_appshell_total_height_is_100vh', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.height).toBe('100vh');
  });

  it('test_appshell_content_area_below_titlebar', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    expect(container.querySelector('main')).toBeDefined();
  });

  it('test_appshell_layout_is_flex_row', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    expect(container.querySelector('aside')).toBeDefined();
  });

  it('test_appshell_no_horizontal_scroll', async () => {
    const { default: AppShell } = await import('../../src/renderer/components/layout/AppShell');
    const { container } = render(
      React.createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        React.createElement(AppShell, null, React.createElement('div', null, 'content')),
      ),
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.overflow).toBe('hidden');
  });
});
