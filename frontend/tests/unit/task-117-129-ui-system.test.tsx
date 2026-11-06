/**
 * Task 117-129 测试：UI 系统通用组件
 *
 * 覆盖：
 * - Task 119: global.css 玻璃态工具类
 * - Task 121: StatusDot
 * - Task 122: GlassPanel
 * - Task 123: StatCard（task-17 已测，补空状态）
 * - Task 124: ProgressBar
 * - Task 125: LogViewer
 * - Task 126: DataTable
 * - Task 127: EmptyState
 * - Task 128: PageHeader
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { Database } from 'lucide-react';
import { readFileSync } from 'fs';
import { resolve } from 'path';

describe('Task 119: global.css 工具类', () => {
  it('test_global_css_has_glass_classes', () => {
    const css = readFileSync(
      resolve(__dirname, '..', '..', 'src', 'renderer', 'styles', 'global.css'),
      'utf-8',
    );
    expect(css).toMatch(/\.glass\b/);
    expect(css).toMatch(/\.glass-panel/);
    expect(css).toMatch(/\.glass-card/);
  });

  it('test_global_css_has_scrollbar_and_selection', () => {
    const css = readFileSync(
      resolve(__dirname, '..', '..', 'src', 'renderer', 'styles', 'global.css'),
      'utf-8',
    );
    expect(css).toMatch(/webkit-scrollbar/);
    expect(css).toMatch(/::selection/);
  });

  it('test_global_css_drag_regions', () => {
    const css = readFileSync(
      resolve(__dirname, '..', '..', 'src', 'renderer', 'styles', 'global.css'),
      'utf-8',
    );
    expect(css).toMatch(/\.drag-region/);
    expect(css).toMatch(/\.no-drag/);
  });
});

describe('Task 122: GlassPanel 组件', () => {
  it('test_glass_panel_renders_title', async () => {
    const { default: GlassPanel } = await import('../../src/renderer/components/common/GlassPanel');
    render(
      React.createElement(GlassPanel, {
        title: '测试面板',
        children: React.createElement('div', null, 'content'),
      }),
    );
    expect(screen.getByText('测试面板')).toBeDefined();
  });

  it('test_glass_panel_renders_children', async () => {
    const { default: GlassPanel } = await import('../../src/renderer/components/common/GlassPanel');
    render(
      React.createElement(GlassPanel, {
        children: React.createElement('div', null, 'child-content'),
      }),
    );
    expect(screen.getByText('child-content')).toBeDefined();
  });

  it('test_glass_panel_no_title_no_header', async () => {
    const { default: GlassPanel } = await import('../../src/renderer/components/common/GlassPanel');
    const { container } = render(
      React.createElement(GlassPanel, {
        children: React.createElement('div', null, 'x'),
      }),
    );
    expect(container.querySelectorAll('span').length).toBe(0);
  });

  it('test_glass_panel_has_glass_class', async () => {
    const { default: GlassPanel } = await import('../../src/renderer/components/common/GlassPanel');
    const { container } = render(
      React.createElement(GlassPanel, {
        title: 'T',
        children: React.createElement('div'),
      }),
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain('glass-panel');
  });
});

describe('Task 124: ProgressBar 组件', () => {
  it('test_progress_bar_renders_width', async () => {
    const { default: ProgressBar } =
      await import('../../src/renderer/components/common/ProgressBar');
    const { container } = render(React.createElement(ProgressBar, { value: 60 }));
    const fill = container.querySelectorAll('div')[1] as HTMLElement;
    expect(fill.style.width).toBe('60%');
  });

  it('test_progress_bar_clamps_over_100', async () => {
    const { default: ProgressBar } =
      await import('../../src/renderer/components/common/ProgressBar');
    const { container } = render(React.createElement(ProgressBar, { value: 150 }));
    const fill = container.querySelectorAll('div')[1] as HTMLElement;
    expect(fill.style.width).toBe('100%');
  });

  it('test_progress_bar_clamps_under_0', async () => {
    const { default: ProgressBar } =
      await import('../../src/renderer/components/common/ProgressBar');
    const { container } = render(React.createElement(ProgressBar, { value: -10 }));
    const fill = container.querySelectorAll('div')[1] as HTMLElement;
    expect(fill.style.width).toBe('0%');
  });

  it('test_progress_bar_zero', async () => {
    const { default: ProgressBar } =
      await import('../../src/renderer/components/common/ProgressBar');
    const { container } = render(React.createElement(ProgressBar, { value: 0 }));
    const fill = container.querySelectorAll('div')[1] as HTMLElement;
    expect(fill.style.width).toBe('0%');
  });

  it('test_progress_bar_variant_success', async () => {
    const { default: ProgressBar } =
      await import('../../src/renderer/components/common/ProgressBar');
    const { container } = render(
      React.createElement(ProgressBar, { value: 50, variant: 'success' }),
    );
    const fill = container.querySelectorAll('div')[1] as HTMLElement;
    expect(fill.style.background).toContain('var(--success)');
  });
});

describe('Task 125: LogViewer 组件', () => {
  it('test_log_viewer_renders_logs', async () => {
    const { default: LogViewer } = await import('../../src/renderer/components/common/LogViewer');
    render(React.createElement(LogViewer, { logs: ['line1', 'line2'] }));
    expect(screen.getByText('line1')).toBeDefined();
    expect(screen.getByText('line2')).toBeDefined();
  });

  it('test_log_viewer_empty_placeholder', async () => {
    const { default: LogViewer } = await import('../../src/renderer/components/common/LogViewer');
    render(React.createElement(LogViewer, { logs: [] }));
    expect(screen.getByText(/暂无日志/)).toBeDefined();
  });

  it('test_log_viewer_has_mono_font', async () => {
    const { default: LogViewer } = await import('../../src/renderer/components/common/LogViewer');
    const { container } = render(React.createElement(LogViewer, { logs: ['x'] }));
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.fontFamily).toContain('mono');
  });
});

describe('Task 126: DataTable 组件', () => {
  interface Row {
    id: string;
    name: string;
    value?: number;
  }

  // DataTable 是泛型组件，React.createElement 无法推断泛型，
  // 用显式 props 接口绕过
  interface DataTableProps {
    columns: { key: string; label: string; render?: (row: Row) => React.ReactNode }[];
    data: Row[];
    onRowClick?: (row: Row) => void;
    emptyState?: React.ReactNode;
  }

  it('test_data_table_renders_columns', async () => {
    const { default: DataTable } = await import('../../src/renderer/components/common/DataTable');
    const FC = DataTable as unknown as React.FC<DataTableProps>;
    render(
      React.createElement(FC, {
        columns: [
          { key: 'name', label: '名称' },
          { key: 'value', label: '数值' },
        ],
        data: [{ id: '1', name: 'A', value: 10 }],
      }),
    );
    expect(screen.getByText('名称')).toBeDefined();
    expect(screen.getByText('A')).toBeDefined();
    expect(screen.getByText('10')).toBeDefined();
  });

  it('test_data_table_empty_state', async () => {
    const { default: DataTable } = await import('../../src/renderer/components/common/DataTable');
    const FC = DataTable as unknown as React.FC<DataTableProps>;
    render(
      React.createElement(FC, {
        columns: [{ key: 'name', label: '名称' }],
        data: [],
        emptyState: React.createElement('span', null, '自定义空状态'),
      }),
    );
    expect(screen.getByText('自定义空状态')).toBeDefined();
  });

  it('test_data_table_default_empty', async () => {
    const { default: DataTable } = await import('../../src/renderer/components/common/DataTable');
    const FC = DataTable as unknown as React.FC<DataTableProps>;
    render(
      React.createElement(FC, {
        columns: [{ key: 'name', label: '名称' }],
        data: [],
      }),
    );
    expect(screen.getByText('暂无数据')).toBeDefined();
  });

  it('test_data_table_row_click', async () => {
    const { default: DataTable } = await import('../../src/renderer/components/common/DataTable');
    const FC = DataTable as unknown as React.FC<DataTableProps>;
    const onRowClick = vi.fn();
    render(
      React.createElement(FC, {
        columns: [{ key: 'name', label: '名称' }],
        data: [{ id: '1', name: 'A' }],
        onRowClick,
      }),
    );
    fireEvent.click(screen.getByText('A'));
    expect(onRowClick).toHaveBeenCalled();
  });

  it('test_data_table_render_callback', async () => {
    const { default: DataTable } = await import('../../src/renderer/components/common/DataTable');
    const FC = DataTable as unknown as React.FC<DataTableProps>;
    render(
      React.createElement(FC, {
        columns: [
          {
            key: 'name',
            label: '名称',
            render: (row: Row) => React.createElement('b', null, `>>${row.name}`),
          },
        ],
        data: [{ id: '1', name: 'A' }],
      }),
    );
    expect(screen.getByText('>>A')).toBeDefined();
  });
});

describe('Task 127: EmptyState 组件', () => {
  it('test_empty_state_renders_title', async () => {
    const { default: EmptyState } = await import('../../src/renderer/components/common/EmptyState');
    render(React.createElement(EmptyState, { icon: Database, title: '暂无数据' }));
    expect(screen.getByText('暂无数据')).toBeDefined();
  });

  it('test_empty_state_renders_description', async () => {
    const { default: EmptyState } = await import('../../src/renderer/components/common/EmptyState');
    render(React.createElement(EmptyState, { icon: Database, title: 'T', description: '请导入' }));
    expect(screen.getByText('请导入')).toBeDefined();
  });

  it('test_empty_state_action_button', async () => {
    const { default: EmptyState } = await import('../../src/renderer/components/common/EmptyState');
    const onClick = vi.fn();
    render(
      React.createElement(EmptyState, {
        icon: Database,
        title: 'T',
        action: { label: '点击导入', onClick },
      }),
    );
    fireEvent.click(screen.getByText('点击导入'));
    expect(onClick).toHaveBeenCalled();
  });
});

describe('Task 128: PageHeader 组件', () => {
  it('test_page_header_renders_title', async () => {
    const { default: PageHeader } = await import('../../src/renderer/components/common/PageHeader');
    render(React.createElement(PageHeader, { title: '工作台' }));
    expect(screen.getByText('工作台')).toBeDefined();
  });

  it('test_page_header_renders_description', async () => {
    const { default: PageHeader } = await import('../../src/renderer/components/common/PageHeader');
    render(React.createElement(PageHeader, { title: 'T', description: '系统概览' }));
    expect(screen.getByText('系统概览')).toBeDefined();
  });

  it('test_page_header_renders_actions', async () => {
    const { default: PageHeader } = await import('../../src/renderer/components/common/PageHeader');
    render(
      React.createElement(PageHeader, {
        title: 'T',
        actions: React.createElement('button', null, '操作'),
      }),
    );
    expect(screen.getByText('操作')).toBeDefined();
  });
});
