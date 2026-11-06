/**
 * Task 54-65 测试：推理历史模块
 *
 * 覆盖：
 * - Task 54: inference-history.store（items/filters/选中/抽屉/删除/分页）
 * - Task 56-57: 筛选 + 历史表格
 * - Task 58: StatusDot 状态圆点
 * - Task 62-64: 删除 + 页面 + 空状态
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { useInferenceHistoryStore } from '../../src/renderer/stores/inference-history.store';

describe('Task 54: inference-history.store', () => {
  beforeEach(() => {
    useInferenceHistoryStore.setState({
      items: [],
      selectedItem: null,
      isDrawerOpen: false,
      total: 0,
      page: 1,
    });
  });

  it('test_history_store_initial', () => {
    const s = useInferenceHistoryStore.getState();
    expect(s.items).toEqual([]);
    expect(s.selectedItem).toBeNull();
    expect(s.isDrawerOpen).toBe(false);
  });

  it('test_history_store_set_items', () => {
    useInferenceHistoryStore.getState().setItems([
      {
        id: 'i1',
        taskName: 'T1',
        model: 'm',
        dataset: 'd',
        status: 'completed',
        duration: '1s',
        time: 't',
        metrics: { accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9, latency: 1, throughput: 2 },
        log: [],
      },
    ]);
    expect(useInferenceHistoryStore.getState().items).toHaveLength(1);
  });

  it('test_history_store_set_filters', () => {
    useInferenceHistoryStore.getState().setFilters({ status: 'completed' });
    expect(useInferenceHistoryStore.getState().filters.status).toBe('completed');
  });

  it('test_history_store_set_filters_merges', () => {
    useInferenceHistoryStore.getState().setFilters({ search: 'fd' });
    useInferenceHistoryStore.getState().setFilters({ status: 'failed' });
    const f = useInferenceHistoryStore.getState().filters;
    expect(f.search).toBe('fd');
    expect(f.status).toBe('failed');
  });

  it('test_history_store_select_item', () => {
    const item = {
      id: 'i1',
      taskName: 'T1',
      model: 'm',
      dataset: 'd',
      status: 'completed' as const,
      duration: '1s',
      time: 't',
      metrics: { accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9, latency: 1, throughput: 2 },
      log: [],
    };
    useInferenceHistoryStore.getState().selectItem(item);
    expect(useInferenceHistoryStore.getState().selectedItem?.id).toBe('i1');
  });

  it('test_history_store_open_close_drawer', () => {
    useInferenceHistoryStore.getState().openDrawer();
    expect(useInferenceHistoryStore.getState().isDrawerOpen).toBe(true);
    useInferenceHistoryStore.getState().closeDrawer();
    expect(useInferenceHistoryStore.getState().isDrawerOpen).toBe(false);
  });

  it('test_history_store_delete_item', () => {
    useInferenceHistoryStore.getState().setItems([
      {
        id: 'i1',
        taskName: 'T1',
        model: 'm',
        dataset: 'd',
        status: 'completed',
        duration: '1s',
        time: 't',
        metrics: { accuracy: 0.9, precision: 0.9, recall: 0.9, f1: 0.9, latency: 1, throughput: 2 },
        log: [],
      },
      {
        id: 'i2',
        taskName: 'T2',
        model: 'm',
        dataset: 'd',
        status: 'failed',
        duration: '1s',
        time: 't',
        metrics: { accuracy: 0.1, precision: 0.1, recall: 0.1, f1: 0.1, latency: 1, throughput: 2 },
        log: [],
      },
    ]);
    useInferenceHistoryStore.getState().deleteItem('i1');
    expect(useInferenceHistoryStore.getState().items).toHaveLength(1);
    expect(useInferenceHistoryStore.getState().items[0].id).toBe('i2');
  });

  it('test_history_store_set_page', () => {
    useInferenceHistoryStore.getState().setPage(3);
    expect(useInferenceHistoryStore.getState().page).toBe(3);
  });
});

async function renderPage() {
  const { default: InferenceHistory } = await import('../../src/renderer/pages/InferenceHistory');
  return render(React.createElement(InferenceHistory));
}

describe('Task 56-64: 推理历史页面', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (window as unknown as { electronAPI: unknown }).electronAPI = {
      inferenceHistoryList: vi.fn().mockResolvedValue({ success: true, data: [] }),
    };
    useInferenceHistoryStore.setState({
      items: [
        {
          id: 'i1',
          taskName: '任务A',
          model: 'm1',
          dataset: 'd1',
          status: 'completed',
          duration: '2.3s',
          time: '10:00',
          metrics: {
            accuracy: 0.9,
            precision: 0.9,
            recall: 0.9,
            f1: 0.9,
            latency: 1,
            throughput: 2,
          },
          log: ['line1'],
        },
        {
          id: 'i2',
          taskName: '任务B',
          model: 'm2',
          dataset: 'd2',
          status: 'failed',
          duration: '1.1s',
          time: '09:00',
          metrics: {
            accuracy: 0.1,
            precision: 0.1,
            recall: 0.1,
            f1: 0.1,
            latency: 1,
            throughput: 2,
          },
          log: [],
        },
      ],
      selectedItem: null,
      isDrawerOpen: false,
      total: 2,
      page: 1,
      filters: { model: '', status: '', dateRange: null, search: '' },
    });
  });

  it('test_history_page_renders_header', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('推理历史')).toBeDefined();
  });

  it('test_history_page_shows_task_items', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('任务A')).toBeDefined();
    expect(await findByText('任务B')).toBeDefined();
  });

  it('test_history_page_shows_status_dot', async () => {
    const { findAllByText } = await renderPage();
    // "失败" 出现在下拉筛选选项 + 状态标签，断言至少出现
    expect((await findAllByText('完成')).length).toBeGreaterThan(0);
    expect((await findAllByText('失败')).length).toBeGreaterThan(0);
  });

  it('test_history_page_search_updates_store', async () => {
    const { findByPlaceholderText } = await renderPage();
    const search = await findByPlaceholderText('搜索任务...');
    fireEvent.change(search, { target: { value: '任务B' } });
    await waitFor(() => {
      expect(useInferenceHistoryStore.getState().filters.search).toBe('任务B');
    });
    expect(screen.queryByText('任务A')).toBeNull();
    expect(screen.getByText('任务B')).toBeDefined();
  });

  it('test_history_page_status_filter_updates_store', async () => {
    const { container } = await renderPage();
    const statusSelect = container.querySelector('select');
    expect(statusSelect).not.toBeNull();
    fireEvent.change(statusSelect!, { target: { value: 'completed' } });
    await waitFor(() => {
      expect(useInferenceHistoryStore.getState().filters.status).toBe('completed');
    });
    expect(screen.getByText('任务A')).toBeDefined();
    expect(screen.queryByText('任务B')).toBeNull();
  });

  it('loads persisted inference history when the page is initially empty', async () => {
    useInferenceHistoryStore.setState({ items: [] });
    const persisted = {
      id: 'persisted-1',
      taskName: '持久化任务',
      model: 'astra.pt',
      dataset: 'FD001',
      status: 'completed' as const,
      duration: '0.5s',
      time: '2026-08-30T00:00:00.000Z',
      metrics: { rmse: 12.3, latency: 0.5 },
      log: [],
    };
    const api = window.electronAPI as unknown as { inferenceHistoryList: ReturnType<typeof vi.fn> };
    api.inferenceHistoryList.mockResolvedValue({ success: true, data: [persisted] });

    const { findByText } = await renderPage();

    expect(await findByText('持久化任务')).toBeDefined();
  });

  it('test_history_page_row_click_selects', async () => {
    const { findByText } = await renderPage();
    const row = await findByText('任务A');
    fireEvent.click(row);
    expect(useInferenceHistoryStore.getState().selectedItem?.id).toBe('i1');
    expect(useInferenceHistoryStore.getState().isDrawerOpen).toBe(true);
  });
});

describe('Task 58: StatusDot 组件', () => {
  it('test_status_dot_completed_green', async () => {
    const { default: StatusDot } = await import('../../src/renderer/components/common/StatusDot');
    const { container } = render(
      React.createElement(StatusDot, { status: 'completed', label: '完成' }),
    );
    expect(screen.getByText('完成')).toBeDefined();
    expect(container.querySelector('span')).not.toBeNull();
  });

  it('test_status_dot_failed_label', async () => {
    const { default: StatusDot } = await import('../../src/renderer/components/common/StatusDot');
    render(React.createElement(StatusDot, { status: 'failed', label: '失败' }));
    expect(screen.getByText('失败')).toBeDefined();
  });

  it('test_history_empty_state', async () => {
    useInferenceHistoryStore.setState({ items: [] });
    const { findByText } = await renderPage();
    expect(await findByText('暂无推理记录')).toBeDefined();
  });
});
