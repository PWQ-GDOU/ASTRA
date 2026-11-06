/**
 * Task 34b 测试：数据集页面多选导入 + 展开收缩
 *
 * 覆盖用户反馈的功能改进：
 * - 一次多选多个文件归为一个数据集
 * - 数据集行可展开/收缩显示包含的文件
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import React from 'react';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';

afterEach(cleanup);

describe('数据集页面多选导入 + 展开收缩', () => {
  const mockDatasetImport = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    (window as unknown as Record<string, unknown>).electronAPI = {
      datasetImport: mockDatasetImport,
    };
  });

  async function renderPage() {
    const { default: DataCollection } = await import('../../src/renderer/pages/DataCollection');
    return render(React.createElement(DataCollection));
  }

  it('test_page_import_button_click', async () => {
    mockDatasetImport.mockResolvedValue({ success: true, data: { id: 'd1', name: 'FD001' } });
    await renderPage();
    const importBtns = screen.getAllByText('导入数据集');
    fireEvent.click(importBtns[0]);
    await waitFor(() => expect(mockDatasetImport).toHaveBeenCalled());
  });

  it('test_page_shows_dataset_with_multiple_files', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001 数据集',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt'],
    });
    const { findByText } = await renderPage();
    expect(await findByText('FD001 数据集')).toBeDefined();
    expect(await findByText('3 个文件')).toBeDefined();
  });

  it('test_page_expand_shows_files', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt', 'test_FD001.txt'],
    });
    const { findByText } = await renderPage();
    const row = await findByText('FD001');
    fireEvent.click(row);
    // 展开后应显示文件列表
    await waitFor(() => {
      expect(screen.getByText('train_FD001.txt')).toBeDefined();
      expect(screen.getByText('test_FD001.txt')).toBeDefined();
    });
  });

  it('test_page_collapse_hides_files', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt'],
    });
    const { findByText } = await renderPage();
    const row = await findByText('FD001');
    fireEvent.click(row); // expand
    await waitFor(() => expect(screen.getByText('train_FD001.txt')).toBeDefined());
    fireEvent.click(row); // collapse
    await waitFor(() => expect(screen.queryByText('train_FD001.txt')).toBeNull());
  });
});
