/**
 * Task 34c 测试：数据集重命名 + 删除只删记录不删文件
 *
 * 覆盖用户反馈：
 * - 数据集名称可修改（重命名功能）
 * - 删除数据集只从软件中移除记录，不删除原文件
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import React from 'react';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';

afterEach(cleanup);

describe('数据集重命名', () => {
  const mockDatasetImport = vi.fn();
  const mockDatasetUpdateType = vi.fn();
  const mockDatasetRename = vi.fn().mockResolvedValue({ success: true });
  const mockDatasetDelete = vi.fn().mockResolvedValue({ success: true });

  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    (window as unknown as Record<string, unknown>).electronAPI = {
      datasetImport: mockDatasetImport,
      datasetUpdateType: mockDatasetUpdateType,
      datasetRename: mockDatasetRename,
      datasetDelete: mockDatasetDelete,
    };
  });

  async function renderPage() {
    const { default: DataCollection } = await import('../../src/renderer/pages/DataCollection');
    return render(React.createElement(DataCollection));
  }

  it('test_store_rename_dataset', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt'],
    });
    useDatasetStore.getState().renameDataset('d1', 'FD001-renamed');
    expect(useDatasetStore.getState().datasets[0].name).toBe('FD001-renamed');
  });

  it('test_page_has_rename_button', async () => {
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
    expect(await findByText('重命名')).toBeDefined();
  });

  it('test_page_rename_flow', async () => {
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
    // 点击重命名 → 出现输入框
    fireEvent.click(await findByText('重命名'));
    const input = (await waitFor(() =>
      screen.getByPlaceholderText('输入新名称'),
    )) as HTMLInputElement;
    // 输入新名称并确认
    fireEvent.change(input, { target: { value: 'FD002' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => {
      expect(useDatasetStore.getState().datasets[0].name).toBe('FD002');
      expect(mockDatasetRename).toHaveBeenCalledWith('d1', 'FD002');
    });
  });

  it('test_delete_confirms_record_only', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt'],
    });
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true);
    const { findByText } = await renderPage();
    // 点击删除按钮
    fireEvent.click(await findByText('删除'));
    // 确认提示应说明只删除记录，明确不删除原文件
    expect(confirmSpy).toHaveBeenCalled();
    const msg = confirmSpy.mock.calls[0][0] as string;
    expect(msg).toContain('记录');
    expect(msg).toContain('不会删除原文件');
    confirmSpy.mockRestore();
  });

  it('test_delete_removes_from_store', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'txt',
      type: 'train',
      createdAt: 't',
      files: ['train_FD001.txt'],
    });
    const confirmSpy = vi.spyOn(window, 'confirm').mockImplementation(() => true);
    const { findByText } = await renderPage();
    fireEvent.click(await findByText('删除'));
    await waitFor(() => {
      expect(useDatasetStore.getState().datasets).toHaveLength(0);
      expect(mockDatasetDelete).toHaveBeenCalledWith('d1');
    });
    confirmSpy.mockRestore();
  });
});
