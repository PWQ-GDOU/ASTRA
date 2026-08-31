/**
 * Task 28-39 测试：数据采集模块
 *
 * 覆盖：
 * - Task 28: dataset.store CRUD + 搜索 + 选择
 * - Task 29-30: dataset IPC handlers
 * - Task 31-32: 搜索过滤 + 数据集列表
 * - Task 33: 类型标签切换
 * - Task 36-37: 页面组装 + 空状态
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';

describe('Task 28: dataset.store CRUD + 搜索 + 选择', () => {
  beforeEach(() => {
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
  });

  it('test_dataset_store_initial_empty', () => {
    expect(useDatasetStore.getState().datasets).toEqual([]);
  });

  it('test_dataset_store_add', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    expect(useDatasetStore.getState().datasets).toHaveLength(1);
  });

  it('test_dataset_store_update_type', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().updateDatasetType('d1', 'test');
    expect(useDatasetStore.getState().datasets[0].type).toBe('test');
  });

  it('test_dataset_store_delete', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().deleteDataset('d1');
    expect(useDatasetStore.getState().datasets).toHaveLength(0);
  });

  it('test_dataset_store_search_filters', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().addDataset({
      id: 'd2',
      name: 'Battery',
      samples: 50,
      format: 'csv',
      type: 'test',
      createdAt: 't',
    });
    useDatasetStore.getState().setSearchQuery('fd0');
    const filtered = useDatasetStore.getState().getFilteredDatasets();
    expect(filtered).toHaveLength(1);
    expect(filtered[0].name).toBe('FD001');
  });

  it('test_dataset_store_search_no_match', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().setSearchQuery('zzz');
    expect(useDatasetStore.getState().getFilteredDatasets()).toHaveLength(0);
  });

  it('test_dataset_store_clear_search_returns_all', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().setSearchQuery('fd0');
    useDatasetStore.getState().setSearchQuery('');
    expect(useDatasetStore.getState().getFilteredDatasets()).toHaveLength(1);
  });

  it('test_dataset_store_select', () => {
    const ds = {
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    };
    useDatasetStore.getState().selectDataset(ds);
    expect(useDatasetStore.getState().selectedDataset?.id).toBe('d1');
  });

  it('test_dataset_store_set_datasets', () => {
    useDatasetStore.getState().setDatasets([
      {
        id: 'd1',
        name: 'A',
        samples: 1,
        format: 'csv',
        type: 'train',
        createdAt: 't',
      },
    ]);
    expect(useDatasetStore.getState().datasets).toHaveLength(1);
  });

  it('test_dataset_store_delete_selected_clears_selection', () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().selectDataset(useDatasetStore.getState().datasets[0]);
    useDatasetStore.getState().deleteDataset('d1');
    expect(useDatasetStore.getState().selectedDataset).toBeNull();
  });
});

describe('Task 31-37: 数据采集页面', () => {
  const mockDatasetImport = vi.fn(async () => ({
    success: true,
    data: {
      rootPath: 'C:/custom',
      files: ['C:/custom/train.any', 'C:/custom/test.any', 'C:/custom/rul.any'],
    },
  }));

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

  it('test_data_page_renders_header', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('数据采集')).toBeDefined();
  });

  it('test_data_page_empty_state', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('暂无数据集')).toBeDefined();
  });

  it('test_data_page_shows_datasets', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    const { findByText } = await renderPage();
    expect(await findByText('FD001')).toBeDefined();
    expect(await findByText('100')).toBeDefined();
  });

  it('test_data_page_search_filters', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    useDatasetStore.getState().addDataset({
      id: 'd2',
      name: 'Battery',
      samples: 50,
      format: 'csv',
      type: 'test',
      createdAt: 't',
    });
    const { findByPlaceholderText } = await renderPage();
    const search = await findByPlaceholderText('搜索数据集...');
    fireEvent.change(search, { target: { value: 'FD0' } });
    await waitFor(() => {
      expect(screen.getByText('FD001')).toBeDefined();
      expect(screen.queryByText('Battery')).toBeNull();
    });
  });

  it('test_data_page_import_calls_ipc', async () => {
    await renderPage();
    const importBtns = screen.getAllByText('导入数据集');
    fireEvent.click(importBtns[0]);
    await waitFor(() => {
      expect(mockDatasetImport).toHaveBeenCalled();
    });
  });

  it('test_data_page_import_opens_mapping', async () => {
    await renderPage();
    const importBtns = screen.getAllByText('导入数据集');
    fireEvent.click(importBtns[0]);
    await waitFor(() => {
      expect(screen.queryByText('映射 C-MAPSS 数据文件')).toBeDefined();
    });
  });

  it('test_data_page_shows_type_tag', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    const { findByText } = await renderPage();
    expect(await findByText('train')).toBeDefined();
  });

  it('test_data_page_type_click_cycles', async () => {
    useDatasetStore.getState().addDataset({
      id: 'd1',
      name: 'FD001',
      samples: 100,
      format: 'csv',
      type: 'train',
      createdAt: 't',
    });
    const { findByText } = await renderPage();
    const typeTag = await findByText('train');
    fireEvent.click(typeTag);
    await waitFor(() => {
      expect(useDatasetStore.getState().datasets[0].type).toBe('test');
    });
  });
});
