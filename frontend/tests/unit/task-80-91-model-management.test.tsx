/**
 * Task 80-91 测试：模型管理模块
 *
 * 覆盖：
 * - Task 80: model.store（CRUD/搜索/来源筛选/默认模型）
 * - Task 82-86: 模型列表 + 卡片 + 详情面板 + 指标
 * - Task 88-90: 删除 + 页面组装
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { useModelStore } from '../../src/renderer/stores/model.store';

describe('Task 80: model.store', () => {
  beforeEach(() => {
    useModelStore.setState({
      models: [],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
  });

  it('test_model_store_initial_empty', () => {
    expect(useModelStore.getState().models).toEqual([]);
  });

  it('test_model_store_add_model', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'model-a',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    expect(useModelStore.getState().models).toHaveLength(1);
  });

  it('test_model_store_delete', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'model-a',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().deleteModel('m1');
    expect(useModelStore.getState().models).toHaveLength(0);
  });

  it('test_model_store_search', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'TCNModel',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().addModel({
      id: 'm2',
      name: 'LSTMModel',
      type: 'lstm',
      status: 'available',
      source: 'trained',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().setSearchQuery('tcn');
    const filtered = useModelStore.getState().getFilteredModels();
    expect(filtered).toHaveLength(1);
    expect(filtered[0].name).toBe('TCNModel');
  });

  it('test_model_store_source_filter', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'A',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().addModel({
      id: 'm2',
      name: 'B',
      type: 'tcn',
      status: 'available',
      source: 'trained',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().setSourceFilter('trained');
    expect(useModelStore.getState().getFilteredModels()).toHaveLength(1);
  });

  it('test_model_store_set_default', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'A',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().addModel({
      id: 'm2',
      name: 'B',
      type: 'tcn',
      status: 'available',
      source: 'trained',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().setDefaultModel('m2');
    const models = useModelStore.getState().models;
    expect(models.find((m) => m.id === 'm2')?.isDefault).toBe(true);
    expect(models.find((m) => m.id === 'm1')?.isDefault).toBe(false);
  });

  it('test_model_store_select', () => {
    const m = {
      id: 'm1',
      name: 'A',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    };
    useModelStore.getState().selectModel(m);
    expect(useModelStore.getState().selectedModel?.id).toBe('m1');
  });

  it('test_model_store_delete_selected_clears', () => {
    useModelStore.getState().addModel({
      id: 'm1',
      name: 'A',
      type: 'tcn',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/p',
      createdAt: 't',
    });
    useModelStore.getState().selectModel(useModelStore.getState().models[0]);
    useModelStore.getState().deleteModel('m1');
    expect(useModelStore.getState().selectedModel).toBeNull();
  });

  it('keeps the selected model detail synchronized when models are refreshed', () => {
    const model = {
      id: 'm1',
      name: 'A',
      type: 'tcn',
      status: 'available' as const,
      source: 'imported' as const,
      isDefault: false,
      path: '/p',
      createdAt: 't',
    };
    useModelStore.setState({ models: [model], selectedModel: model });
    useModelStore.getState().setModels([
      {
        ...model,
        performance: {
          taskId: 'run-1',
          datasetId: 'd1',
          datasetName: 'FD001',
          rmse: 2,
          sampleCount: 100,
          seconds: 0.5,
          evaluatedAt: '2026-08-30T08:00:00.000Z',
        },
      },
    ]);

    expect(useModelStore.getState().selectedModel?.performance?.rmse).toBe(2);
  });
});

describe('Task 82-90: 模型管理页面', () => {
  const modelDelete = vi.fn().mockResolvedValue({ success: true });
  const modelImport = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    useModelStore.setState({
      models: [
        {
          id: 'm1',
          name: 'multiscale-tcn',
          type: 'tcn',
          status: 'available',
          source: 'trained',
          isDefault: true,
          path: '/models/m1',
          size: '2.1 MB',
          createdAt: '2026-08-01',
          metrics: {
            accuracy: 0.92,
            paramCount: '410K',
            framework: 'PyTorch',
            inputShape: '[B,60,24]',
          },
          performance: {
            taskId: 'run-1',
            datasetId: 'fd1',
            datasetName: 'FD001 E2E',
            rmse: 13.537,
            mae: 10.25,
            score: 2.48,
            sampleCount: 100,
            seconds: 0.742,
            evaluatedAt: '2026-08-30T08:00:00.000Z',
          },
        },
      ],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
    Object.defineProperty(window, 'electronAPI', {
      configurable: true,
      value: { modelDelete, modelImport },
    });
  });

  async function renderPage() {
    const { default: ModelManagement } = await import('../../src/renderer/pages/ModelManagement');
    return render(React.createElement(ModelManagement));
  }

  it('test_model_page_renders_header', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('模型管理')).toBeDefined();
  });

  it('test_model_page_shows_model', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('multiscale-tcn')).toBeDefined();
  });

  it('test_model_page_shows_default_badge', async () => {
    const { findByText } = await renderPage();
    expect(await findByText('DEFAULT')).toBeDefined();
  });

  it('test_model_page_empty_state', async () => {
    useModelStore.setState({ models: [], selectedModel: null });
    const { findByText } = await renderPage();
    expect(await findByText('暂无模型')).toBeDefined();
  });

  it('test_model_page_click_selects', async () => {
    const { findByText } = await renderPage();
    fireEvent.click(await findByText('multiscale-tcn'));
    await waitFor(() => {
      expect(useModelStore.getState().selectedModel?.id).toBe('m1');
    });
  });

  it('test_model_page_shows_detail', async () => {
    useModelStore.getState().selectModel(useModelStore.getState().models[0]);
    const { findByText } = await renderPage();
    expect(await findByText('/models/m1')).toBeDefined();
    expect(await findByText('92.0%')).toBeDefined();
    expect(await findByText('最近实测效果')).toBeDefined();
    expect(await findByText('FD001 E2E')).toBeDefined();
    expect(await findByText('13.537')).toBeDefined();
    expect(await findByText('100 个样本')).toBeDefined();
  });

  it('test_model_page_search_filters', async () => {
    useModelStore.getState().addModel({
      id: 'm2',
      name: 'LSTM-base',
      type: 'lstm',
      status: 'available',
      source: 'imported',
      isDefault: false,
      path: '/models/m2',
      createdAt: 't',
    });
    const { findByPlaceholderText } = await renderPage();
    const search = await findByPlaceholderText('搜索模型...');
    fireEvent.change(search, { target: { value: 'LSTM' } });
    await waitFor(() => {
      expect(screen.getByText('LSTM-base')).toBeDefined();
      expect(screen.queryByText('multiscale-tcn')).toBeNull();
    });
  });

  it('persists model deletion before removing it from the UI', async () => {
    useModelStore.getState().selectModel(useModelStore.getState().models[0]);
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    await renderPage();
    fireEvent.click(await screen.findByText('删除模型'));
    await waitFor(() => {
      expect(modelDelete).toHaveBeenCalledWith('m1');
      expect(useModelStore.getState().models).toHaveLength(0);
    });
  });

  it('does not report a user-cancelled model dialog as an import error', async () => {
    modelImport.mockResolvedValue({ success: false, error: 'File selection cancelled' });
    const alert = vi.spyOn(window, 'alert').mockImplementation(() => {});
    await renderPage();

    fireEvent.click(await screen.findByText('导入模型'));

    await waitFor(() => expect(modelImport).toHaveBeenCalled());
    expect(alert).not.toHaveBeenCalled();
  });
});
