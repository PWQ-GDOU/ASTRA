/**
 * Task: 远程连接 UI 测试（状态条 + 配置弹窗 + 密码/SSH切换）
 *
 * 覆盖：
 * - 训练页有「配置服务器」按钮
 * - 有服务器状态显示
 * - 点击配置按钮打开弹窗
 * - 弹窗支持密码/SSH密钥方式切换
 * - 测试连接调用 IPC
 * - 断开连接调用 IPC
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import React from 'react';
import { useTrainingStore } from '../../src/renderer/stores/training.store';
import { useDatasetStore } from '../../src/renderer/stores/dataset.store';
import { useModelStore } from '../../src/renderer/stores/model.store';

afterEach(cleanup);

const mockRemoteTestConnection = vi.fn();
const mockRemoteDisconnect = vi.fn();
const mockTrainingStartLocal = vi.fn();

function setElectronApi() {
  (window as unknown as Record<string, unknown>).electronAPI = {
    remoteTestConnection: mockRemoteTestConnection,
    remoteDisconnect: mockRemoteDisconnect,
    trainingStartLocal: mockTrainingStartLocal,
    trainingStop: vi.fn(async () => ({ success: true, data: undefined })),
    onTrainingProgress: vi.fn(() => () => {}),
    onTrainingResult: vi.fn(() => () => {}),
    systemCudaStatus: vi.fn(async () => ({ success: true, data: { cudaAvailable: false } })),
    datasetList: vi.fn(async () => ({ success: true, data: [] })),
    modelList: vi.fn(async () => ({ success: true, data: [] })),
  };
}

describe('远程连接 UI', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setElectronApi();
    useTrainingStore.setState({
      mode: 'remote',
      config: {
        dataset: 'FD002',
        baseModel: 'multiscale',
        outputName: 'run1',
        hyperParams: { epochs: 10, batchSize: 32, learningRate: 0.001, optimizer: 'adam' },
      },
      progress: null,
      isRunning: false,
    });
    useDatasetStore.setState({ datasets: [], searchQuery: '', selectedDataset: null });
    useModelStore.setState({
      models: [],
      selectedModel: null,
      searchQuery: '',
      sourceFilter: 'all',
    });
  });

  async function renderTraining() {
    const { default: Training } = await import('../../src/renderer/pages/Training');
    return render(React.createElement(Training));
  }

  it('test_training_page_has_config_server_button', async () => {
    const { findByText } = await renderTraining();
    expect(await findByText('配置服务器')).toBeDefined();
  });

  it('test_training_page_has_server_status', async () => {
    const { findAllByText } = await renderTraining();
    // 状态条 + 系统状态面板都可能有"未连接"
    expect((await findAllByText(/未连接/)).length).toBeGreaterThan(0);
  });

  it('test_config_button_opens_modal', async () => {
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    // 弹窗打开，显示连接配置标题
    expect(await screen.findByText('服务器连接')).toBeDefined();
  });

  it('test_modal_has_password_and_ssh_toggle', async () => {
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    // 认证切换按钮在弹窗内
    const pwBtns = await screen.findAllByText('密码');
    expect(pwBtns.length).toBeGreaterThan(0);
    expect(screen.getByText('SSH密钥')).toBeDefined();
  });

  it('test_password_mode_shows_password_field', async () => {
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    // 密码方式应显示密码输入框
    const pwInputs = document.querySelectorAll('input[type="password"]');
    expect(pwInputs.length).toBeGreaterThan(0);
  });

  it('test_ssh_mode_shows_private_key_field', async () => {
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    // 切到 SSH密钥
    fireEvent.click(await screen.findByText('SSH密钥'));
    expect(await screen.findByPlaceholderText(/id_rsa/)).toBeDefined();
  });

  it('test_test_connection_calls_ipc', async () => {
    mockRemoteTestConnection.mockResolvedValue({
      success: true,
      data: { connected: true, message: 'ok' },
    });
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    // 填必填字段
    fireEvent.click(await screen.findByText('测试连接'));
    await waitFor(() => expect(mockRemoteTestConnection).toHaveBeenCalled());
  });

  it('test_connected_status_shows_connected', async () => {
    mockRemoteTestConnection.mockResolvedValue({
      success: true,
      data: {
        connected: true,
        message: '已连接',
        serverInfo: { host: '1.2.3.4', port: 22, username: 'u' },
      },
    });
    const { findByText } = await renderTraining();
    fireEvent.click(await findByText('配置服务器'));
    fireEvent.click(await screen.findByText('测试连接'));
    // 弹窗内状态提示显示已连接
    await waitFor(() => {
      expect(screen.getAllByText(/已连接/).length).toBeGreaterThan(0);
    });
  });
});
