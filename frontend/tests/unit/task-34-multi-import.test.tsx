import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('数据集文件夹导入', () => {
  const mockHandle = vi.fn();
  const mockShowOpenDialog = vi.fn();

  beforeEach(() => { vi.clearAllMocks(); });

  async function loadIpc() {
    vi.resetModules();
    vi.doMock('electron', () => ({ ipcMain: { handle: (...args: unknown[]) => mockHandle(...args) }, dialog: { showOpenDialog: (...args: unknown[]) => mockShowOpenDialog(...args) } }));
    vi.doMock('../../src/main/lib/storage', () => ({ readJSON: vi.fn(() => []), appendJSON: vi.fn(), updateJSON: vi.fn(), deleteJSON: vi.fn(), DATASETS_FILE: 'datasets.json' }));
    const mod = await import('../../src/main/ipc/dataset.ipc');
    mod.registerDatasetHandlers();
  }

  it('opens a directory once instead of selecting individual files', async () => {
    await loadIpc();
    mockShowOpenDialog.mockResolvedValue({ canceled: true, filePaths: [] });
    const handler = mockHandle.mock.calls.find((call) => call[0] === 'dataset:import')?.[1];
    await handler();
    expect(mockShowOpenDialog.mock.calls[0][0].properties).toEqual(['openDirectory']);
  });

  it('registers a separate mapping persistence handler', async () => {
    await loadIpc();
    expect(mockHandle.mock.calls.some((call) => call[0] === 'dataset:save-mapping')).toBe(true);
  });
});
