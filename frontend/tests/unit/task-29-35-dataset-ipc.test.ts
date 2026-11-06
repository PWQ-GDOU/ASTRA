/**
 * Task 29-30, 34-35 测试：数据集 IPC handlers
 *
 * 验证 dataset.ipc.ts 的 list/import/update-type/delete handlers
 * 通过 storage 层读写 JSON。
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

// Mock electron + storage
const mockHandle = vi.fn();
const mockShowOpenDialog = vi.fn();
const mockReadJSON = vi.fn();
const mockAppendJSON = vi.fn();
const mockUpdateJSON = vi.fn();
const mockDeleteJSON = vi.fn();

vi.mock('electron', () => ({
  ipcMain: { handle: (...args: unknown[]) => mockHandle(...args) },
  dialog: { showOpenDialog: (...args: unknown[]) => mockShowOpenDialog(...args) },
}));

vi.mock('../../src/main/lib/storage', () => ({
  readJSON: (...args: unknown[]) => mockReadJSON(...args),
  appendJSON: (...args: unknown[]) => mockAppendJSON(...args),
  updateJSON: (...args: unknown[]) => mockUpdateJSON(...args),
  deleteJSON: (...args: unknown[]) => mockDeleteJSON(...args),
  DATASETS_FILE: 'datasets.json',
}));

describe('Task 29-30, 34-35: 数据集 IPC handlers', () => {
  let register: () => void;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  async function load() {
    const mod = await import('../../src/main/ipc/dataset.ipc');
    register = mod.registerDatasetHandlers;
    register();
  }

  function findHandler(channel: string) {
    return mockHandle.mock.calls.find((c: unknown[]) => c[0] === channel)?.[1];
  }

  it('test_dataset_list_registers_handler', async () => {
    await load();
    expect(findHandler('dataset:list')).toBeDefined();
  });

  it('test_dataset_list_returns_data', async () => {
    mockReadJSON.mockReturnValue([
      { id: 'd1', name: 'A', samples: 1, format: 'csv', type: 'train', createdAt: 't' },
    ]);
    await load();
    const result = findHandler('dataset:list')();
    expect(result).toMatchObject({ success: true });
    expect(result.data).toHaveLength(1);
  });

  it('test_dataset_list_empty_returns_array', async () => {
    mockReadJSON.mockReturnValue(null);
    await load();
    const result = findHandler('dataset:list')();
    expect(result).toMatchObject({ success: true });
    expect(Array.isArray(result.data)).toBe(true);
    expect(result.data).toHaveLength(0);
  });

  it('test_dataset_import_registers_handler', async () => {
    await load();
    expect(findHandler('dataset:import')).toBeDefined();
  });

  it('test_dataset_import_cancelled_returns_error', async () => {
    mockShowOpenDialog.mockResolvedValue({ canceled: true, filePaths: [] });
    await load();
    const result = await findHandler('dataset:import')();
    expect(result.success).toBe(false);
  });

  it('test_dataset_import_returns_folder_candidates', async () => {
    const folder = mkdtempSync(join(tmpdir(), 'dataset-folder-'));
    ['train.any', 'test.any', 'rul.any'].forEach((name) => writeFileSync(join(folder, name), '1'));
    mockShowOpenDialog.mockResolvedValue({
      canceled: false,
      filePaths: [folder],
    });
    await load();
    const result = await findHandler('dataset:import')();
    expect(result.success).toBe(true);
    expect(result.data.rootPath).toBe(folder);
    expect(result.data.files).toHaveLength(3);
    expect(mockAppendJSON).not.toHaveBeenCalled();
    rmSync(folder, { recursive: true, force: true });
  });

  it('rejects mapped files from a sibling folder with the same path prefix', async () => {
    const root = mkdtempSync(join(tmpdir(), 'dataset-root-'));
    const sibling = `${root}-escape`;
    const { mkdirSync } = await import('fs');
    mkdirSync(sibling);
    const files = ['train.txt', 'test.txt', 'rul.txt'].map((name) => {
      const path = join(sibling, name);
      writeFileSync(path, '1');
      return path;
    });
    await load();

    const result = findHandler('dataset:save-mapping')(
      {},
      {
        name: 'escaped',
        rootPath: root,
        trainFile: files[0],
        testFile: files[1],
        rulFile: files[2],
      },
    );

    expect(result.success).toBe(false);
    expect(mockAppendJSON).not.toHaveBeenCalled();
    rmSync(root, { recursive: true, force: true });
    rmSync(sibling, { recursive: true, force: true });
  });

  it('test_dataset_update_type_registers_handler', async () => {
    await load();
    expect(findHandler('dataset:update-type')).toBeDefined();
  });

  it('test_dataset_update_type_calls_update', async () => {
    await load();
    const result = findHandler('dataset:update-type')({}, 'd1', 'test');
    expect(mockUpdateJSON).toHaveBeenCalledWith('datasets.json', 'd1', expect.any(Function));
    expect(result.success).toBe(true);
  });

  it('persists dataset renames', async () => {
    await load();
    const handler = findHandler('dataset:rename');
    expect(handler).toBeDefined();
    const result = handler({}, 'd1', 'FD002 custom');
    expect(mockUpdateJSON).toHaveBeenCalledWith('datasets.json', 'd1', expect.any(Function));
    const updater = mockUpdateJSON.mock.calls[0][2];
    expect(updater({ id: 'd1', name: 'old' }).name).toBe('FD002 custom');
    expect(result.success).toBe(true);
  });

  it('test_dataset_delete_registers_handler', async () => {
    await load();
    expect(findHandler('dataset:delete')).toBeDefined();
  });

  it('test_dataset_delete_calls_delete', async () => {
    await load();
    const result = findHandler('dataset:delete')({}, 'd1');
    expect(mockDeleteJSON).toHaveBeenCalledWith('datasets.json', 'd1');
    expect(result.success).toBe(true);
  });
});
