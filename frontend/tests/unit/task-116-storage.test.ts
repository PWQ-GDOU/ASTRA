/**
 * Task 116 测试：数据持久化层 storage
 *
 * 验证 src/main/lib/storage.ts 的 JSON 读写。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

vi.mock('electron', () => ({
  app: { getPath: vi.fn(() => '/mock/userdata') },
}));

describe('Task 116: storage 持久化层', () => {
  let tmp: string;

  beforeEach(() => {
    tmp = mkdtempSync(join(tmpdir(), 'storage-'));
  });

  afterEach(() => {
    rmSync(tmp, { recursive: true, force: true });
  });

  it('test_storage_exports_functions', async () => {
    const mod = await import('../../src/main/lib/storage');
    expect(typeof mod.readJSON).toBe('function');
    expect(typeof mod.writeJSON).toBe('function');
    expect(typeof mod.appendJSON).toBe('function');
    expect(typeof mod.deleteJSON).toBe('function');
  });

  it('test_storage_file_constants', async () => {
    const mod = await import('../../src/main/lib/storage');
    expect(mod.DATASETS_FILE).toBe('datasets.json');
    expect(mod.MODELS_FILE).toBe('models.json');
    expect(mod.TRAINING_CONFIG_FILE).toBe('training-configs.json');
  });

  it('test_storage_read_missing_returns_null', async () => {
    const mod = await import('../../src/main/lib/storage');
    // 用不存在的文件
    expect(mod.readJSON('nonexistent.json')).toBeNull();
  });

  it('test_storage_write_read_roundtrip', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    const items = [{ id: 'a', name: 'x' }];
    mod.writeJSON('test.json', items);
    const read = mod.readJSON<{ id: string; name: string }>('test.json');
    expect(read).toEqual(items);
  });

  it('test_storage_append_adds_item', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    mod.appendJSON('test.json', { id: '1', name: 'a' });
    mod.appendJSON('test.json', { id: '2', name: 'b' });
    const read = mod.readJSON<{ id: string; name: string }>('test.json');
    expect(read).toHaveLength(2);
  });

  it('test_storage_update_merges_item', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    mod.appendJSON('test.json', { id: '1', name: 'a', type: 'old' });
    mod.updateJSON<{ id: string; name: string; type: string }>('test.json', '1', (item) => ({
      ...item,
      type: 'new',
    }));
    const read = mod.readJSON<{ id: string; type: string }>('test.json');
    expect(read?.[0].type).toBe('new');
  });

  it('test_storage_delete_removes_item', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    mod.appendJSON('test.json', { id: '1', name: 'a' });
    mod.appendJSON('test.json', { id: '2', name: 'b' });
    mod.deleteJSON('test.json', '1');
    const read = mod.readJSON<{ id: string }>('test.json');
    expect(read).toHaveLength(1);
    expect(read?.[0].id).toBe('2');
  });

  it('test_storage_update_missing_id_noop', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    mod.appendJSON('test.json', { id: '1', name: 'a' });
    mod.updateJSON('test.json', '999', (item) => ({ ...item, name: 'changed' }));
    const read = mod.readJSON<{ name: string }>('test.json');
    expect(read?.[0].name).toBe('a');
  });

  it('test_storage_delete_missing_id_noop', async () => {
    vi.doMock('electron', () => ({
      app: { getPath: vi.fn(() => tmp) },
    }));
    vi.resetModules();
    const mod = await import('../../src/main/lib/storage');
    mod.appendJSON('test.json', { id: '1', name: 'a' });
    mod.deleteJSON('test.json', '999');
    const read = mod.readJSON<{ id: string }>('test.json');
    expect(read).toHaveLength(1);
  });
});
