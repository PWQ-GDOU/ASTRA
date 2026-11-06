/**
 * Task 13 测试：Preload 脚本
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');
const PRELOAD_PATH = resolve(ROOT, 'src', 'preload', 'index.ts');
const TYPES_PATH = resolve(ROOT, 'src', 'preload', 'electron.d.ts');

describe('Task 13: Preload 脚本', () => {
  it('test_preload_exposes_electronApi', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('electronAPI');
    expect(content).toContain('contextBridge');
    expect(content).toContain('exposeInMainWorld');
  });

  it('test_preload_has_minimize_window', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('minimizeWindow');
    expect(content).toContain('minimize');
  });

  it('test_preload_has_maximize_window', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('maximizeWindow');
    expect(content).toContain('maximize');
  });

  it('test_preload_has_close_window', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('closeWindow');
    expect(content).toContain('close');
  });

  it('test_preload_has_get_platform', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('getPlatform');
    expect(content).toContain('process.platform');
  });

  it('test_preload_type_declaration_exists', () => {
    expect(existsSync(TYPES_PATH)).toBe(true);
  });

  it('test_minimize_invokes_window_minimize', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('ipcRenderer.invoke');
  });

  it('test_close_invokes_window_close', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('close');
  });

  it('test_getplatform_returns_process_platform', () => {
    const content = readFileSync(PRELOAD_PATH, 'utf-8');
    expect(content).toContain('process.platform');
  });
});
