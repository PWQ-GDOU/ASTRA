/**
 * Task 2 测试：Electron 主进程入口 + 窗口配置
 *
 * 测试 Electron 主进程创建 BrowserWindow 的配置参数。
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Mock modules before any imports
const mockBrowserWindow = vi.fn();
const mockLoadURL = vi.fn();
const mockLoadFile = vi.fn();
const mockMinimize = vi.fn();
const mockMaximize = vi.fn();
const mockUnmaximize = vi.fn();
const mockClose = vi.fn();
const mockQuit = vi.fn();
const mockIsMaximized = vi.fn(() => false);
const mockSetWindowOpenHandler = vi.fn();
const mockOpenDevTools = vi.fn();
const mockSetMenu = vi.fn();
const mockWhenReady = vi.fn();
const mockOn = vi.fn();
const mockGetPath = vi.fn(() => '/mock/userdata');
const mockGetAllWindows = vi.fn(() => [{ id: 1 }]);
const mockJoin = vi.fn((...args: string[]) => args.join('/'));
const mockOpenExternal = vi.fn();

let mainWindowInstance: {
  loadURL: ReturnType<typeof vi.fn>;
  loadFile: ReturnType<typeof vi.fn>;
  minimize: ReturnType<typeof vi.fn>;
  maximize: ReturnType<typeof vi.fn>;
  unmaximize: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  isMaximized: ReturnType<typeof vi.fn>;
  on: ReturnType<typeof vi.fn>;
  isDestroyed: ReturnType<typeof vi.fn>;
  webContents: {
    setWindowOpenHandler: ReturnType<typeof vi.fn>;
    openDevTools: ReturnType<typeof vi.fn>;
  };
} | null = null;

vi.mock('electron', () => {
  const BWMock = vi.fn().mockImplementation((opts: Record<string, unknown>) => {
    mainWindowInstance = {
      loadURL: mockLoadURL,
      loadFile: mockLoadFile,
      minimize: mockMinimize,
      maximize: mockMaximize,
      unmaximize: mockUnmaximize,
      close: mockClose,
      isMaximized: mockIsMaximized,
      on: mockOn,
      isDestroyed: vi.fn(() => false),
      webContents: {
        setWindowOpenHandler: mockSetWindowOpenHandler,
        openDevTools: mockOpenDevTools,
      },
    };
    mockBrowserWindow(opts);
    return mainWindowInstance;
  });
  BWMock.getAllWindows = () => mockGetAllWindows();
  return {
    app: {
      whenReady: () => mockWhenReady(),
      on: mockOn,
      quit: mockQuit,
      getPath: mockGetPath,
      isPackaged: false,
    },
    BrowserWindow: BWMock,
    Menu: {
      setApplicationMenu: mockSetMenu,
    },
    shell: {
      openExternal: mockOpenExternal,
    },
    ipcMain: {
      handle: vi.fn(),
    },
  };
});

vi.mock('path', async (importOriginal) => {
  const actual = await importOriginal<typeof import('path')>();
  return {
    ...actual,
    default: actual,
    join: (...args: string[]) => mockJoin(...args),
    resolve: (...args: string[]) => mockJoin(...args),
  };
});

describe('Task 2: Electron 主进程入口 + 窗口配置', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockWhenReady.mockResolvedValue(undefined);
    mainWindowInstance = null;
    // Reset process.env.NODE_ENV to development by default
    process.env.NODE_ENV = 'development';
  });

  afterEach(() => {
    process.env.NODE_ENV = 'development';
  });

  it('test_main_window_created_on_ready', async () => {
    // Dynamic import to trigger the module code
    await import('../../src/main/index');
    // Simulate the whenReady callback explicitly
    expect(mockWhenReady).toHaveBeenCalled();
  });

  it('test_window_width_1440', async () => {
    // Clear module cache to get fresh import
    vi.resetModules();
    await import('../../src/main/index');
    // verify BrowserWindow was called with width: 1440
    const calls = mockBrowserWindow.mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    expect(opts?.width).toBe(1440);
  });

  it('test_window_height_1000', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    expect(opts?.height).toBe(1000);
  });

  it('test_window_frame_false', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    expect(opts?.frame).toBe(false);
  });

  it('test_background_color_is_white', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    expect(calls.length).toBeGreaterThan(0);
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    expect(opts?.backgroundColor).toBe('#ffffff');
  });

  it('test_context_isolation_enabled', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    const webPrefs = opts?.webPreferences as Record<string, unknown> | undefined;
    expect(webPrefs?.contextIsolation).toBe(true);
  });

  it('test_node_integration_disabled', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    const webPrefs = opts?.webPreferences as Record<string, unknown> | undefined;
    expect(webPrefs?.nodeIntegration).toBe(false);
  });

  it('test_preload_path_set', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const calls = mockBrowserWindow.mock.calls;
    const opts = calls[0]?.[0] as Record<string, unknown> | undefined;
    const webPrefs = opts?.webPreferences as Record<string, unknown> | undefined;
    expect(webPrefs?.preload).toBeDefined();
    expect(typeof webPrefs?.preload).toBe('string');
    expect((webPrefs?.preload as string).length).toBeGreaterThan(0);
  });

  it('test_menu_set_null_in_production', async () => {
    process.env.NODE_ENV = 'production';
    vi.resetModules();
    await import('../../src/main/index');
    // Verify Menu.setApplicationMenu was called with null
    expect(mockSetMenu).toHaveBeenCalledWith(null);
  });

  it('test_window_all_closed_quits_app', async () => {
    vi.resetModules();
    await import('../../src/main/index');
    const allClosedCall = mockOn.mock.calls.find(
      ([event]: [string, ...unknown[]]) => event === 'window-all-closed',
    );
    expect(allClosedCall).toBeDefined();
    // Call the registered handler
    const handler = allClosedCall![1] as () => void;
    handler();
    expect(mockQuit).toHaveBeenCalled();
  });
});
