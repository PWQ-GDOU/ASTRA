import { app, BrowserWindow, Menu, shell } from 'electron';
import { join } from 'path';
import { registerAllHandlers } from './ipc/register-all';
import { setMainWindow } from './ipc/system.ipc';
import { refreshCudaStatus } from './ipc/system.ipc';

let mainWindow: BrowserWindow | null = null;

const isDev = process.env.NODE_ENV === 'development';

function createWindow(): void {
  const preloadPath = join(__dirname, '..', '..', 'dist-electron', 'preload', 'index.js');

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    frame: false,
    titleBarStyle: 'hidden',
    backgroundColor: '#ffffff',
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  if (mainWindow) {
    setMainWindow(mainWindow);
    registerAllHandlers(mainWindow);
  }

  // Remove default menu in production
  if (!isDev) {
    Menu.setApplicationMenu(null);
  }

  // Eager GPU detection on startup (non-blocking, fire-and-forget)
  refreshCudaStatus();

  if (isDev && process.env['VITE_DEV_SERVER_URL']) {
    mainWindow.loadURL(process.env['VITE_DEV_SERVER_URL']);
    mainWindow.webContents.openDevTools();
  } else {
    const appPath = app.isPackaged
      ? join(__dirname, '..', '..', 'dist', 'index.html')
      : join(__dirname, '..', '..', 'dist', 'index.html');
    mainWindow.loadFile(appPath);
  }

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
