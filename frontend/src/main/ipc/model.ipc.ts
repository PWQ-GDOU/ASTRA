import { ipcMain, dialog } from 'electron';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type { Model } from '../../shared/types/model';
import { readJSON, writeJSON, appendJSON, deleteJSON, MODELS_FILE } from '../lib/storage';

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function registerModelHandlers(): void {
  ipcMain.handle(IPC_CHANNELS.MODEL_LIST, (): IPCResult<Model[]> => {
    try {
      const models = readJSON<Model>(MODELS_FILE) || [];
      return { success: true, data: models };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.MODEL_IMPORT, async (): Promise<IPCResult<Model>> => {
    try {
      const result = await dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [{ name: 'ASTRA Checkpoints', extensions: ['pt'] }],
      });
      if (result.canceled || result.filePaths.length === 0) {
        return { success: false, error: 'File selection cancelled' };
      }
      const folderPath = result.filePaths[0];
      const folderName = folderPath.split(/[/\\]/).pop() || 'checkpoint.pt';
      const existing = readJSON<Model>(MODELS_FILE) || [];
      const model: Model = {
        id: genId(),
        name: folderName,
        status: 'available',
        source: 'imported',
        isDefault: existing.length === 0,
        path: folderPath,
        createdAt: new Date().toISOString(),
      };
      appendJSON(MODELS_FILE, model);
      return { success: true, data: model };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.MODEL_DELETE, (_e, id: string): IPCResult<void> => {
    try {
      deleteJSON(MODELS_FILE, id);
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.MODEL_SET_DEFAULT, (_e, id: string): IPCResult<void> => {
    try {
      const models = readJSON<Model>(MODELS_FILE) || [];
      const updated = models.map((m) => ({ ...m, isDefault: m.id === id }));
      writeJSON(MODELS_FILE, updated);
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });
}
