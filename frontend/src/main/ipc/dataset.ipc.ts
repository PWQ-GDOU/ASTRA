import { ipcMain, dialog } from 'electron';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type { Dataset } from '../../shared/types/dataset';
import type { DatasetFolderCandidate, DatasetMapping } from '../../shared/types/dataset';
import { existsSync, readdirSync, statSync } from 'fs';
import { isAbsolute, join, relative, resolve } from 'path';
import { readJSON, appendJSON, updateJSON, deleteJSON, DATASETS_FILE } from '../lib/storage';

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function registerDatasetHandlers(): void {
  ipcMain.handle(IPC_CHANNELS.DATASET_LIST, (): IPCResult<Dataset[]> => {
    try {
      const data = readJSON<Dataset>(DATASETS_FILE) || [];
      return { success: true, data };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(
    IPC_CHANNELS.DATASET_IMPORT,
    async (): Promise<IPCResult<DatasetFolderCandidate>> => {
      try {
        const result = await dialog.showOpenDialog({
          properties: ['openDirectory'],
        });
        if (result.canceled || result.filePaths.length === 0) {
          return { success: false, error: 'File selection cancelled' };
        }
        const rootPath = resolve(result.filePaths[0]);
        const files = readdirSync(rootPath)
          .map((name) => join(rootPath, name))
          .filter((path) => statSync(path).isFile());
        return { success: true, data: { rootPath, files } };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(
    IPC_CHANNELS.DATASET_SAVE_MAPPING,
    (_e, mapping: DatasetMapping): IPCResult<Dataset> => {
      try {
        const paths = [mapping.trainFile, mapping.testFile, mapping.rulFile].map((path) =>
          resolve(path),
        );
        if (!mapping.name.trim() || new Set(paths).size !== 3 || paths.some((p) => !existsSync(p)))
          throw new Error('请完整映射三个不同且存在的文件');
        const rootPath = resolve(mapping.rootPath);
        if (
          paths.some((path) => {
            const fromRoot = relative(rootPath, path);
            return fromRoot.startsWith('..') || isAbsolute(fromRoot);
          })
        )
          throw new Error('映射文件必须位于所选文件夹中');
        const ds: Dataset = {
          id: genId(),
          name: mapping.name.trim(),
          samples: 0,
          format: 'cmapss',
          type: 'unlabeled',
          createdAt: new Date().toISOString(),
          files: paths,
          rootPath,
          trainFile: paths[0],
          testFile: paths[1],
          rulFile: paths[2],
          astraCompatible: true,
        };
        appendJSON(DATASETS_FILE, ds);
        return { success: true, data: ds };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(
    IPC_CHANNELS.DATASET_UPDATE_TYPE,
    (_e, id: string, type: string): IPCResult<void> => {
      try {
        updateJSON<Dataset>(DATASETS_FILE, id, (ds) => ({ ...ds, type: type as Dataset['type'] }));
        return { success: true, data: undefined };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(IPC_CHANNELS.DATASET_RENAME, (_e, id: string, name: string): IPCResult<void> => {
    try {
      const normalized = name.trim();
      if (!normalized) throw new Error('数据集名称不能为空');
      updateJSON<Dataset>(DATASETS_FILE, id, (ds) => ({ ...ds, name: normalized }));
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.DATASET_DELETE, (_e, id: string): IPCResult<void> => {
    try {
      deleteJSON(DATASETS_FILE, id);
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });
}
