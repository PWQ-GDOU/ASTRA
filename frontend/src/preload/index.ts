import { contextBridge, ipcRenderer } from 'electron';
import { IPC_CHANNELS } from '../shared/types/ipc';

const electronAPI = {
  // Window controls
  minimizeWindow: () => ipcRenderer.invoke(IPC_CHANNELS.WINDOW_MINIMIZE),
  maximizeWindow: () => ipcRenderer.invoke(IPC_CHANNELS.WINDOW_MAXIMIZE),
  closeWindow: () => ipcRenderer.invoke(IPC_CHANNELS.WINDOW_CLOSE),

  // Platform
  getPlatform: () => process.platform,
  getElectronVersion: () => process.versions.electron || 'unknown',

  // Dataset
  datasetList: () => ipcRenderer.invoke(IPC_CHANNELS.DATASET_LIST),
  datasetImport: () => ipcRenderer.invoke(IPC_CHANNELS.DATASET_IMPORT),
  datasetSaveMapping: (mapping: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.DATASET_SAVE_MAPPING, mapping),
  datasetUpdateType: (id: string, type: string) =>
    ipcRenderer.invoke(IPC_CHANNELS.DATASET_UPDATE_TYPE, id, type),
  datasetRename: (id: string, name: string) =>
    ipcRenderer.invoke(IPC_CHANNELS.DATASET_RENAME, id, name),
  datasetDelete: (id: string) => ipcRenderer.invoke(IPC_CHANNELS.DATASET_DELETE, id),

  // Inference
  inferenceRun: (req: unknown) => ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_RUN, req),
  inferenceRunBatch: (reqs: unknown[]) =>
    ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_RUN_BATCH, reqs),
  inferenceCancel: (taskId: string) => ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_CANCEL, taskId),
  onInferenceProgress: (cb: (evt: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, evt: unknown) => cb(evt);
    ipcRenderer.on(IPC_CHANNELS.INFERENCE_PROGRESS, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.INFERENCE_PROGRESS, listener);
    };
  },
  onInferenceResult: (cb: (result: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, result: unknown) => cb(result);
    ipcRenderer.on(IPC_CHANNELS.INFERENCE_RESULT, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.INFERENCE_RESULT, listener);
    };
  },

  // Inference History
  inferenceHistoryList: (filters?: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_HISTORY_LIST, filters),
  inferenceHistoryDetail: (id: string) =>
    ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_HISTORY_DETAIL, id),
  inferenceHistoryDelete: (id: string) =>
    ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_HISTORY_DELETE, id),
  inferenceHistoryExport: (ids: string[], format: string) =>
    ipcRenderer.invoke(IPC_CHANNELS.INFERENCE_HISTORY_EXPORT, ids, format),

  // Training
  trainingStartLocal: (config: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.TRAINING_START_LOCAL, config),
  trainingStartRemote: (config: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.TRAINING_START_REMOTE, config),
  trainingStop: (runId: string) => ipcRenderer.invoke(IPC_CHANNELS.TRAINING_STOP, runId),
  onTrainingProgress: (cb: (evt: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, evt: unknown) => cb(evt);
    ipcRenderer.on(IPC_CHANNELS.TRAINING_PROGRESS, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.TRAINING_PROGRESS, listener);
    };
  },
  onTrainingResult: (cb: (result: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, result: unknown) => cb(result);
    ipcRenderer.on(IPC_CHANNELS.TRAINING_RESULT, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.TRAINING_RESULT, listener);
    };
  },
  trainingConfigSave: (config: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.TRAINING_CONFIG_SAVE, config),
  trainingConfigLoad: () => ipcRenderer.invoke(IPC_CHANNELS.TRAINING_CONFIG_LOAD),
  trainingLogPathSelect: () => ipcRenderer.invoke(IPC_CHANNELS.TRAINING_LOG_PATH_SELECT),

  // Model
  modelList: () => ipcRenderer.invoke(IPC_CHANNELS.MODEL_LIST),
  modelImport: () => ipcRenderer.invoke(IPC_CHANNELS.MODEL_IMPORT),
  modelDelete: (id: string) => ipcRenderer.invoke(IPC_CHANNELS.MODEL_DELETE, id),
  modelSetDefault: (id: string) => ipcRenderer.invoke(IPC_CHANNELS.MODEL_SET_DEFAULT, id),
  onModelStatus: (cb: (status: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, status: unknown) => cb(status);
    ipcRenderer.on(IPC_CHANNELS.MODEL_STATUS, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.MODEL_STATUS, listener);
    };
  },

  // System
  systemCudaStatus: () => ipcRenderer.invoke(IPC_CHANNELS.SYSTEM_CUDA_STATUS),
  onCudaStatus: (cb: (status: unknown) => void) => {
    const listener = (_e: Electron.IpcRendererEvent, status: unknown) => cb(status);
    ipcRenderer.on(IPC_CHANNELS.SYSTEM_CUDA_STATUS, listener);
    return () => {
      ipcRenderer.removeListener(IPC_CHANNELS.SYSTEM_CUDA_STATUS, listener);
    };
  },

  // Statistics
  statisticsQuery: (query?: unknown) => ipcRenderer.invoke(IPC_CHANNELS.STATISTICS_QUERY, query),

  // Remote (SSH)
  remoteTestConnection: (config: unknown) =>
    ipcRenderer.invoke(IPC_CHANNELS.REMOTE_TEST_CONNECTION, config),
  remoteDisconnect: () => ipcRenderer.invoke(IPC_CHANNELS.REMOTE_DISCONNECT),
};

contextBridge.exposeInMainWorld('electronAPI', electronAPI);

export type ElectronAPI = typeof electronAPI;
