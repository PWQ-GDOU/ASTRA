export const IPC_CHANNELS = {
  // Window
  WINDOW_MINIMIZE: 'window:minimize',
  WINDOW_MAXIMIZE: 'window:maximize',
  WINDOW_CLOSE: 'window:close',

  // Dataset
  DATASET_LIST: 'dataset:list',
  DATASET_IMPORT: 'dataset:import',
  DATASET_SAVE_MAPPING: 'dataset:save-mapping',
  DATASET_UPDATE_TYPE: 'dataset:update-type',
  DATASET_RENAME: 'dataset:rename',
  DATASET_DELETE: 'dataset:delete',

  // Inference
  INFERENCE_RUN: 'inference:run',
  INFERENCE_RUN_BATCH: 'inference:run-batch',
  INFERENCE_CANCEL: 'inference:cancel',
  INFERENCE_PROGRESS: 'inference:progress',
  INFERENCE_RESULT: 'inference:result',

  // Inference History
  INFERENCE_HISTORY_LIST: 'inference:history-list',
  INFERENCE_HISTORY_DETAIL: 'inference:history-detail',
  INFERENCE_HISTORY_DELETE: 'inference:history-delete',
  INFERENCE_HISTORY_EXPORT: 'inference:history-export',

  // Training
  TRAINING_START_LOCAL: 'training:start-local',
  TRAINING_START_REMOTE: 'training:start-remote',
  TRAINING_STOP: 'training:stop',
  TRAINING_PROGRESS: 'training:progress',
  TRAINING_RESULT: 'training:result',
  TRAINING_CONFIG_SAVE: 'training:config-save',
  TRAINING_CONFIG_LOAD: 'training:config-load',
  TRAINING_LOG_PATH_SELECT: 'training:log-path-select',

  // Model
  MODEL_LIST: 'model:list',
  MODEL_IMPORT: 'model:import',
  MODEL_DELETE: 'model:delete',
  MODEL_SET_DEFAULT: 'model:set-default',
  MODEL_STATUS: 'model:status',

  // System
  SYSTEM_CUDA_STATUS: 'system:cuda-status',

  // Remote (SSH)
  REMOTE_TEST_CONNECTION: 'remote:test-connection',
  REMOTE_DISCONNECT: 'remote:disconnect',

  // Statistics
  STATISTICS_QUERY: 'statistics:query',
} as const;

export type IpcChannelName = (typeof IPC_CHANNELS)[keyof typeof IPC_CHANNELS];
