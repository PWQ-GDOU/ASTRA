export interface InferenceRequest {
  datasetId: string;
  modelId: string;
  batchSize: number;
  device: 'cpu' | 'cuda' | `cuda:${number}`;
  applyRulCap: boolean;
  rulCap: number;
}

export interface InferenceProgressEvent {
  taskId: string;
  progress: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  elapsed?: number;
  message?: string;
}

export interface InferenceResult {
  taskId: string;
  units: number[];
  predictions: number[];
  targets?: number[];
  rmse?: number;
  mae?: number;
  score?: number;
  seconds: number;
}
