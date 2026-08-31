export interface Model {
  id: string;
  name: string;
  status: 'available' | 'training' | 'error';
  source: 'imported' | 'trained' | 'pretrained';
  isDefault: boolean;
  path: string;
  size?: string;
  createdAt: string;
  performance?: ModelPerformance;
}

export interface ModelPerformance {
  taskId: string;
  datasetId: string;
  datasetName: string;
  rmse?: number;
  mae?: number;
  score?: number;
  sampleCount: number;
  seconds: number;
  evaluatedAt: string;
}

export interface ModelMetrics {
  accuracy: number;
  paramCount: string;
  framework: string;
  inputShape: string;
}

export type ModelStatus = 'available' | 'training' | 'error';
export type ModelSource = 'imported' | 'trained' | 'pretrained';
