import { create } from 'zustand';
import { useDatasetStore } from './dataset.store';
import { useModelStore } from './model.store';
import { useInferenceStore } from './inference.store';
import { useInferenceHistoryStore } from './inference-history.store';
import { useTrainingStore } from './training.store';

// Core types for dashboard
export interface StatData {
  label: string;
  value: string | number | null;
  change?: number;
  icon: string;
}

export interface CudaStatus {
  cudaAvailable: boolean;
  gpuName?: string;
  gpuMemory?: string;
  utilization?: number;
}

export interface DashboardStats {
  datasetCount: number;
  modelCount: number;
  inferenceCount: number;
  systemStatus: string;
  gpuInfo: CudaStatus | null;
}

export interface RecentTask {
  id: string;
  name: string;
  model: string;
  dataset: string;
  status: 'completed' | 'running' | 'failed' | 'pending';
  time: string;
}

export interface TrainingTask {
  name: string;
  progress: number;
}

export interface AccuracyPoint {
  label: string;
  value: number;
}

export interface DashboardState {
  stats: DashboardStats;
  recentTasks: RecentTask[];
  trainingProgress: TrainingTask[];
  accuracyTrend: AccuracyPoint[];
  lastUpdated: string | null;
  refreshStats: () => void;
  refreshRecentTasks: () => Promise<void>;
  refreshTrainingProgress: () => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  stats: {
    datasetCount: 0,
    modelCount: 0,
    inferenceCount: 0,
    systemStatus: 'offline',
    gpuInfo: null,
  },
  recentTasks: [],
  trainingProgress: [],
  accuracyTrend: [],
  lastUpdated: null,

  refreshStats: () => {
    // 跨 store 汇总：datasetStore / modelStore / inferenceStore / trainingStore
    // 每个 store 单独 try/catch，某个失败不影响其他字段
    let datasetCount = 0;
    let modelCount = 0;
    let inferenceCount = 0;
    let systemStatus = 'offline';
    let gpuInfo: CudaStatus | null = null;

    try {
      datasetCount = useDatasetStore.getState().datasets.length;
    } catch {
      datasetCount = 0;
    }

    try {
      modelCount = useModelStore.getState().models.length;
    } catch {
      modelCount = 0;
    }

    try {
      const s = useInferenceStore.getState();
      inferenceCount = s.queue.length + (s.result ? 1 : 0);
    } catch {
      inferenceCount = 0;
    }

    try {
      const s = useTrainingStore.getState();
      systemStatus = s.systemStatus.cuda.available ? 'online' : 'offline';
      gpuInfo = s.systemStatus.cuda.available
        ? {
            cudaAvailable: true,
            gpuName: s.systemStatus.cuda.gpuName,
            gpuMemory: s.systemStatus.cuda.memory,
          }
        : null;
    } catch {
      systemStatus = 'offline';
      gpuInfo = null;
    }

    set({
      stats: {
        datasetCount,
        modelCount,
        inferenceCount,
        systemStatus,
        gpuInfo,
      },
      lastUpdated: new Date().toISOString(),
    });
  },

  refreshRecentTasks: async () => {
    let recentTasks: RecentTask[] = [];
    try {
      const items = useInferenceHistoryStore.getState().items;
      recentTasks = items.slice(0, 10).map((i) => ({
        id: i.id,
        name: i.taskName,
        model: i.model,
        dataset: i.dataset,
        status: i.status === 'failed' ? 'failed' : 'completed',
        time: i.time,
      }));
    } catch {
      recentTasks = [];
    }
    set({ recentTasks });
  },

  refreshTrainingProgress: () => {
    let trainingProgress: TrainingTask[] = [];
    try {
      const s = useTrainingStore.getState();
      if (s.isRunning && s.progress) {
        trainingProgress = [
          {
            name: s.config.outputName || '当前训练任务',
            progress: Math.round((s.progress.epoch / (s.config.hyperParams.epochs || 1)) * 100),
          },
        ];
      }
    } catch {
      trainingProgress = [];
    }
    set({ trainingProgress });
  },
}));
