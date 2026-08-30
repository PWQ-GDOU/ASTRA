import { create } from 'zustand';

export type InferenceStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface InferenceRequest {
  datasetId: string;
  modelId: string;
  name?: string;
  batchSize?: number;
}

export interface InferenceProgress {
  taskId: string;
  progress: number;
  status: InferenceStatus;
  log: string[];
  elapsed?: number;
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

export interface TaskQueueItem {
  id: string;
  name: string;
  model: string;
  dataset: string;
  status: InferenceStatus;
  progress: number;
  createdAt: string;
}

interface InferenceState {
  activeTask: InferenceProgress | null;
  queue: TaskQueueItem[];
  result: InferenceResult | null;
  isRunning: boolean;
  setActiveTask: (task: InferenceProgress | null) => void;
  updateProgress: (progress: Partial<InferenceProgress>) => void;
  addLog: (line: string) => void;
  setResult: (result: InferenceResult | null) => void;
  setQueue: (queue: TaskQueueItem[]) => void;
  addToQueue: (item: TaskQueueItem) => void;
  updateQueueItem: (id: string, updates: Partial<TaskQueueItem>) => void;
  setIsRunning: (v: boolean) => void;
}

export const useInferenceStore = create<InferenceState>((set) => ({
  activeTask: null,
  queue: [],
  result: null,
  isRunning: false,

  setActiveTask: (task) => set({ activeTask: task }),
  updateProgress: (progress) =>
    set((s) => ({
      activeTask: s.activeTask ? { ...s.activeTask, ...progress } : null,
    })),
  addLog: (line) =>
    set((s) => ({
      activeTask: s.activeTask ? { ...s.activeTask, log: [...s.activeTask.log, line] } : null,
    })),
  setResult: (result) => set({ result }),
  setQueue: (queue) => set({ queue }),
  addToQueue: (item) => set((s) => ({ queue: [...s.queue, item] })),
  updateQueueItem: (id, updates) =>
    set((s) => ({
      queue: s.queue.map((q) => (q.id === id ? { ...q, ...updates } : q)),
    })),
  setIsRunning: (v) => set({ isRunning: v }),
}));
