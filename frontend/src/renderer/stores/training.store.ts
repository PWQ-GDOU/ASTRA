import { create } from 'zustand';
import type { TrainingConfig as SharedTrainingConfig } from '../../shared/types/training';

export type TrainingMode = 'remote' | 'local';

export type TrainingConfig = SharedTrainingConfig;

export interface TrainingProgress {
  epoch: number;
  step: number;
  loss: number;
  valLoss?: number;
  log: string[];
}

export interface SystemStatus {
  cuda: { available: boolean; gpuName?: string; memory?: string };
  cpu: { usage: number; cores: number };
  ram: { used: string; total: string; percent: number };
}

interface TrainingState {
  mode: TrainingMode;
  config: TrainingConfig;
  progress: TrainingProgress | null;
  systemStatus: SystemStatus;
  isRunning: boolean;
  savedConfigs: TrainingConfig[];
  setMode: (mode: TrainingMode) => void;
  setConfig: (config: Partial<TrainingConfig>) => void;
  setProgress: (
    progress:
      TrainingProgress | null | ((prev: TrainingProgress | null) => TrainingProgress | null),
  ) => void;
  addLog: (line: string) => void;
  setSystemStatus: (status: SystemStatus) => void;
  setIsRunning: (v: boolean) => void;
  saveConfig: (config: TrainingConfig) => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  mode: 'local',
  config: {
    dataset: '',
    modelArch: 'multiscale',
    outputName: '',
    sensorMode: 'all24',
    seqLen: 60,
    rulCap: 125.0,
    seeds: '42',
    patience: 60,
    device: 'cuda',
    dataRoot: 'data/processed',
    outputDir: 'outputs/clean_benchmark',
    splitSeed: 2026,
    overWeight: 0.02,
    valRatio: 0.2,
    hyperParams: {
      epochs: 10,
      batchSize: 32,
      learningRate: 0.001,
      optimizer: 'adamw',
    },
  },
  progress: null,
  systemStatus: {
    cuda: { available: false },
    cpu: { usage: 0, cores: 0 },
    ram: { used: '...', total: '...', percent: 0 },
  },
  isRunning: false,
  savedConfigs: [],

  setMode: (mode) => set({ mode }),
  setConfig: (config) => set((s) => ({ config: { ...s.config, ...config } })),
  setProgress: (
    progress:
      TrainingProgress | null | ((prev: TrainingProgress | null) => TrainingProgress | null),
  ) =>
    set((s) => ({
      ...s,
      progress: typeof progress === 'function' ? progress(s.progress) : progress,
    })),
  addLog: (line) =>
    set((s) => ({
      progress: s.progress ? { ...s.progress, log: [...s.progress.log, line] } : null,
    })),
  setSystemStatus: (status) => set({ systemStatus: status }),
  setIsRunning: (v) => set({ isRunning: v }),
  saveConfig: (config) => set((s) => ({ savedConfigs: [...s.savedConfigs, config] })),
}));
