import { create } from 'zustand';

export interface StatisticsData {
  trainingTrend: { epoch: number; loss: number; valLoss?: number }[];
  inferenceAccuracyTrend: { date: string; accuracy: number }[];
  datasetDistribution: { name: string; count: number; type: string }[];
  modelPerformance: {
    name: string;
    accuracy: number;
    speed: number;
    size: number;
    params: number;
  }[];
  taskStats: { completed: number; failed: number; running: number; pending: number };
  modelStatusDist: { status: string; count: number }[];
}

interface StatisticsState {
  data: StatisticsData | null;
  dateRange: [string, string] | null;
  isLoading: boolean;
  setData: (data: StatisticsData) => void;
  setDateRange: (range: [string, string] | null) => void;
  setLoading: (v: boolean) => void;
}

export const useStatisticsStore = create<StatisticsState>((set) => ({
  data: null,
  dateRange: null,
  isLoading: false,
  setData: (data) => set({ data }),
  setDateRange: (range) => set({ dateRange: range }),
  setLoading: (v) => set({ isLoading: v }),
}));
