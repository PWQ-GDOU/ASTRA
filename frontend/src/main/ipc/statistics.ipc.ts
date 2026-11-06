import { ipcMain } from 'electron';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type { StatisticsData } from '../../shared/types/statistics';
import type { Dataset } from '../../shared/types/dataset';
import type { Model } from '../../shared/types/model';
import { DATASETS_FILE, INFERENCE_HISTORY_FILE, MODELS_FILE, readJSON } from '../lib/storage';

interface StoredInference {
  status?: string;
  time?: string;
  metrics?: { rmse?: number };
}

export function registerStatisticsHandlers(): void {
  ipcMain.handle(IPC_CHANNELS.STATISTICS_QUERY, (): IPCResult<StatisticsData> => {
    try {
      const datasets = readJSON<Dataset>(DATASETS_FILE) || [];
      const models = readJSON<Model>(MODELS_FILE) || [];
      const history = readJSON<StoredInference>(INFERENCE_HISTORY_FILE) || [];
      const counts = (values: string[]) =>
        values.reduce<Record<string, number>>((acc, value) => {
          acc[value] = (acc[value] || 0) + 1;
          return acc;
        }, {});
      const datasetCounts = counts(datasets.map((item) => item.type));
      const modelCounts = counts(models.map((item) => item.status));
      const taskCounts = counts(history.map((item) => item.status || 'failed'));
      const data: StatisticsData = {
        trainingTrend: [],
        inferenceAccuracyTrend: history
          .filter((item) => item.time && item.metrics?.rmse != null)
          .map((item) => ({ date: item.time!.slice(0, 10), accuracy: item.metrics!.rmse! })),
        datasetDistribution: Object.entries(datasetCounts).map(([type, count]) => ({
          name: type,
          count,
          type,
        })),
        modelPerformance: [],
        taskStats: {
          completed: taskCounts.completed || 0,
          failed: taskCounts.failed || 0,
          running: taskCounts.running || 0,
          pending: taskCounts.pending || 0,
        },
        modelStatusDist: Object.entries(modelCounts).map(([status, count]) => ({ status, count })),
      };
      return { success: true, data };
    } catch (error) {
      return { success: false, error: String(error) };
    }
  });
}
