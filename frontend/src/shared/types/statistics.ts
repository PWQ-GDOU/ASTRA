export interface StatisticsQuery {
  dateRange?: [string, string];
}

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
