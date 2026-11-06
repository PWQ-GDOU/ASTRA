/** 模型架构（clean_benchmark.py 的 --model choices） */
export type TrainingModelArch = 'v2' | 'multiscale' | 'condition' | 'ttsnet';

/** 传感器模式（clean_benchmark.py 的 --sensor-mode choices） */
export type SensorMode = 'all24' | 'settings14' | 'condnorm24' | 'condnorm17' | 'tts14';

/** 计算设备 */
export type TrainingDevice = 'cpu' | 'cuda' | `cuda:${number}`;

export interface TrainingConfig {
  /** 数据管理中持久化的数据集 ID */
  dataset: string;
  trainFile?: string;
  testFile?: string;
  rulFile?: string;
  /** 模型架构，对应 --model（从零训练，无需预训练输入模型） */
  modelArch: TrainingModelArch;
  /** 输出模型名（可空，纯训练不需要保存命名） */
  outputName: string;
  /** 传感器模式，对应 --sensor-mode */
  sensorMode: SensorMode;
  /** 滑动窗口长度，对应 --seq-len */
  seqLen: number;
  /** RUL 标签上限，对应 --rul-cap */
  rulCap: number;
  /** 训练种子（逗号分隔多值），对应 --seeds */
  seeds: string;
  /** 早停耐心，对应 --patience */
  patience: number;
  /** 计算设备，对应 --device */
  device: TrainingDevice;
  dataRoot: string;
  outputDir: string;
  splitSeed: number;
  overWeight: number;
  valRatio: number;
  hyperParams: {
    epochs: number;
    batchSize: number;
    learningRate: number;
    optimizer: string;
  };
}

export interface TrainingProgressEvent {
  epoch: number;
  step: number;
  loss: number;
  valLoss?: number;
  log?: string;
  status?: 'running' | 'completed' | 'failed' | 'cancelled';
}

/** 训练完成后返回的指标（来自 clean_benchmark.py 结果 JSON） */
export interface TrainingResult {
  runId: string;
  seed: number;
  bestEpoch: number;
  valRmse: number;
  testRmse: number;
  testScore: number;
  testRmseCap: number;
  testScoreCap: number;
  nParams: number;
  seconds: number;
  history: Array<{ epoch: number; loss: number; valRmse: number; valScore: number }>;
}

export interface SystemStatus {
  cuda: { available: boolean; gpuName?: string; memory?: string };
  cpu: { usage: number; cores: number };
  ram: { used: string; total: string; percent: number };
}
