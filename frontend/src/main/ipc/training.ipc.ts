import { ipcMain, dialog as electronDialog, BrowserWindow } from 'electron';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type { TrainingConfig, TrainingProgressEvent } from '../../shared/types/training';
import {
  readJSON,
  appendJSON,
  TRAINING_CONFIG_FILE,
  DATASETS_FILE,
  MODELS_FILE,
} from '../lib/storage';
import type { Dataset } from '../../shared/types/dataset';
import type { Model } from '../../shared/types/model';
import { spawnTraining, isAstraAvailable, OUTPUT_DIR, ASTRA_DIR } from '../lib/python-bridge';
import {
  parseEpochLine,
  isSeedStartLine,
  parsePrimarySeed,
  readTrainingResultFile,
} from '../lib/training-progress';
import { resolve } from 'path';

export type { ParsedProgress } from '../lib/training-progress';
export { parseEpochLine, parseResultLine, isSeedStartLine } from '../lib/training-progress';

/** 活跃训练进程：runId → Python 子进程 */
interface ActiveTraining {
  /** 真实子进程（spawn 模式） */
  process?: {
    kill: () => void;
  };
}

const activeTraining = new Map<string, ActiveTraining>();

function genId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function safeDatasetId(id: string): string {
  return Array.from(id)
    .map((char) => (/[\p{L}\p{N}_-]/u.test(char) ? char : '_'))
    .join('');
}

/* ===== 训练结果读取（任务 156）===== */

export function registerTrainingHandlers(mainWindow: BrowserWindow): void {
  ipcMain.handle(
    IPC_CHANNELS.TRAINING_START_LOCAL,
    async (_e, config: TrainingConfig): Promise<IPCResult<string>> => {
      try {
        const dataset = (readJSON<Dataset>(DATASETS_FILE) || []).find(
          (item) => item.id === config.dataset,
        );
        if (
          !dataset?.astraCompatible ||
          !dataset.trainFile ||
          !dataset.testFile ||
          !dataset.rulFile
        )
          return { success: false, error: '请选择已完成 C-MAPSS 文件映射的数据集' };
        const resolvedConfig = {
          ...config,
          trainFile: dataset.trainFile,
          testFile: dataset.testFile,
          rulFile: dataset.rulFile,
        };

        // 真实后端可用 → spawn ASTRA
        if (isAstraAvailable()) {
          const spawned = spawnTraining(
            resolvedConfig,
            (line) => {
              // 任务 155：解析 stdout 推进度
              const epoch = parseEpochLine(line);
              if (epoch) {
                const evt: TrainingProgressEvent = {
                  epoch: epoch.epoch,
                  step: epoch.epoch * 100,
                  loss: epoch.loss,
                  valLoss: epoch.valLoss,
                  status: 'running',
                };
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, evt);
                return;
              }
              // 日志行直接透传
              if (isSeedStartLine(line) || line.startsWith('FD=')) {
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                  epoch: 0,
                  step: 0,
                  loss: 0,
                  log: line,
                  status: 'running',
                } as TrainingProgressEvent);
              }
            },
            (line) => {
              mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                epoch: 0,
                step: 0,
                loss: 0,
                log: `[stderr] ${line}`,
                status: 'running',
              } satisfies TrainingProgressEvent);
            },
          );

          activeTraining.set(spawned.runId, { process: { kill: spawned.kill } });

          // 训练完成 → 读结果并推送
          spawned.promise
            .then(({ exitCode }) => {
              activeTraining.delete(spawned.runId);
              if (exitCode !== 0) {
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                  epoch: 0,
                  step: 0,
                  loss: 0,
                  log: `ASTRA 训练失败（退出码 ${exitCode ?? '未知'}）`,
                  status: 'failed',
                } satisfies TrainingProgressEvent);
                return;
              }
              const result = readTrainingResultFile(
                parsePrimarySeed(config.seeds || '42'),
                config.dataset || 'FD002',
                config.modelArch || 'multiscale',
                ASTRA_DIR,
                config.outputDir || OUTPUT_DIR,
                config.sensorMode || 'all24',
              );
              if (result) {
                result.runId = spawned.runId;
                const seed = parsePrimarySeed(config.seeds || '42');
                const checkpointFilename = `${safeDatasetId(config.dataset)}_${config.modelArch || 'multiscale'}_${config.sensorMode || 'all24'}_s${seed}.pt`;
                const checkpointPath = resolve(
                  ASTRA_DIR,
                  config.outputDir || OUTPUT_DIR,
                  checkpointFilename,
                );
                const models = readJSON<Model>(MODELS_FILE) || [];
                const existingModel = models.find((model) => model.path === checkpointPath);
                const generatedModel: Model = existingModel || {
                  id: genId(),
                  name: config.outputName?.trim() || checkpointFilename,
                  status: 'available',
                  source: 'trained',
                  isDefault: models.length === 0,
                  path: checkpointPath,
                  createdAt: new Date().toISOString(),
                };
                if (!existingModel) appendJSON(MODELS_FILE, generatedModel);
                mainWindow.webContents.send(IPC_CHANNELS.MODEL_STATUS, generatedModel);
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_RESULT, result);
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                  epoch: result.bestEpoch,
                  step: result.bestEpoch * 100,
                  loss: result.history.at(-1)?.loss ?? 0,
                  log: 'ASTRA 训练完成，结果和 checkpoint 已生成',
                  status: 'completed',
                } satisfies TrainingProgressEvent);
              } else {
                mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                  epoch: 0,
                  step: 0,
                  loss: 0,
                  log: 'ASTRA 已退出，但未找到训练结果文件',
                  status: 'failed',
                } satisfies TrainingProgressEvent);
              }
            })
            .catch(() => {
              activeTraining.delete(spawned.runId);
              mainWindow.webContents.send(IPC_CHANNELS.TRAINING_PROGRESS, {
                epoch: 0,
                step: 0,
                loss: 0,
                log: 'ASTRA 训练进程异常终止',
                status: 'failed',
              } satisfies TrainingProgressEvent);
            });

          return { success: true, data: spawned.runId };
        }

        // 后端不可用时明确返回错误，不伪造训练结果。
        return { success: false, error: `ASTRA 后端不可用：${ASTRA_DIR}` };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(IPC_CHANNELS.TRAINING_START_REMOTE, (): IPCResult<string> => {
    try {
      return { success: false, error: '远程训练尚未接入真实 ASTRA 服务' };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.TRAINING_STOP, (_e, runId: string): IPCResult<void> => {
    try {
      const training = activeTraining.get(runId);
      if (training) {
        if (training.process) {
          training.process.kill();
        }
        activeTraining.delete(runId);
      }
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(
    IPC_CHANNELS.TRAINING_CONFIG_SAVE,
    (_e, config: TrainingConfig): IPCResult<void> => {
      try {
        appendJSON(TRAINING_CONFIG_FILE, config);
        return { success: true, data: undefined };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(IPC_CHANNELS.TRAINING_CONFIG_LOAD, (): IPCResult<TrainingConfig[]> => {
    try {
      const configs = readJSON<TrainingConfig>(TRAINING_CONFIG_FILE) || [];
      return { success: true, data: configs as TrainingConfig[] };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });

  ipcMain.handle(IPC_CHANNELS.TRAINING_LOG_PATH_SELECT, async (): Promise<IPCResult<string>> => {
    try {
      const result = await electronDialog.showOpenDialog({
        properties: ['openFile'],
        filters: [{ name: 'Log Files', extensions: ['log', 'txt'] }],
      });
      if (result.canceled) {
        return { success: false, error: 'Cancelled' };
      }
      return { success: true, data: result.filePaths[0] };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });
}
