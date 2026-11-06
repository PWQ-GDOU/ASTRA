import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn, type ChildProcess } from 'child_process';
import { existsSync, readFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type {
  InferenceRequest,
  InferenceResult,
  InferenceProgressEvent,
} from '../../shared/types/inference';
import type { Model } from '../../shared/types/model';
import type { Dataset } from '../../shared/types/dataset';
import { ASTRA_DIR } from '../lib/python-bridge';
import {
  MODELS_FILE,
  DATASETS_FILE,
  INFERENCE_HISTORY_FILE,
  readJSON,
  appendJSON,
  updateJSON,
  deleteJSON,
} from '../lib/storage';

const activeProcesses = new Map<string, ChildProcess>();

interface StoredInferenceHistory {
  id: string;
  taskName: string;
  model: string;
  dataset: string;
  status: 'completed' | 'failed';
  duration: string;
  time: string;
  metrics: Record<string, number>;
  log: string[];
}

interface InferenceHistoryFilters {
  model?: string;
  status?: string;
  search?: string;
  dateRange?: [string, string] | null;
}

function genId(): string {
  return `${Date.now().toString(36)}-${process.pid}-${Math.random().toString(36).slice(2, 8)}`;
}

function isFiniteNumberArray(value: unknown): value is number[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isFinite(item))
  );
}

function sendProgress(mainWindow: BrowserWindow, event: InferenceProgressEvent): void {
  mainWindow.webContents.send(IPC_CHANNELS.INFERENCE_PROGRESS, event);
}

export function registerInferenceHandlers(mainWindow: BrowserWindow): void {
  ipcMain.handle(
    IPC_CHANNELS.INFERENCE_HISTORY_LIST,
    (_event, filters: InferenceHistoryFilters = {}): IPCResult<StoredInferenceHistory[]> => {
      try {
        const search = filters.search?.trim().toLocaleLowerCase();
        const [dateFrom, dateTo] = filters.dateRange || [];
        const history = (readJSON<StoredInferenceHistory>(INFERENCE_HISTORY_FILE) || [])
          .filter((item) => !filters.model || item.model === filters.model)
          .filter((item) => !filters.status || item.status === filters.status)
          .filter((item) => {
            if (!search) return true;
            return [item.taskName, item.model, item.dataset].some((value) =>
              value.toLocaleLowerCase().includes(search),
            );
          })
          .filter(
            (item) => (!dateFrom || item.time >= dateFrom) && (!dateTo || item.time <= dateTo),
          )
          .sort((a, b) => b.time.localeCompare(a.time));
        return { success: true, data: history };
      } catch (error) {
        return { success: false, error: error instanceof Error ? error.message : String(error) };
      }
    },
  );
  ipcMain.handle(
    IPC_CHANNELS.INFERENCE_HISTORY_DETAIL,
    (_event, id: string): IPCResult<StoredInferenceHistory> => {
      const item = (readJSON<StoredInferenceHistory>(INFERENCE_HISTORY_FILE) || []).find(
        (history) => history.id === id,
      );
      return item ? { success: true, data: item } : { success: false, error: '推理记录不存在' };
    },
  );
  ipcMain.handle(IPC_CHANNELS.INFERENCE_HISTORY_DELETE, (_event, id: string): IPCResult<void> => {
    try {
      deleteJSON(INFERENCE_HISTORY_FILE, id);
      return { success: true, data: undefined };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : String(error) };
    }
  });
  ipcMain.handle(IPC_CHANNELS.INFERENCE_HISTORY_EXPORT, (): IPCResult<void> => ({
    success: false,
    error: '当前版本暂不支持历史记录导出',
  }));

  ipcMain.handle(
    IPC_CHANNELS.INFERENCE_RUN,
    (_event, request: InferenceRequest): IPCResult<string> => {
      try {
        const model = (readJSON<Model>(MODELS_FILE) || []).find(
          (item) => item.id === request.modelId,
        );
        const dataset = (readJSON<Dataset>(DATASETS_FILE) || []).find(
          (item) => item.id === request.datasetId,
        );
        if (!model || !existsSync(model.path)) throw new Error('所选 checkpoint 不存在');
        if (
          !dataset?.astraCompatible ||
          !dataset.trainFile ||
          !dataset.testFile ||
          !dataset.rulFile
        )
          throw new Error('请选择已完成 C-MAPSS 文件映射的数据集');
        if (!Number.isInteger(request.batchSize) || request.batchSize < 1)
          throw new Error('批大小必须为正整数');
        if (!/^(cpu|cuda(?::\d+)?)$/.test(request.device)) throw new Error('推理设备参数无效');
        if (
          request.applyRulCap &&
          (typeof request.rulCap !== 'number' ||
            !Number.isFinite(request.rulCap) ||
            request.rulCap <= 0)
        )
          throw new Error('RUL 上限必须为正数');

        const taskId = genId();
        const outputPath = join(app.getPath('temp'), `astra-inference-${taskId}.json`);
        const inferenceScript = join('scripts', 'predict_checkpoint.py');
        if (!existsSync(join(ASTRA_DIR, inferenceScript)))
          throw new Error(`ASTRA 推理脚本不存在：${join(ASTRA_DIR, inferenceScript)}`);
        const args = [
          inferenceScript,
          '--checkpoint',
          model.path,
          '--dataset-id',
          dataset.id,
          '--train-file',
          dataset.trainFile,
          '--test-file',
          dataset.testFile,
          '--rul-file',
          dataset.rulFile,
          '--device',
          request.device,
          '--batch-size',
          String(request.batchSize),
          '--output',
          outputPath,
        ];
        if (request.applyRulCap) args.push('--rul-cap', String(request.rulCap));
        const child = spawn(process.env.ASTRA_PYTHON || 'python', ['-u', ...args], {
          cwd: ASTRA_DIR,
          windowsHide: true,
        });
        activeProcesses.set(taskId, child);
        sendProgress(mainWindow, { taskId, progress: 5, status: 'running', elapsed: 0 });

        let stderr = '';
        let settled = false;
        const startedAt = Date.now();
        const persistHistory = (
          status: 'completed' | 'failed',
          seconds: number,
          metrics: { rmse?: number; mae?: number; score?: number },
          log: string[],
        ) => {
          try {
            appendJSON(INFERENCE_HISTORY_FILE, {
              id: taskId,
              taskName: `推理-${dataset.name}`,
              model: model.name,
              dataset: dataset.name,
              status,
              duration: `${seconds.toFixed(2)}s`,
              time: new Date().toISOString(),
              metrics: { ...metrics, latency: seconds },
              log,
            });
          } catch {
            // 历史写入失败不应吞掉已经成功返回的 ASTRA 推理结果。
          }
        };
        const fail = (message: string) => {
          if (settled) return;
          settled = true;
          activeProcesses.delete(taskId);
          const seconds = (Date.now() - startedAt) / 1000;
          persistHistory('failed', seconds, {}, [message]);
          try {
            unlinkSync(outputPath);
          } catch {
            // 进程可能尚未生成结果文件。
          }
          sendProgress(mainWindow, {
            taskId,
            progress: 0,
            status: 'failed',
            elapsed: seconds,
            message,
          });
        };
        child.stderr?.on('data', (chunk: Buffer) => {
          stderr += chunk.toString();
        });
        child.on('close', (code) => {
          if (code !== 0) {
            fail(stderr.trim() || `ASTRA 推理失败（退出码 ${code ?? '未知'}）`);
            return;
          }
          try {
            const payload = JSON.parse(readFileSync(outputPath, 'utf8')) as {
              units: number[];
              predictions: number[];
              targets?: number[];
              metrics: { rmse?: number; mae?: number; score?: number };
              seconds: number;
            };
            if (
              !isFiniteNumberArray(payload.units) ||
              !isFiniteNumberArray(payload.predictions) ||
              payload.units.length === 0 ||
              payload.units.length !== payload.predictions.length ||
              (payload.targets !== undefined &&
                (!isFiniteNumberArray(payload.targets) ||
                  payload.targets.length !== payload.units.length)) ||
              !payload.metrics ||
              typeof payload.metrics !== 'object' ||
              !Object.values(payload.metrics).every(
                (value) => typeof value === 'number' && Number.isFinite(value),
              ) ||
              typeof payload.seconds !== 'number' ||
              !Number.isFinite(payload.seconds) ||
              payload.seconds < 0
            ) {
              throw new Error('ASTRA 推理结果格式无效');
            }
            const result: InferenceResult = {
              taskId,
              units: payload.units,
              predictions: payload.predictions,
              targets: payload.targets,
              rmse: payload.metrics.rmse,
              mae: payload.metrics.mae,
              score: payload.metrics.score,
              seconds: payload.seconds,
            };
            settled = true;
            activeProcesses.delete(taskId);
            persistHistory('completed', payload.seconds, payload.metrics, ['ASTRA 推理完成']);
            const evaluatedModel: Model = {
              ...model,
              performance: {
                taskId,
                datasetId: dataset.id,
                datasetName: dataset.name,
                rmse: payload.metrics.rmse,
                mae: payload.metrics.mae,
                score: payload.metrics.score,
                sampleCount: payload.units.length,
                seconds: payload.seconds,
                evaluatedAt: new Date().toISOString(),
              },
            };
            try {
              updateJSON<Model>(MODELS_FILE, model.id, (storedModel) => ({
                ...storedModel,
                performance: evaluatedModel.performance,
              }));
            } catch {
              // 模型指标写入失败不应吞掉已经完成的推理结果。
            }
            mainWindow.webContents.send(IPC_CHANNELS.MODEL_STATUS, evaluatedModel);
            sendProgress(mainWindow, {
              taskId,
              progress: 100,
              status: 'completed',
              elapsed: payload.seconds,
            });
            mainWindow.webContents.send(IPC_CHANNELS.INFERENCE_RESULT, result);
            try {
              unlinkSync(outputPath);
            } catch {
              /* 临时文件清理失败不影响结果 */
            }
          } catch (error) {
            fail(
              `ASTRA 推理结果读取失败：${error instanceof Error ? error.message : String(error)}`,
            );
          }
        });
        child.on('error', (error) => {
          fail(`ASTRA 推理进程启动失败：${error instanceof Error ? error.message : String(error)}`);
        });
        return { success: true, data: taskId };
      } catch (error) {
        return { success: false, error: error instanceof Error ? error.message : String(error) };
      }
    },
  );

  ipcMain.handle(IPC_CHANNELS.INFERENCE_RUN_BATCH, (): IPCResult<string[]> => ({
    success: false,
    error: '暂不支持批量任务',
  }));
  ipcMain.handle(IPC_CHANNELS.INFERENCE_CANCEL, (_event, taskId: string): IPCResult<void> => {
    activeProcesses.get(taskId)?.kill();
    activeProcesses.delete(taskId);
    return { success: true, data: undefined };
  });
}
