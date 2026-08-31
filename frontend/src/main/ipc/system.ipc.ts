import { ipcMain, BrowserWindow, app } from 'electron';
import { exec } from 'child_process';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import { cpus, totalmem, freemem } from 'os';

let mainWindow: BrowserWindow | null = null;
let cachedCudaStatus: {
  cudaAvailable: boolean;
  gpuName?: string;
  gpuMemory?: string;
  utilization?: number;
} | null = null;

export function setMainWindow(win: BrowserWindow): void {
  mainWindow = win;
}

/* ===== 真实 GPU/CUDA 检测 ===== */

interface GpuInfo {
  name: string;
  memoryTotalMb: number;
  memoryUsedMb: number;
  utilizationPercent: number;
}

/** 通过 nvidia-smi 获取 GPU 信息 */
function detectViaNvidiaSmi(): Promise<GpuInfo[]> {
  return new Promise((resolve) => {
    exec(
      'nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits',
      { timeout: 8000 },
      (error, stdout) => {
        if (error || !stdout.trim()) {
          resolve([]);
          return;
        }
        const gpus: GpuInfo[] = stdout
          .trim()
          .split('\n')
          .map((line) => {
            const parts = line.split(',').map((s) => s.trim());
            return {
              name: parts[0] || 'NVIDIA GPU',
              memoryTotalMb: parseInt(parts[1], 10) || 0,
              memoryUsedMb: parseInt(parts[2], 10) || 0,
              utilizationPercent: parseInt(parts[3], 10) || 0,
            };
          })
          .filter((g) => g.memoryTotalMb > 0);
        resolve(gpus);
      },
    );
  });
}

/** Electron API 后备: 枚举 GPUFeatureStatus */
async function detectViaElectron(): Promise<GpuInfo[]> {
  try {
    const info = (await app.getGPUInfo('basic')) as {
      gpuDevice?: Array<{ deviceString?: string }>;
    } | null;
    // app.getGPUInfo 返回的格式因平台而异，提取有意义的 GPU 信息
    if (info?.gpuDevice && Array.isArray(info.gpuDevice)) {
      const gpu = info.gpuDevice[0];
      if (gpu?.deviceString && gpu.deviceString.length > 0) {
        return [
          {
            name: gpu.deviceString,
            memoryTotalMb: 0,
            memoryUsedMb: 0,
            utilizationPercent: 0,
          },
        ];
      }
    }
  } catch {
    // Electron API not available or failed
  }
  return [];
}

/** 更新缓存的 CUDA 状态 */
export async function refreshCudaStatus(): Promise<void> {
  // 优先使用 nvidia-smi
  let gpus = await detectViaNvidiaSmi();

  // 后备: Electron API
  if (gpus.length === 0) {
    gpus = await detectViaElectron();
  }

  if (gpus.length > 0) {
    const gpu = gpus[0];
    const memTotalGb =
      gpu.memoryTotalMb > 0 ? `${(gpu.memoryTotalMb / 1024).toFixed(1)} GB` : undefined;
    const memUsedGb =
      gpu.memoryUsedMb > 0 ? `${(gpu.memoryUsedMb / 1024).toFixed(1)} GB` : undefined;
    cachedCudaStatus = {
      cudaAvailable: true,
      gpuName: gpu.name,
      gpuMemory: memUsedGb && memTotalGb ? `${memUsedGb} / ${memTotalGb}` : memTotalGb,
      utilization: gpu.utilizationPercent > 0 ? gpu.utilizationPercent : undefined,
    };
  } else {
    cachedCudaStatus = {
      cudaAvailable: false,
    };
  }
}

/** 获取当前缓存状态（首次调用时立刻刷新） */
function getCudaStatus() {
  return cachedCudaStatus ?? { cudaAvailable: false };
}

export function registerSystemHandlers(): void {
  // Window controls
  ipcMain.handle(IPC_CHANNELS.WINDOW_MINIMIZE, (): IPCResult<void> => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.minimize();
    }
    return { success: true, data: undefined };
  });

  ipcMain.handle(IPC_CHANNELS.WINDOW_MAXIMIZE, (): IPCResult<void> => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMaximized()) {
        mainWindow.unmaximize();
      } else {
        mainWindow.maximize();
      }
    }
    return { success: true, data: undefined };
  });

  ipcMain.handle(IPC_CHANNELS.WINDOW_CLOSE, (): IPCResult<void> => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.close();
    }
    return { success: true, data: undefined };
  });

  // CUDA status — real system detection (nvidia-smi + Electron fallback)
  // CPU/RAM 提前读取避免被 nvidia-smi 阻塞；GPU 走缓存
  ipcMain.handle(IPC_CHANNELS.SYSTEM_CUDA_STATUS, async () => {
    // 先读 CPU/RAM（同步，零延迟）
    let cpuCores: number | undefined;
    let ramUsed: string | undefined;
    let ramTotal: string | undefined;
    let ramPercent: number | undefined;

    try {
      cpuCores = cpus().length;
    } catch {
      cpuCores = undefined;
    }

    try {
      const total = totalmem();
      const free = freemem();
      const used = total - free;
      ramTotal = `${(total / (1024 * 1024 * 1024)).toFixed(0)} GB`;
      ramUsed = `${(used / (1024 * 1024 * 1024)).toFixed(0)} GB`;
      ramPercent = Math.round((used / total) * 100);
    } catch {
      ramTotal = undefined;
      ramUsed = undefined;
      ramPercent = undefined;
    }

    // GPU 检测异步，首次无缓存时用副触发不阻塞
    const gpuPromise: Promise<void> = (async () => {
      if (cachedCudaStatus === null) {
        await refreshCudaStatus();
      }
    })();

    // 不等 nvidia-smi 完成就先返回 CPU/RAM；GPU 用缓存或默认 false
    // 如果缓存已就绪（非首次调用），瞬间返回
    if (cachedCudaStatus !== null) {
      await gpuPromise; // 已缓存，瞬间完成
    }
    // 首次调用：不 await gpuPromise，立即返回 CPU/RAM + 默认 GPU=false
    // nvidia-smi 结果会在下一次轮询时通过缓存命中

    const status = getCudaStatus();

    return {
      success: true,
      data: {
        cudaAvailable: status.cudaAvailable,
        gpuName: status.gpuName,
        gpuMemory: status.gpuMemory,
        utilization: status.utilization,
        cpuCores,
        ramUsed,
        ramTotal,
        ramPercent,
      },
    };
  });
}
