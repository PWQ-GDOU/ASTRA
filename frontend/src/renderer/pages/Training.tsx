import React, { useCallback, useEffect, useState } from 'react';
import { useTrainingStore } from '../stores/training.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import ProgressBar from '../components/common/ProgressBar';
import LogViewer from '../components/common/LogViewer';
import RemoteConnectionDialog from '../components/remote/RemoteConnectionDialog';
import { Play, Square, Wifi, WifiOff, Settings } from 'lucide-react';
import { useDatasetStore } from '../stores/dataset.store';

const MODEL_ARCHS = ['v2', 'multiscale', 'condition', 'ttsnet'];
const SENSOR_MODES = ['all24', 'settings14', 'condnorm24', 'condnorm17', 'tts14'];

const Training: React.FC = () => {
  const datasets = useDatasetStore((s) => s.datasets).filter((d) => d.astraCompatible);
  const {
    mode,
    config,
    progress,

    isRunning,
    setMode,
    setConfig,
    setIsRunning,
    setProgress,
  } = useTrainingStore();

  // 远程连接状态
  const [remoteConnected, setRemoteConnected] = useState(false);
  // 远程连接配置弹窗
  const [remoteDialogOpen, setRemoteDialogOpen] = useState(false);
  const [numericDrafts, setNumericDrafts] = useState<Record<string, string>>({});
  // 已连接的服务器信息
  const [connectedServer, setConnectedServer] = useState<{
    host: string;
    port: number;
    username: string;
  } | null>(null);

  // Active training runId + final result
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [finalResult, setFinalResult] = useState<{
    valRmse: number;
    testRmse: number;
    testScore: number;
  } | null>(null);

  // Live system status from main process
  const [liveSysStatus, setLiveSysStatus] = useState({
    cudaAvailable: false,
    gpuName: undefined as string | undefined,
    gpuMemory: undefined as string | undefined,
    cpuCores: 0,
    ramUsed: '...',
    ramTotal: '...',
    ramPercent: 0,
  });

  useEffect(() => {
    const poll = () => {
      if (window.electronAPI?.systemCudaStatus) {
        window.electronAPI
          .systemCudaStatus()
          .then((r: unknown) => {
            const d = (
              r as {
                success: boolean;
                data?: {
                  cudaAvailable: boolean;
                  gpuName?: string;
                  gpuMemory?: string;
                  utilization?: number;
                  cpuCores?: number;
                  ramUsed?: string;
                  ramTotal?: string;
                  ramPercent?: number;
                };
              }
            ).data;
            if (d) {
              setLiveSysStatus({
                cudaAvailable: d.cudaAvailable,
                gpuName: d.gpuName,
                gpuMemory: d.gpuMemory,
                cpuCores: d.cpuCores ?? 0,
                ramUsed: d.ramUsed ?? '...',
                ramTotal: d.ramTotal ?? '...',
                ramPercent: d.ramPercent ?? 0,
              });
            }
          })
          .catch(() => {});
      }
    };
    // Immediate poll with a retry at 1s to catch preload readiness
    poll();
    const retry = setTimeout(poll, 1000);
    // Keep updating every 5 s
    const iv = setInterval(poll, 5000);

    // 训练进度推送 → 更新 store
    let unsubscribeProgress: (() => void) | undefined;
    let unsubscribeResult: (() => void) | undefined;
    if (window.electronAPI?.onTrainingProgress) {
      unsubscribeProgress = window.electronAPI.onTrainingProgress((evt: unknown) => {
        const e = evt as {
          epoch: number;
          step: number;
          loss: number;
          valLoss?: number;
          log?: string;
          status?: 'running' | 'completed' | 'failed' | 'cancelled';
        };
        setProgress((p) => {
          const log =
            p?.log && e.log !== undefined ? [...p.log, e.log] : p?.log || (e.log ? [e.log] : []);
          return {
            epoch: e.epoch,
            step: e.step,
            loss: e.loss,
            valLoss: e.valLoss,
            log,
          };
        });
        if (e.status === 'failed' || e.status === 'completed' || e.status === 'cancelled') {
          setIsRunning(false);
          setActiveRunId(null);
        } else if (e.epoch > 0 || e.status === 'running') {
          setIsRunning(true);
        }
      });
    }
    if (window.electronAPI?.onTrainingResult) {
      unsubscribeResult = window.electronAPI.onTrainingResult((result: unknown) => {
        const r = result as {
          valRmse: number;
          testRmse: number;
          testScore: number;
        };
        setFinalResult(r);
        setIsRunning(false);
      });
    }

    return () => {
      clearTimeout(retry);
      clearInterval(iv);
      unsubscribeProgress?.();
      unsubscribeResult?.();
    };
  }, [setProgress, setIsRunning]);

  const handleStartTraining = useCallback(async () => {
    if (isRunning) {
      // 停止训练
      if (window.electronAPI?.trainingStop) {
        await window.electronAPI.trainingStop(activeRunId || '');
      }
      setIsRunning(false);
      setActiveRunId(null);
      setProgress(null);
      return;
    }
    if (!config.dataset) {
      alert('请先选择数据集');
      return;
    }
    if (mode === 'remote' && !remoteConnected) {
      alert('远程模式需要先连接服务器，请点击「测试连接」');
      return;
    }

    if (mode === 'local' && window.electronAPI?.trainingStartLocal) {
      // 真实后端：调用 IPC spawn ASTRA 训练
      setProgress({
        epoch: 0,
        step: 0,
        loss: 0,
        log: [`[${new Date().toLocaleTimeString()}] 训练启动 (本地 → ASTRA)...`],
      });
      setIsRunning(true);
      setFinalResult(null);
      try {
        const res = await window.electronAPI.trainingStartLocal(config);
        const r = res as { success: boolean; data?: string; error?: string };
        if (r.success && r.data) {
          setActiveRunId(r.data);
        } else {
          setProgress((p) =>
            p
              ? {
                  ...p,
                  log: [...p.log, `[错误] ${r.error || '启动失败'}`],
                }
              : null,
          );
          setIsRunning(false);
        }
      } catch (err) {
        setProgress((p) => (p ? { ...p, log: [...p.log, `[错误] ${String(err)}`] } : null));
        setIsRunning(false);
      }
      return;
    }

    setIsRunning(false);
    setProgress({
      epoch: 0,
      step: 0,
      loss: 0,
      log: [`[错误] ${mode === 'remote' ? '远程训练尚未接入真实 ASTRA 服务' : 'ASTRA 后端不可用'}`],
    });
    let epoch = 0;
    const interval = setInterval(() => {
      epoch++;
      if (epoch > config.hyperParams.epochs) {
        clearInterval(interval);
        setIsRunning(false);
        setProgress((p) =>
          p
            ? {
                ...p,
                epoch: config.hyperParams.epochs,
                log: [...(p.log || []), `[${new Date().toLocaleTimeString()}] 训练完成`],
              }
            : null,
        );
        return;
      }
      setProgress({
        epoch,
        step: epoch * 100,
        loss: 2.5 - epoch * (2.0 / config.hyperParams.epochs) + Math.random() * 0.2,
        valLoss: 2.6 - epoch * (1.8 / config.hyperParams.epochs) + Math.random() * 0.25,
        log: [
          `[Epoch ${epoch}] loss=${(2.0 - epoch * 0.1).toFixed(4)}  val_loss=${(2.1 - epoch * 0.09).toFixed(4)}`,
        ],
      });
    }, 1200);
  }, [isRunning, mode, config, remoteConnected, setIsRunning, setProgress, activeRunId]);

  const handleConfigChange = useCallback(
    (field: string, value: string | number) => {
      // 直接字段（dataset / modelArch / outputName / sensorMode / device / seeds）
      if (
        field === 'dataset' ||
        field === 'modelArch' ||
        field === 'outputName' ||
        field === 'sensorMode' ||
        field === 'device' ||
        field === 'seeds' ||
        field === 'dataRoot' ||
        field === 'outputDir'
      ) {
        setConfig({ [field]: value });
      } else if (
        field === 'seqLen' ||
        field === 'rulCap' ||
        field === 'patience' ||
        field === 'splitSeed' ||
        field === 'overWeight' ||
        field === 'valRatio'
      ) {
        setConfig({ [field]: Number(value) || 0 });
      } else if (field === 'epochs' || field === 'batchSize' || field === 'learningRate') {
        setConfig({ hyperParams: { ...config.hyperParams, [field]: Number(value) || 0 } });
      } else if (field === 'optimizer') {
        setConfig({ hyperParams: { ...config.hyperParams, optimizer: String(value) } });
      }
    },
    [config, setConfig],
  );

  const numericValue = (field: string, value: number) => numericDrafts[field] ?? String(value);
  const editNumber = (field: string, value: string) =>
    setNumericDrafts((drafts) => ({ ...drafts, [field]: value }));
  const commitNumber = (field: string, fallback: number) => {
    const draft = numericDrafts[field];
    if (draft === undefined) return;
    const parsed = Number(draft);
    if (draft.trim() && Number.isFinite(parsed)) handleConfigChange(field, parsed);
    setNumericDrafts((drafts) => {
      const next = { ...drafts };
      delete next[field];
      return next;
    });
    if (!draft.trim() || !Number.isFinite(parsed)) handleConfigChange(field, fallback);
  };

  return (
    <div className="space-y-5">
      <PageHeader
        title="训练模块"
        description="远程 / 本地训练配置与管理"
        actions={
          <div className="flex items-center gap-2">
            <div
              className="flex rounded-lg overflow-hidden"
              style={{ border: '1px solid var(--border)' }}
            >
              <button
                onClick={() => {
                  setMode('local');
                  setRemoteConnected(false);
                }}
                className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  background: mode === 'local' ? 'var(--accent)' : 'transparent',
                  color: mode === 'local' ? '#fff' : 'var(--foreground-muted)',
                }}
              >
                <Wifi size={12} /> 本地
              </button>
              <button
                onClick={() => setMode('remote')}
                className="flex items-center gap-1 px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  background: mode === 'remote' ? 'var(--accent)' : 'transparent',
                  color: mode === 'remote' ? '#fff' : 'var(--foreground-muted)',
                }}
              >
                <WifiOff size={12} /> 远程
              </button>
            </div>
            {/* 服务器状态指示 */}
            <div
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg cursor-pointer transition-colors"
              style={{
                border: '1px solid var(--border)',
                background: 'var(--surface)',
              }}
              onClick={() => setRemoteDialogOpen(true)}
              title="点击配置服务器"
            >
              <span
                className="inline-block rounded-full"
                style={{
                  width: 8,
                  height: 8,
                  backgroundColor: remoteConnected
                    ? 'var(--success)'
                    : mode === 'remote'
                      ? 'var(--warning)'
                      : 'var(--foreground-dim)',
                  boxShadow: remoteConnected ? '0 0 6px var(--success-glow)' : undefined,
                }}
              />
              <span className="text-xs" style={{ color: 'var(--foreground-muted)' }}>
                {remoteConnected
                  ? connectedServer
                    ? `${connectedServer.username}@${connectedServer.host}`
                    : '已连接'
                  : mode === 'remote'
                    ? '未连接'
                    : '服务器'}
              </span>
            </div>

            <button
              onClick={() => setRemoteDialogOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
              style={{
                background: 'var(--surface)',
                color: 'var(--foreground)',
                border: '1px solid var(--border)',
              }}
            >
              <Settings size={12} />
              配置服务器
            </button>

            <button
              onClick={handleStartTraining}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
              style={{ background: isRunning ? 'var(--danger)' : 'var(--success)', color: '#fff' }}
            >
              {isRunning ? <Square size={14} /> : <Play size={14} />}
              {isRunning ? '停止' : '开始训练'}
            </button>
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-5">
        <GlassPanel title="训练配置">
          <div className="p-4 space-y-3">
            <ConfigField label="数据集">
              <select
                value={config.dataset}
                onChange={(e) => handleConfigChange('dataset', e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm outline-none cursor-pointer transition-colors hover:brightness-95"
                style={{
                  background: 'var(--surface)',
                  color: 'var(--foreground)',
                  border: '1px solid var(--border)',
                }}
              >
                <option value="">-- 选择数据集 --</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
            </ConfigField>

            <ConfigField label="模型架构">
              <select
                value={config.modelArch}
                onChange={(e) => handleConfigChange('modelArch', e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm outline-none cursor-pointer transition-colors hover:brightness-95"
                style={{
                  background: 'var(--surface)',
                  color: 'var(--foreground)',
                  border: '1px solid var(--border)',
                }}
              >
                {MODEL_ARCHS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </ConfigField>

            <ConfigField label="传感器模式">
              <select
                value={config.sensorMode}
                onChange={(e) => handleConfigChange('sensorMode', e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm outline-none cursor-pointer transition-colors hover:brightness-95"
                style={{
                  background: 'var(--surface)',
                  color: 'var(--foreground)',
                  border: '1px solid var(--border)',
                }}
              >
                {SENSOR_MODES.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </ConfigField>

            <ConfigField label="输出模型名（可空）">
              <input
                type="text"
                placeholder="纯训练可不填..."
                value={config.outputName}
                onChange={(e) => handleConfigChange('outputName', e.target.value)}
                className="text-right font-mono text-sm bg-transparent outline-none w-40"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>

            <ConfigField label="序列长度">
              <input
                type="number"
                min={1}
                value={numericValue('seqLen', config.seqLen)}
                onChange={(e) => editNumber('seqLen', e.target.value)}
                onBlur={() => commitNumber('seqLen', config.seqLen)}
                className="text-right font-mono text-sm bg-transparent outline-none w-20"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>

            <ConfigField label="RUL 上限">
              <input
                type="number"
                min={1}
                value={numericValue('rulCap', config.rulCap)}
                onChange={(e) => editNumber('rulCap', e.target.value)}
                onBlur={() => commitNumber('rulCap', config.rulCap)}
                className="text-right font-mono text-sm bg-transparent outline-none w-20"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>

            <ConfigField label="种子">
              <input
                type="text"
                placeholder="42,123,456"
                value={config.seeds}
                onChange={(e) => handleConfigChange('seeds', e.target.value)}
                className="text-right font-mono text-sm bg-transparent outline-none w-24"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>

            <ConfigField label="设备">
              <select
                value={config.device}
                onChange={(e) => handleConfigChange('device', e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm outline-none cursor-pointer transition-colors hover:brightness-95"
                style={{
                  background: 'var(--surface)',
                  color: 'var(--foreground)',
                  border: '1px solid var(--border)',
                }}
              >
                <option value="cuda">cuda</option>
                <option value="cuda:0">cuda:0</option>
                <option value="cuda:1">cuda:1</option>
                <option value="cuda:2">cuda:2</option>
                <option value="cpu">cpu</option>
              </select>
            </ConfigField>

            <ConfigField label="划分种子">
              <input
                type="number"
                value={numericValue('splitSeed', config.splitSeed)}
                onChange={(e) => editNumber('splitSeed', e.target.value)}
                onBlur={() => commitNumber('splitSeed', config.splitSeed)}
                className="text-right font-mono text-sm bg-transparent outline-none w-24"
              />
            </ConfigField>
            <ConfigField label="高估惩罚权重">
              <input
                type="number"
                min={0}
                step={0.01}
                value={numericValue('overWeight', config.overWeight)}
                onChange={(e) => editNumber('overWeight', e.target.value)}
                onBlur={() => commitNumber('overWeight', config.overWeight)}
                className="text-right font-mono text-sm bg-transparent outline-none w-24"
              />
            </ConfigField>
            <ConfigField label="验证集比例">
              <input
                type="number"
                min={0.05}
                max={0.5}
                step={0.05}
                value={numericValue('valRatio', config.valRatio)}
                onChange={(e) => editNumber('valRatio', e.target.value)}
                onBlur={() => commitNumber('valRatio', config.valRatio)}
                className="text-right font-mono text-sm bg-transparent outline-none w-24"
              />
            </ConfigField>
            <ConfigField label="输出目录">
              <input
                type="text"
                value={config.outputDir}
                onChange={(e) => handleConfigChange('outputDir', e.target.value)}
                className="text-right font-mono text-sm bg-transparent outline-none w-48"
              />
            </ConfigField>

            <ConfigField label="Epochs">
              <input
                type="number"
                min={1}
                max={1000}
                value={numericValue('epochs', config.hyperParams.epochs)}
                onChange={(e) => editNumber('epochs', e.target.value)}
                onBlur={() => commitNumber('epochs', config.hyperParams.epochs)}
                className="text-right font-mono text-sm bg-transparent outline-none w-20"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>
            <ConfigField label="Batch Size">
              <input
                type="number"
                min={1}
                max={512}
                value={numericValue('batchSize', config.hyperParams.batchSize)}
                onChange={(e) => editNumber('batchSize', e.target.value)}
                onBlur={() => commitNumber('batchSize', config.hyperParams.batchSize)}
                className="text-right font-mono text-sm bg-transparent outline-none w-20"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>
            <ConfigField label="学习率">
              <input
                type="number"
                step={0.0001}
                min={0.00001}
                value={numericValue('learningRate', config.hyperParams.learningRate)}
                onChange={(e) => editNumber('learningRate', e.target.value)}
                onBlur={() => commitNumber('learningRate', config.hyperParams.learningRate)}
                className="text-right font-mono text-sm bg-transparent outline-none w-24"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>
            <ConfigField label="早停耐心">
              <input
                type="number"
                min={1}
                value={numericValue('patience', config.patience)}
                onChange={(e) => editNumber('patience', e.target.value)}
                onBlur={() => commitNumber('patience', config.patience)}
                className="text-right font-mono text-sm bg-transparent outline-none w-20"
                style={{ color: 'var(--foreground)' }}
              />
            </ConfigField>
            <ConfigField label="优化器">
              <select
                value={config.hyperParams.optimizer}
                onChange={(e) => handleConfigChange('optimizer', e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm outline-none cursor-pointer transition-colors hover:brightness-95"
                style={{
                  background: 'var(--surface)',
                  color: 'var(--foreground)',
                  border: '1px solid var(--border)',
                }}
              >
                <option value="adamw">ADAMW（后端固定）</option>
              </select>
            </ConfigField>
          </div>
        </GlassPanel>

        {/* System Status */}
        <GlassPanel title="系统状态">
          <div className="p-4 space-y-3">
            {/* 模式指示行 */}
            <div className="flex items-center gap-2">
              <span
                className="inline-block rounded-full"
                style={{
                  width: 8,
                  height: 8,
                  backgroundColor:
                    mode === 'remote'
                      ? remoteConnected
                        ? 'var(--success)'
                        : 'var(--warning)'
                      : liveSysStatus.cudaAvailable
                        ? 'var(--success)'
                        : 'var(--danger)',
                  boxShadow:
                    mode === 'local' && liveSysStatus.cudaAvailable
                      ? '0 0 6px var(--success-glow)'
                      : undefined,
                }}
              />
              <span className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
                {mode === 'remote'
                  ? remoteConnected
                    ? '远程模式 — 已连接到服务器'
                    : '远程模式 — 未连接到服务器'
                  : liveSysStatus.cudaAvailable
                    ? `GPU: ${liveSysStatus.gpuName || 'NVIDIA GPU'}`
                    : '本地模式 — 未检测到 GPU'}
              </span>
            </div>

            {/* GPU 显存（本地模式） */}
            {mode === 'local' && liveSysStatus.cudaAvailable && liveSysStatus.gpuMemory && (
              <div className="text-xs" style={{ color: 'var(--foreground-muted)' }}>
                显存: {liveSysStatus.gpuMemory}
              </div>
            )}

            {/* 远程未连接提示 */}
            {mode === 'remote' && !remoteConnected && (
              <div
                className="p-2 rounded text-xs"
                style={{ background: 'rgba(245,158,11,0.08)', color: 'var(--warning)' }}
              >
                远程模式未连接 — 请先在训练配置中点击「测试连接」
              </div>
            )}

            {/* 系统资源 — 仅本地模式 */}
            {mode === 'local' && (
              <>
                <div className="text-xs space-y-1" style={{ color: 'var(--foreground-dim)' }}>
                  <div>CPU: {liveSysStatus.cpuCores} 核</div>
                  <div>
                    RAM: {liveSysStatus.ramUsed} / {liveSysStatus.ramTotal} (
                    {liveSysStatus.ramPercent}%)
                  </div>
                </div>
                <ProgressBar value={liveSysStatus.ramPercent} variant="accent" />
              </>
            )}
          </div>
        </GlassPanel>
      </div>

      {progress && (
        <GlassPanel title="训练进度">
          <div className="p-4 space-y-3">
            <div className="flex gap-6 text-sm">
              <span style={{ color: 'var(--foreground-muted)' }}>
                Epoch:{' '}
                <span style={{ color: 'var(--foreground)' }}>
                  {progress.epoch} / {config.hyperParams.epochs}
                </span>
              </span>
              <span style={{ color: 'var(--foreground-muted)' }}>
                Step: <span style={{ color: 'var(--foreground)' }}>{progress.step}</span>
              </span>
              <span style={{ color: 'var(--foreground-muted)' }}>
                Loss:{' '}
                <span style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}>
                  {progress.loss.toFixed(4)}
                </span>
              </span>
            </div>
            <ProgressBar
              value={(progress.epoch / config.hyperParams.epochs) * 100}
              variant="accent"
              showGlow
            />
            <LogViewer logs={progress.log} maxHeight="200px" />
          </div>
        </GlassPanel>
      )}

      {finalResult && (
        <GlassPanel title="训练结果">
          <div className="p-4 space-y-2">
            <div className="flex gap-6 text-sm">
              <span style={{ color: 'var(--foreground-muted)' }}>
                Val RMSE:{' '}
                <span style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}>
                  {finalResult.valRmse.toFixed(3)}
                </span>
              </span>
              <span style={{ color: 'var(--foreground-muted)' }}>
                Test RMSE:{' '}
                <span style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}>
                  {finalResult.testRmse.toFixed(3)}
                </span>
              </span>
              <span style={{ color: 'var(--foreground-muted)' }}>
                Test Score:{' '}
                <span style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}>
                  {finalResult.testScore.toFixed(1)}
                </span>
              </span>
            </div>
          </div>
        </GlassPanel>
      )}

      {/* 远程连接配置弹窗 */}
      <RemoteConnectionDialog
        open={remoteDialogOpen}
        onClose={() => setRemoteDialogOpen(false)}
        onConnected={(info) => {
          setRemoteConnected(true);
          setConnectedServer(info);
        }}
      />
    </div>
  );
};

function ConfigField({ label, children }: { label: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span style={{ color: 'var(--foreground-muted)' }}>{label}</span>
      {children}
    </div>
  );
}

export default Training;
