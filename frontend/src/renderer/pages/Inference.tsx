import React, { useCallback, useMemo } from 'react';
import { useInferenceStore } from '../stores/inference.store';
import { useModelStore } from '../stores/model.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import ProgressBar from '../components/common/ProgressBar';
import LogViewer from '../components/common/LogViewer';
import DataTable from '../components/common/DataTable';
import { Play, Square } from 'lucide-react';
import type { InferenceRequest } from '../../shared/types/inference';
import { useDatasetStore } from '../stores/dataset.store';
import InferenceResultAnalysis from '../components/inference/InferenceResultAnalysis';

const Inference: React.FC = () => {
  const {
    activeTask,
    queue,
    result,
    isRunning,
    setIsRunning,
    setActiveTask,
    setResult,
    addToQueue,
  } = useInferenceStore();
  const models = useModelStore((s) => s.models);
  const datasets = useDatasetStore((s) => s.datasets).filter((d) => d.astraCompatible);

  const [selectedDataset, setSelectedDataset] = React.useState('');
  const [selectedModelId, setSelectedModelId] = React.useState('');
  const [batchSize, setBatchSize] = React.useState(128);
  const [device, setDevice] = React.useState<InferenceRequest['device']>('cpu');
  const [applyRulCap, setApplyRulCap] = React.useState(true);
  const [rulCap, setRulCap] = React.useState(125);

  const selectedModel = useMemo(
    () => models.find((m) => m.id === selectedModelId),
    [models, selectedModelId],
  );

  const canStart = selectedDataset && selectedModelId && !isRunning;

  React.useEffect(() => {
    const removeProgress = window.electronAPI?.onInferenceProgress?.((raw) => {
      const event = raw as {
        taskId: string;
        progress: number;
        status: 'pending' | 'running' | 'completed' | 'failed';
        elapsed?: number;
        message?: string;
      };
      useInferenceStore.getState().updateProgress(event);
      useInferenceStore
        .getState()
        .updateQueueItem(event.taskId, { progress: event.progress, status: event.status });
      if (event.message) useInferenceStore.getState().addLog(`[错误] ${event.message}`);
      if (event.status === 'completed' || event.status === 'failed') setIsRunning(false);
    });
    const removeResult = window.electronAPI?.onInferenceResult?.((raw) => {
      setResult(raw as Parameters<typeof setResult>[0]);
      setIsRunning(false);
    });
    return () => {
      removeProgress?.();
      removeResult?.();
    };
  }, [setIsRunning, setResult]);

  const handleInference = useCallback(async () => {
    if (isRunning) {
      if (activeTask?.taskId) await window.electronAPI?.inferenceCancel(activeTask.taskId);
      setIsRunning(false);
      setActiveTask(null);
      return;
    }

    const datasetName = datasets.find((d) => d.id === selectedDataset)?.name || selectedDataset;
    const modelName = selectedModel?.name || selectedModelId;

    const response = (await window.electronAPI?.inferenceRun({
      datasetId: selectedDataset,
      modelId: selectedModelId,
      batchSize,
      device,
      applyRulCap,
      rulCap,
    })) as { success: boolean; data?: string; error?: string } | undefined;
    if (!response?.success || !response.data) {
      setActiveTask({
        taskId: 'error',
        progress: 0,
        status: 'failed',
        log: [response?.error || '推理后端不可用'],
      });
      return;
    }
    const taskId = response.data;
    setIsRunning(true);
    addToQueue({
      id: taskId,
      name: `推理-${datasetName}`,
      model: modelName,
      dataset: datasetName,
      status: 'running' as const,
      progress: 0,
      createdAt: new Date().toISOString(),
    });
    setActiveTask({
      taskId,
      progress: 0,
      status: 'running',
      log: [
        `[${new Date().toLocaleTimeString()}] 开始 ASTRA 推理: model=${modelName}, dataset=${datasetName}`,
      ],
    });
  }, [
    isRunning,
    activeTask,
    selectedModel,
    datasets,
    selectedDataset,
    selectedModelId,
    setIsRunning,
    setActiveTask,
    addToQueue,
    batchSize,
    device,
    applyRulCap,
    rulCap,
  ]);

  return (
    <div className="space-y-5">
      <PageHeader
        title="推理模块"
        description="执行模型推理任务"
        actions={
          <button
            onClick={handleInference}
            disabled={!canStart && !isRunning}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-40"
            style={{
              background: isRunning
                ? 'var(--danger)'
                : canStart
                  ? 'var(--accent)'
                  : 'var(--surface)',
              color: '#fff',
              cursor: !canStart && !isRunning ? 'not-allowed' : 'pointer',
            }}
          >
            {isRunning ? <Square size={14} /> : <Play size={14} />}
            {isRunning ? '取消推理' : '开始推理'}
          </button>
        }
      />

      {/* Dataset + Model selectors */}
      <div className="grid grid-cols-2 gap-5">
        <GlassPanel title="选择数据集">
          <div className="p-3">
            <div className="flex flex-col gap-1">
              {datasets.map((dataset) => (
                <button
                  type="button"
                  key={dataset.id}
                  onClick={() => setSelectedDataset(dataset.id)}
                  className="flex w-full items-center justify-between px-3 py-2 rounded-lg text-left cursor-pointer transition-colors"
                  style={{
                    background:
                      selectedDataset === dataset.id ? 'var(--surface-hover)' : 'transparent',
                    border:
                      selectedDataset === dataset.id
                        ? '1px solid var(--accent)'
                        : '1px solid transparent',
                  }}
                >
                  <span className="text-sm" style={{ color: 'var(--foreground)' }}>
                    {dataset.name}
                  </span>
                  <span
                    className="text-xs px-1.5 py-0.5 rounded"
                    style={{ background: 'var(--surface)', color: 'var(--foreground-muted)' }}
                  >
                    ASTRA
                  </span>
                </button>
              ))}
            </div>
          </div>
        </GlassPanel>

        <GlassPanel title="选择模型">
          <div className="p-3">
            {models.length === 0 ? (
              <span className="text-xs" style={{ color: 'var(--foreground-dim)' }}>
                暂无模型，请先在「模型管理」中导入
              </span>
            ) : (
              <div className="flex flex-col gap-1">
                {models.map((m) => (
                  <button
                    type="button"
                    key={m.id}
                    onClick={() => setSelectedModelId(m.id)}
                    className="flex w-full items-center justify-between px-3 py-2 rounded-lg text-left cursor-pointer transition-colors"
                    style={{
                      background: selectedModelId === m.id ? 'var(--surface-hover)' : 'transparent',
                      border:
                        selectedModelId === m.id
                          ? '1px solid var(--accent)'
                          : '1px solid transparent',
                    }}
                  >
                    <span className="text-sm" style={{ color: 'var(--foreground)' }}>
                      {m.name}
                    </span>
                    <span
                      className="text-xs px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--surface)', color: 'var(--foreground-muted)' }}
                    >
                      {m.status}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </GlassPanel>
      </div>

      <GlassPanel title="推理参数">
        <div className="p-4 grid grid-cols-5 gap-4">
          <label className="text-sm">
            批大小
            <input
              aria-label="批大小"
              type="number"
              min={1}
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              className="w-full mt-1 p-2 rounded bg-transparent border"
            />
          </label>
          <label className="text-sm">
            计算设备
            <select
              aria-label="计算设备"
              value={device}
              onChange={(e) => setDevice(e.target.value as InferenceRequest['device'])}
              className="w-full mt-1 p-2 rounded bg-transparent border"
            >
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA（自动）</option>
              <option value="cuda:0">CUDA:0</option>
              <option value="cuda:1">CUDA:1</option>
              <option value="cuda:2">CUDA:2</option>
            </select>
          </label>
          <label className="text-sm">
            RUL 上限
            <input
              aria-label="RUL 上限"
              type="number"
              min={1}
              disabled={!applyRulCap}
              value={rulCap}
              onChange={(e) => setRulCap(Number(e.target.value))}
              className="w-full mt-1 p-2 rounded bg-transparent border"
            />
          </label>
          <label className="text-sm flex items-center gap-2">
            <input
              type="checkbox"
              checked={applyRulCap}
              onChange={(e) => setApplyRulCap(e.target.checked)}
            />
            应用 RUL 截断
          </label>
        </div>
      </GlassPanel>

      {/* Active Task */}
      {activeTask && (
        <GlassPanel title={`活跃任务: ${activeTask.taskId}`}>
          <div className="p-4 space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-xs" style={{ color: 'var(--foreground-muted)' }}>
                进度:
              </span>
              <div className="flex-1">
                <ProgressBar value={activeTask.progress} showGlow variant="accent" />
              </div>
              <span className="text-xs font-mono" style={{ color: 'var(--foreground)' }}>
                {activeTask.progress}%
              </span>
            </div>
            <LogViewer logs={activeTask.log} maxHeight="200px" />
          </div>
        </GlassPanel>
      )}

      {/* Result */}
      {result && <InferenceResultAnalysis result={result} />}

      {/* Queue */}
      <GlassPanel title="任务队列">
        <DataTable
          columns={[
            { key: 'name', label: '任务名' },
            { key: 'model', label: '模型' },
            { key: 'dataset', label: '数据集' },
            { key: 'status', label: '状态' },
            { key: 'progress', label: '进度', render: (r) => `${r.progress}%` },
          ]}
          data={queue}
        />
      </GlassPanel>
    </div>
  );
};

export default Inference;
