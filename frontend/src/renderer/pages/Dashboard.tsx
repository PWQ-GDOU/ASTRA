import React, { useEffect, useRef } from 'react';
import { useDashboardStore } from '../stores/dashboard.store';
import PageHeader from '../components/common/PageHeader';
import StatCard from '../components/common/StatCard';
import GlassPanel from '../components/common/GlassPanel';
import DataTable from '../components/common/DataTable';
import ProgressBar from '../components/common/ProgressBar';
import { Database, Box, Brain, Activity } from 'lucide-react';

const Dashboard: React.FC = () => {
  const { trainingProgress, recentTasks, lastUpdated, refreshStats, refreshRecentTasks, stats } =
    useDashboardStore();
  const datasetCount = stats.datasetCount;
  const modelCount = stats.modelCount;
  const inferenceCount = stats.inferenceCount;

  useEffect(() => {
    refreshStats();
    refreshRecentTasks();
  }, [refreshStats, refreshRecentTasks]);

  // Track previous values with refs to compute real deltas
  const prevRef = useRef({ ds: 0, mdl: 0, inf: 0 });
  const dsDelta = datasetCount - prevRef.current.ds;
  const mdlDelta = modelCount - prevRef.current.mdl;
  const infDelta = inferenceCount - prevRef.current.inf;

  // Update refs after computing deltas
  useEffect(() => {
    prevRef.current = { ds: datasetCount, mdl: modelCount, inf: inferenceCount };
  }, [datasetCount, modelCount, inferenceCount]);

  const [cudaStatus, setCudaStatus] = React.useState<{
    cudaAvailable: boolean;
    gpuName?: string;
  }>({ cudaAvailable: false });

  React.useEffect(() => {
    const poll = () => {
      if (window.electronAPI?.systemCudaStatus) {
        window.electronAPI
          .systemCudaStatus()
          .then((r: unknown) => {
            const d = (
              r as { success: boolean; data?: { cudaAvailable: boolean; gpuName?: string } }
            ).data;
            if (d) setCudaStatus(d);
          })
          .catch(() => {});
      }
    };
    poll();
    const iv = setInterval(poll, 15000);
    return () => clearInterval(iv);
  }, []);

  // Disable progress bar if no changes
  const liveStats = [
    {
      label: '数据集',
      value: datasetCount,
      change: dsDelta !== 0 ? dsDelta : undefined,
      icon: Database,
    },
    { label: '模型', value: modelCount, change: mdlDelta !== 0 ? mdlDelta : undefined, icon: Box },
    {
      label: '推理次数',
      value: inferenceCount,
      change: infDelta !== 0 ? infDelta : undefined,
      icon: Brain,
    },
    {
      label: cudaStatus.cudaAvailable ? 'GPU' : '系统状态',
      value: cudaStatus.cudaAvailable ? cudaStatus.gpuName || 'CUDA' : '正常',
      change: undefined,
      icon: Activity,
    },
  ];

  return (
    <div className="space-y-5">
      <PageHeader title="工作台" description="系统运行概览与关键指标" />

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-4">
        {liveStats.map((stat) => (
          <StatCard
            key={stat.label}
            label={stat.label}
            value={stat.value}
            change={stat.change}
            icon={stat.icon}
          />
        ))}
      </div>

      {/* Training + Recent Inference */}
      <div className="grid grid-cols-2 gap-5">
        <GlassPanel title="训练任务进度">
          <div className="p-4 space-y-3">
            {trainingProgress.length > 0 ? (
              trainingProgress.map((t) => (
                <div key={t.name} className="flex flex-col gap-1.5">
                  <div className="flex justify-between text-xs">
                    <span style={{ color: 'var(--foreground-muted)' }}>{t.name}</span>
                    <span style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}>
                      {t.progress}%
                    </span>
                  </div>
                  <ProgressBar value={t.progress} variant="accent" showGlow />
                </div>
              ))
            ) : (
              <span className="text-sm" style={{ color: 'var(--foreground-dim)' }}>
                暂无训练任务
              </span>
            )}
          </div>
        </GlassPanel>

        <GlassPanel title="最近推理任务">
          <DataTable
            columns={[
              { key: 'name', label: '任务名' },
              { key: 'model', label: '模型' },
              { key: 'status', label: '状态' },
              { key: 'time', label: '时间' },
            ]}
            data={recentTasks}
          />
        </GlassPanel>
      </div>

      {/* Footer */}
      {lastUpdated && (
        <div className="text-xs text-right" style={{ color: 'var(--foreground-dim)' }}>
          最后更新: {lastUpdated}
        </div>
      )}
    </div>
  );
};

export default Dashboard;
