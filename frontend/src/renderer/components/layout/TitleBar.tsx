import React, { useEffect, useState } from 'react';

const TitleBar: React.FC = () => {
  const [cudaStatus, setCudaStatus] = useState<{
    cudaAvailable: boolean;
    gpuName?: string;
    gpuMemory?: string;
    utilization?: number;
  }>({ cudaAvailable: false });

  useEffect(() => {
    // Poll CUDA status (first poll after 1s, then every 15s)
    const poll = () => {
      if (window.electronAPI?.systemCudaStatus) {
        window.electronAPI
          .systemCudaStatus()
          .then((result: unknown) => {
            const r = result as {
              success: boolean;
              data?: {
                cudaAvailable: boolean;
                gpuName?: string;
                gpuMemory?: string;
                utilization?: number;
              };
            };
            if (r.success && r.data) {
              setCudaStatus(r.data);
            }
          })
          .catch(() => {});
      }
    };

    // First poll sooner so the indicator updates quickly
    const initialTimer = setTimeout(() => {
      poll();
    }, 500);

    const interval = setInterval(poll, 5000);

    // Listen for push updates
    let unsubscribe: (() => void) | undefined;
    if (window.electronAPI?.onCudaStatus) {
      unsubscribe = window.electronAPI.onCudaStatus((status: unknown) => {
        const s = status as {
          cudaAvailable: boolean;
          gpuName?: string;
          gpuMemory?: string;
          utilization?: number;
        };
        if (s) setCudaStatus(s);
      });
    }

    return () => {
      clearTimeout(initialTimer);
      clearInterval(interval);
      unsubscribe?.();
    };
  }, []);

  const handleMinimize = () => window.electronAPI?.minimizeWindow();
  const handleMaximize = () => window.electronAPI?.maximizeWindow();
  const handleClose = () => window.electronAPI?.closeWindow();

  return (
    <div
      className="drag-region glass flex items-center justify-between px-4 select-none flex-shrink-0"
      style={{
        height: '44px',
        minHeight: '44px',
        WebkitAppRegion: 'drag',
        borderBottom: '1px solid var(--border)',
        background: 'rgba(255,255,255,0.75)',
        backdropFilter: 'blur(20px)',
      }}
    >
      <div className="flex items-center gap-2">
        <span
          className="text-xs font-semibold tracking-wide"
          style={{ color: 'var(--foreground-muted)', fontSize: '12px', fontWeight: 600 }}
        >
          Spacecraft 推理训练平台
        </span>
      </div>

      <div className="flex items-center gap-3">
        {/* CUDA status indicator */}
        <div
          className="flex items-center gap-1.5"
          title={
            cudaStatus.cudaAvailable
              ? [
                  cudaStatus.gpuName,
                  cudaStatus.gpuMemory ? `显存: ${cudaStatus.gpuMemory}` : null,
                  cudaStatus.utilization != null ? `使用率: ${cudaStatus.utilization}%` : null,
                ]
                  .filter(Boolean)
                  .join('\n')
              : 'CUDA 不可用'
          }
        >
          <span
            className="inline-block rounded-full"
            style={{
              width: '8px',
              height: '8px',
              backgroundColor: cudaStatus.cudaAvailable
                ? 'var(--success)'
                : 'var(--foreground-dim)',
              boxShadow: cudaStatus.cudaAvailable ? '0 0 6px var(--success-glow)' : undefined,
            }}
          />
          <span className="text-xs" style={{ color: 'var(--foreground-muted)', fontSize: '11px' }}>
            {cudaStatus.cudaAvailable ? cudaStatus.gpuName || 'CUDA' : 'CUDA 不可用'}
            {cudaStatus.utilization != null && (
              <span style={{ color: 'var(--foreground-dim)' }}> {cudaStatus.utilization}%</span>
            )}
          </span>
        </div>

        {/* System indicator — reflects training/inference activity */}
        {(() => {
          const w = window as Window & {
            __trainingActive?: boolean;
            __inferenceActive?: boolean;
          };
          const isActive = window.electronAPI && (w.__trainingActive || w.__inferenceActive);
          return (
            <div className="flex items-center gap-1.5" title={isActive ? '系统忙碌中' : '系统空闲'}>
              <span
                className="inline-block rounded-full"
                style={{
                  width: '8px',
                  height: '8px',
                  backgroundColor: isActive ? 'var(--success)' : 'var(--foreground-dim)',
                  boxShadow: isActive ? '0 0 6px var(--success-glow)' : undefined,
                  animation: isActive ? 'pulse-glow 2s ease-in-out infinite' : undefined,
                }}
              />
              <span
                className="text-xs"
                style={{ color: 'var(--foreground-muted)', fontSize: '11px' }}
              >
                {isActive ? 'Active' : 'Idle'}
              </span>
            </div>
          );
        })()}

        <div
          className="no-drag flex items-center gap-1 ml-2"
          style={{ WebkitAppRegion: 'no-drag' }}
        >
          <button
            onClick={handleMinimize}
            className="flex items-center justify-center w-8 h-8 rounded hover:bg-white/10 transition-colors"
            title="最小化"
            style={{ WebkitAppRegion: 'no-drag' }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <rect x="1" y="5.5" width="10" height="1" fill="var(--foreground-muted)" />
            </svg>
          </button>
          <button
            onClick={handleMaximize}
            className="flex items-center justify-center w-8 h-8 rounded hover:bg-white/10 transition-colors"
            title="最大化"
            style={{ WebkitAppRegion: 'no-drag' }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <rect
                x="1.5"
                y="1.5"
                width="9"
                height="9"
                rx="1"
                stroke="var(--foreground-muted)"
                strokeWidth="1"
              />
            </svg>
          </button>
          <button
            onClick={handleClose}
            className="flex items-center justify-center w-8 h-8 rounded hover:bg-red-500/30 hover:text-white transition-colors"
            title="关闭"
            style={{ WebkitAppRegion: 'no-drag' }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M1 1L11 11M11 1L1 11" stroke="var(--foreground-muted)" strokeWidth="1.5" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
};

export default TitleBar;
