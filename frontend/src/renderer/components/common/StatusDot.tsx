import React from 'react';

export type StatusVariant = 'completed' | 'running' | 'failed' | 'pending';

interface StatusDotProps {
  status: StatusVariant;
  label?: string;
}

const STATUS_STYLES: Record<StatusVariant, { bg: string; glow?: string; animate?: string }> = {
  completed: {
    bg: 'var(--success)',
    glow: '0 0 6px var(--success-glow)',
  },
  running: {
    bg: 'var(--accent)',
    glow: '0 0 8px var(--accent-glow)',
    animate: 'pulse-glow 2s ease-in-out infinite',
  },
  failed: {
    bg: 'var(--danger)',
    glow: '0 0 6px var(--danger-glow)',
  },
  pending: {
    bg: 'var(--foreground-muted)',
  },
};

const StatusDot: React.FC<StatusDotProps> = ({ status, label }) => {
  const style = STATUS_STYLES[status];

  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block rounded-full"
        style={{
          width: '8px',
          height: '8px',
          backgroundColor: style.bg,
          boxShadow: style.glow,
          animation: style.animate,
        }}
      />
      {label && (
        <span className="text-xs" style={{ color: 'var(--foreground-muted)' }}>
          {label}
        </span>
      )}
    </span>
  );
};

export default StatusDot;
