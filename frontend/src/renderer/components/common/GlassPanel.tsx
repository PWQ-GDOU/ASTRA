import React from 'react';

interface GlassPanelProps {
  title?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

const GlassPanel: React.FC<GlassPanelProps> = ({
  title,
  actions,
  children,
  className = '',
  style,
}) => {
  return (
    <div
      className={`glass-panel flex flex-col ${className}`}
      style={{
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border)',
        background: 'var(--surface)',
        backdropFilter: 'blur(10px)',
        overflow: 'hidden',
        ...style,
      }}
    >
      {title && (
        <div
          className="flex items-center justify-between px-5 py-3"
          style={{
            borderBottom: '1px solid var(--border-light)',
          }}
        >
          <span className="text-sm font-semibold" style={{ color: 'var(--foreground)' }}>
            {title}
          </span>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className="flex-1">{children}</div>
    </div>
  );
};

export default GlassPanel;
