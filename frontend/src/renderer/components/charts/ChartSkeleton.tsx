import React from 'react';

interface ChartSkeletonProps {
  height?: number;
  className?: string;
  /** 骨架中包含几个占位条 */
  bars?: number;
}

const ChartSkeleton: React.FC<ChartSkeletonProps> = ({
  height = 260,
  className = '',
  bars = 4,
}) => {
  return (
    <div className={`flex flex-col justify-end gap-2 px-4 pb-4 ${className}`} style={{ height }}>
      <div className="flex items-end gap-3 flex-1">
        {Array.from({ length: bars }).map((_, i) => {
          const h = 40 + Math.sin((i / (bars - 1)) * Math.PI) * 60;
          return (
            <div
              key={i}
              className="flex-1 rounded-sm animate-pulse"
              style={{
                height: `${h}%`,
                background: 'var(--border)',
                animationDelay: `${i * 100}ms`,
                animationDuration: '1.8s',
              }}
            />
          );
        })}
      </div>
      <div className="flex gap-3">
        {Array.from({ length: bars }).map((_, i) => (
          <div
            key={i}
            className="flex-1 h-3 rounded-sm"
            style={{
              background: 'var(--border-light)',
              height: '12px',
            }}
          />
        ))}
      </div>
    </div>
  );
};

export default ChartSkeleton;
