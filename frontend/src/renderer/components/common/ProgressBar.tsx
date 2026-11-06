import React, { useEffect, useState, useRef } from 'react';

export type ProgressVariant = 'accent' | 'success' | 'warning' | 'danger';

interface ProgressBarProps {
  value: number;
  showGlow?: boolean;
  variant?: ProgressVariant;
  className?: string;
}

const VARIANT_COLORS: Record<ProgressVariant, string> = {
  accent: 'var(--accent)',
  success: 'var(--success)',
  warning: 'var(--warning)',
  danger: 'var(--danger)',
};

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  showGlow = false,
  variant = 'accent',
  className = '',
}) => {
  const clamped = Math.max(0, Math.min(100, value));
  const color = VARIANT_COLORS[variant];

  // No more "jump to zero" on mount — start at the clamped target,
  // but with the CSS transition so the bar eases in from the very beginning.
  const [displayWidth, setDisplayWidth] = useState(clamped);
  const prevClampedRef = useRef(clamped);

  useEffect(() => {
    if (clamped === prevClampedRef.current) return;
    prevClampedRef.current = clamped;
    setDisplayWidth(clamped);
  }, [clamped]);

  return (
    <div
      className={`w-full rounded-full overflow-hidden ${className}`}
      style={{ height: '6px', background: 'var(--border-light)' }}
    >
      <div
        className="h-full rounded-full"
        style={{
          width: `${displayWidth}%`,
          transition: 'width 800ms cubic-bezier(0.16, 1, 0.3, 1)',
          background:
            variant === 'accent'
              ? `linear-gradient(90deg, var(--accent), var(--accent-secondary))`
              : color,
          boxShadow: showGlow ? `0 0 10px ${color}` : undefined,
          animation: showGlow ? 'pulse-glow 2s ease-in-out infinite' : undefined,
        }}
      />
    </div>
  );
};

export default ProgressBar;
