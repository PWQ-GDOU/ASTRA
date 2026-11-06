import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number | null;
  change?: number;
  icon: LucideIcon;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, change, icon: Icon }) => {
  const showChange = change != null && change !== 0;
  const isPositive = change != null && change > 0;

  return (
    <div
      className="glass-card p-5 flex flex-col gap-3 cursor-pointer"
      style={{ transition: 'all var(--transition-fast)' }}
    >
      <div className="flex items-center justify-between">
        <span
          className="text-xs uppercase tracking-wider font-semibold"
          style={{ color: 'var(--foreground-muted)' }}
        >
          {label}
        </span>
        <Icon size={18} style={{ color: 'var(--foreground-muted)' }} />
      </div>

      <div className="flex items-end gap-2">
        <span
          className="font-extrabold tracking-tight"
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '32px',
            color: value == null ? 'var(--foreground-dim)' : 'var(--foreground)',
          }}
        >
          {value == null ? '—' : value}
        </span>
        {showChange && (
          <span
            className="text-xs font-semibold pb-1"
            style={{ color: isPositive ? 'var(--success)' : 'var(--danger)' }}
          >
            {isPositive ? '↑' : '↓'} {Math.abs(change)}%
          </span>
        )}
      </div>

      {/* Mini progress bar — shows change as a visual indicator */}
      {showChange && (
        <div
          className="h-1 rounded-full overflow-hidden"
          style={{ background: 'var(--border-light)' }}
        >
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${Math.min(Math.abs(change), 100)}%`,
              background: isPositive ? 'var(--success)' : 'var(--danger)',
            }}
          />
        </div>
      )}
    </div>
  );
};

export default StatCard;
