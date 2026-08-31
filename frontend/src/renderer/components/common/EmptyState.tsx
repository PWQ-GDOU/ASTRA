import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  action?: { label: string; onClick: () => void };
}

const EmptyState: React.FC<EmptyStateProps> = ({ icon: Icon, title, description, action }) => {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <Icon size={48} style={{ color: 'var(--foreground-muted)' }} />
      <span className="text-lg font-semibold" style={{ color: 'var(--foreground)' }}>
        {title}
      </span>
      {description && (
        <span className="text-sm text-center" style={{ color: 'var(--foreground-muted)' }}>
          {description}
        </span>
      )}
      {action && (
        <button
          onClick={action.onClick}
          className="mt-2 px-4 py-2 rounded-lg border transition-colors text-sm"
          style={{
            borderColor: 'var(--border)',
            color: 'var(--foreground)',
            background: 'var(--surface)',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.background = 'var(--surface)';
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};

export default EmptyState;
