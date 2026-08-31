import React from 'react';

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, description, actions }) => {
  return (
    <div className="flex items-end justify-between mb-6">
      <div className="flex flex-col gap-1">
        <h1
          className="font-bold tracking-tight"
          style={{ fontSize: '26px', fontWeight: 700, color: 'var(--foreground)' }}
        >
          {title}
        </h1>
        {description && (
          <p className="text-sm" style={{ color: 'var(--foreground-muted)' }}>
            {description}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
};

export default PageHeader;
