import React, { useState } from 'react';

export interface Column<T> {
  key: string;
  label: string;
  render?: (row: T) => React.ReactNode;
  align?: 'left' | 'right' | 'center';
  width?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  onRowClick?: (row: T) => void;
  emptyState?: React.ReactNode;
  /** 展开行内容：行点击后在其下方渲染（用于数据集显示文件列表等） */
  renderExpanded?: (row: T) => React.ReactNode;
}

function DataTable<T extends { id?: string }>({
  columns,
  data,
  onRowClick,
  emptyState,
  renderExpanded,
}: DataTableProps<T>) {
  // 展开行的 key（id），点击切换
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        {emptyState || (
          <span style={{ color: 'var(--foreground-muted)', fontSize: '14px' }}>暂无数据</span>
        )}
      </div>
    );
  }

  const handleRowClick = (row: T) => {
    onRowClick?.(row);
    if (renderExpanded) {
      const key = row.id ?? '';
      setExpandedKey((prev) => (prev === key ? null : key));
    }
  };
  const rowsAreInteractive = Boolean(onRowClick || renderExpanded);

  return (
    <table className="w-full border-collapse">
      <thead>
        <tr
          style={{
            borderBottom: '1px solid var(--border)',
          }}
        >
          {renderExpanded && <th className="py-2 px-2" style={{ width: '30px' }} />}
          {columns.map((col) => (
            <th
              key={col.key}
              className="py-2 px-4 text-left"
              style={{
                fontSize: '10px',
                fontWeight: 700,
                textTransform: 'uppercase',
                color: 'var(--foreground-muted)',
                letterSpacing: '0.05em',
                textAlign: col.align || 'left',
                width: col.width,
              }}
            >
              {col.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, idx) => {
          const key = row.id ?? '';
          const isExpanded = renderExpanded ? expandedKey === key : false;
          return (
            <React.Fragment key={row.id || idx}>
              <tr
                onClick={rowsAreInteractive ? () => handleRowClick(row) : undefined}
                className={`transition-colors ${rowsAreInteractive ? 'cursor-pointer' : ''}`}
                style={{
                  borderBottom: '1px solid var(--border-light)',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'var(--surface-hover)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = 'transparent';
                }}
              >
                {renderExpanded && (
                  <td
                    className="py-2.5 px-2 text-center"
                    style={{ color: 'var(--foreground-dim)' }}
                  >
                    {isExpanded ? '▾' : '▸'}
                  </td>
                )}
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className="py-2.5 px-4"
                    style={{
                      fontSize: '13px',
                      color: 'var(--foreground)',
                      textAlign: col.align || 'left',
                    }}
                  >
                    {col.render
                      ? col.render(row)
                      : String((row as Record<string, unknown>)[col.key] ?? '')}
                  </td>
                ))}
              </tr>
              {isExpanded && renderExpanded && (
                <tr
                  style={{
                    background: 'var(--surface)',
                    borderBottom: '1px solid var(--border-light)',
                  }}
                >
                  <td colSpan={columns.length + 1} className="px-6 py-3">
                    {renderExpanded(row)}
                  </td>
                </tr>
              )}
            </React.Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

export default DataTable;
