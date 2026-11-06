import React, { useEffect, useMemo } from 'react';
import { useInferenceHistoryStore } from '../stores/inference-history.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import DataTable from '../components/common/DataTable';
import EmptyState from '../components/common/EmptyState';
import StatusDot from '../components/common/StatusDot';
import { History } from 'lucide-react';

const InferenceHistory: React.FC = () => {
  const { items, filters, setItems, setFilters, selectItem, openDrawer } =
    useInferenceHistoryStore();

  useEffect(() => {
    if (items.length > 0 || !window.electronAPI?.inferenceHistoryList) return;
    let active = true;
    window.electronAPI
      .inferenceHistoryList()
      .then((raw) => {
        const result = raw as { success: boolean; data?: typeof items };
        if (active && result.success && result.data) setItems(result.data);
      })
      .catch(() => {
        // 非 Electron 环境或历史文件不可读时保留空状态。
      });
    return () => {
      active = false;
    };
  }, [items.length, setItems]);

  const filteredItems = useMemo(() => {
    const search = filters.search.trim().toLocaleLowerCase();
    return items.filter((item) => {
      if (filters.status && item.status !== filters.status) return false;
      if (filters.model && item.model !== filters.model) return false;
      if (
        search &&
        ![item.taskName, item.model, item.dataset].some((value) =>
          value.toLocaleLowerCase().includes(search),
        )
      )
        return false;
      if (filters.dateRange) {
        const [from, to] = filters.dateRange;
        if ((from && item.time < from) || (to && item.time > to)) return false;
      }
      return true;
    });
  }, [filters, items]);

  return (
    <div className="space-y-5">
      <PageHeader title="推理历史" description="查看历史推理任务记录" />

      {/* Filters */}
      <div className="flex items-center gap-3 flex-wrap">
        <input
          type="text"
          placeholder="搜索任务..."
          value={filters.search}
          onChange={(e) => setFilters({ search: e.target.value })}
          className="px-3 py-1.5 rounded-lg text-sm outline-none"
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            color: 'var(--foreground)',
            width: '200px',
          }}
        />
        <select
          value={filters.status}
          onChange={(e) => setFilters({ status: e.target.value })}
          className="px-3 py-1.5 rounded-lg text-sm outline-none"
          style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            color: 'var(--foreground)',
          }}
        >
          <option value="">全部状态</option>
          <option value="completed">已完成</option>
          <option value="failed">失败</option>
        </select>
      </div>

      <GlassPanel>
        {filteredItems.length > 0 ? (
          <DataTable
            columns={[
              { key: 'taskName', label: '任务名' },
              { key: 'model', label: '模型' },
              { key: 'dataset', label: '数据集' },
              {
                key: 'status',
                label: '状态',
                render: (row) => (
                  <StatusDot
                    status={row.status as 'completed' | 'failed'}
                    label={row.status === 'completed' ? '完成' : '失败'}
                  />
                ),
              },
              { key: 'duration', label: '耗时' },
              { key: 'time', label: '时间' },
            ]}
            data={filteredItems}
            onRowClick={(row) => {
              selectItem(row);
              openDrawer();
            }}
          />
        ) : (
          <EmptyState
            icon={History}
            title="暂无推理记录"
            description="执行推理任务后，记录会显示在这里"
          />
        )}
      </GlassPanel>
    </div>
  );
};

export default InferenceHistory;
