import React, { useCallback, useRef } from 'react';
import { useModelStore } from '../stores/model.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import EmptyState from '../components/common/EmptyState';
import StatusDot from '../components/common/StatusDot';
import { Box, Download } from 'lucide-react';
import ModelPerformancePanel from '../components/model/ModelPerformancePanel';

const ModelManagement: React.FC = () => {
  const {
    models,
    selectedModel,
    searchQuery,
    sourceFilter,
    selectModel,
    setSearchQuery,
    addModel,
    deleteModel,
  } = useModelStore();
  const dirInputRef = useRef<HTMLInputElement>(null);

  const filtered = models.filter((m) => {
    if (searchQuery && !m.name.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    if (sourceFilter !== 'all' && m.source !== sourceFilter) return false;
    return true;
  });

  const handleImport = useCallback(() => {
    // Try Electron dialog (opens native folder picker)
    if (window.electronAPI?.modelImport) {
      window.electronAPI
        .modelImport()
        .then((result: unknown) => {
          const r = result as {
            success: boolean;
            data?: {
              id: string;
              name: string;
              path: string;
              createdAt: string;
              isDefault: boolean;
            };
            error?: string;
          };
          if (r.success && r.data) {
            addModel({
              ...r.data,
              type: 'ASTRA checkpoint',
              status: 'available' as const,
              source: 'imported' as const,
            });
          } else if (!r.success && r.error && r.error !== 'File selection cancelled') {
            alert(`导入失败: ${r.error}`);
          }
        })
        .catch(() => {
          dirInputRef.current?.click();
        });
    } else {
      dirInputRef.current?.click();
    }
  }, [addModel]);

  const handleFolderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      // Use the first file's parent folder as model name
      const firstFile = files[0];
      const folderName = firstFile.webkitRelativePath?.split('/')[0] || firstFile.name;
      addModel({
        id: Date.now().toString(36),
        name: folderName,
        type: 'unknown',
        status: 'available',
        source: 'imported',
        isDefault: models.length === 0,
        path: folderName,
        createdAt: new Date().toISOString(),
      });
      e.target.value = '';
    },
    [addModel, models.length],
  );

  return (
    <div className="space-y-5">
      {/* Hidden folder input for browser fallback */}
      <input
        ref={dirInputRef}
        type="file"
        // @ts-expect-error webkitdirectory is not in React types
        webkitdirectory=""
        multiple
        onChange={handleFolderChange}
        style={{ display: 'none' }}
      />

      <PageHeader
        title="模型管理"
        description="管理已导入和训练好的模型"
        actions={
          <button
            onClick={handleImport}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
            style={{ background: 'var(--accent)', color: '#fff' }}
          >
            <Download size={14} />
            导入模型
          </button>
        }
      />

      <div className="grid grid-cols-3 gap-5" style={{ height: 'calc(100vh - 180px)' }}>
        <div className="col-span-1 flex flex-col gap-3">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="搜索模型..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="flex-1 px-3 py-1.5 rounded-lg text-sm outline-none"
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                color: 'var(--foreground)',
              }}
            />
          </div>
          <GlassPanel className="flex-1 overflow-y-auto">
            {filtered.length > 0 ? (
              filtered.map((m) => (
                <button
                  type="button"
                  key={m.id}
                  onClick={() => selectModel(m)}
                  className="w-full p-3 text-left cursor-pointer transition-colors border-b"
                  style={{
                    background: selectedModel?.id === m.id ? 'var(--surface-hover)' : 'transparent',
                    borderColor: 'var(--border-light)',
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium" style={{ color: 'var(--foreground)' }}>
                      {m.name}
                      {m.isDefault && (
                        <span
                          className="ml-1.5 text-xs px-1 py-0.5 rounded"
                          style={{ background: 'var(--accent)', color: '#fff', fontSize: '9px' }}
                        >
                          DEFAULT
                        </span>
                      )}
                    </span>
                    <StatusDot
                      status={
                        m.status === 'error'
                          ? 'failed'
                          : m.status === 'training'
                            ? 'running'
                            : 'completed'
                      }
                    />
                  </div>
                  <div className="text-xs mt-1" style={{ color: 'var(--foreground-muted)' }}>
                    {m.source} · {m.type}
                  </div>
                </button>
              ))
            ) : (
              <EmptyState
                icon={Box}
                title="暂无模型"
                description="点击「导入模型」选择本地文件夹"
                action={{ label: '导入模型', onClick: handleImport }}
              />
            )}
          </GlassPanel>
        </div>

        <div className="col-span-2">
          <GlassPanel className="h-full">
            {selectedModel ? (
              <div className="p-5 space-y-4">
                <h2 className="text-lg font-bold" style={{ color: 'var(--foreground)' }}>
                  {selectedModel.name}
                </h2>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    ['类型', selectedModel.type],
                    ['路径', selectedModel.path],
                    ['大小', selectedModel.size || '—'],
                    ['创建时间', selectedModel.createdAt],
                    ['来源', selectedModel.source],
                    ['状态', selectedModel.status],
                  ].map(([l, v]) => (
                    <div key={l} className="text-sm">
                      <span style={{ color: 'var(--foreground-muted)' }}>{l}: </span>
                      <span style={{ color: 'var(--foreground)' }}>{v}</span>
                    </div>
                  ))}
                </div>
                {selectedModel.performance && (
                  <ModelPerformancePanel performance={selectedModel.performance} />
                )}
                {selectedModel.metrics && (
                  <div
                    className="grid grid-cols-2 gap-3 mt-4 p-4 rounded-lg"
                    style={{ background: 'var(--surface)' }}
                  >
                    {[
                      ['准确率', `${((selectedModel.metrics?.accuracy ?? 0) * 100).toFixed(1)}%`],
                      ['参数量', selectedModel.metrics?.paramCount ?? '—'],
                      ['框架', selectedModel.metrics?.framework ?? '—'],
                      ['输入形状', selectedModel.metrics?.inputShape ?? '—'],
                    ].map(([l, v]) => (
                      <div key={l} className="text-sm">
                        <span style={{ color: 'var(--foreground-muted)' }}>{l}: </span>
                        <span
                          style={{ color: 'var(--foreground)', fontFamily: 'var(--font-mono)' }}
                        >
                          {v}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="flex gap-2 pt-2">
                  <button
                    onClick={() => {
                      if (confirm(`确定删除模型 "${selectedModel.name}" 吗？`)) {
                        void (async () => {
                          const result = (await window.electronAPI?.modelDelete?.(
                            selectedModel.id,
                          )) as { success: boolean; error?: string } | undefined;
                          if (!result || result.success) deleteModel(selectedModel.id);
                          else alert(`删除失败: ${result.error || '未知错误'}`);
                        })();
                      }
                    }}
                    className="text-xs px-3 py-1.5 rounded-lg transition-colors"
                    style={{
                      color: 'var(--danger)',
                      border: '1px solid var(--danger)',
                      background: 'transparent',
                    }}
                  >
                    删除模型
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full">
                <span style={{ color: 'var(--foreground-dim)' }}>选择模型查看详情</span>
              </div>
            )}
          </GlassPanel>
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;
