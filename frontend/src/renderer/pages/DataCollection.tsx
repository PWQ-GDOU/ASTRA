import React, { useCallback, useRef, useState } from 'react';
import { useDatasetStore } from '../stores/dataset.store';
import PageHeader from '../components/common/PageHeader';
import GlassPanel from '../components/common/GlassPanel';
import DataTable, { Column } from '../components/common/DataTable';
import EmptyState from '../components/common/EmptyState';
import type { Dataset } from '../stores/dataset.store';
import { Database, Upload, Pencil } from 'lucide-react';
import Modal from '../components/common/Modal';
import type { DatasetFolderCandidate } from '../../shared/types/dataset';

const ACCEPT = '.csv,.json,.jsonl,.parquet,.txt,.zip,.npz,.mat';

const DataCollection: React.FC = () => {
  const {
    searchQuery,
    setSearchQuery,
    getFilteredDatasets,
    updateDatasetType,
    renameDataset,
    deleteDataset,
    addDataset,
  } = useDatasetStore();
  const fileInputRef = useRef<HTMLInputElement>(null);
  // 正在重命名的数据集 id + 临时名称
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [candidate, setCandidate] = useState<DatasetFolderCandidate | null>(null);
  const [mapping, setMapping] = useState({ name: '', trainFile: '', testFile: '', rulFile: '' });

  const filtered = getFilteredDatasets();

  const startRename = useCallback((row: Dataset) => {
    setEditingId(row.id);
    setEditName(row.name);
  }, []);

  const commitRename = useCallback(
    async (row: Dataset) => {
      const newName = editName.trim();
      if (newName && newName !== row.name) {
        renameDataset(row.id, newName);
        const result = (await window.electronAPI?.datasetRename?.(row.id, newName)) as
          { success: boolean; error?: string } | undefined;
        if (result && !result.success) {
          renameDataset(row.id, row.name);
          alert(`重命名失败: ${result.error || '未知错误'}`);
        }
      }
      setEditingId(null);
      setEditName('');
    },
    [editName, renameDataset],
  );

  const handleImport = useCallback(() => {
    // Try Electron file dialog first
    if (window.electronAPI?.datasetImport) {
      window.electronAPI
        .datasetImport()
        .then((result: unknown) => {
          const r = result as { success: boolean; data?: DatasetFolderCandidate; error?: string };
          if (r.success && r.data?.rootPath && Array.isArray(r.data.files)) {
            setCandidate(r.data);
            setMapping({
              name: r.data.rootPath.split(/[/\\]/).pop() || '',
              trainFile: '',
              testFile: '',
              rulFile: '',
            });
          } else if (!r.success && r.error && r.error !== 'File selection cancelled') {
            alert(`导入失败: ${r.error}`);
          }
        })
        .catch((e) => alert(`导入失败: ${String(e)}`));
    }
  }, []);

  const saveMapping = useCallback(async () => {
    if (!candidate) return;
    const result = (await window.electronAPI?.datasetSaveMapping({
      ...mapping,
      rootPath: candidate.rootPath,
    })) as { success: boolean; data?: Dataset; error?: string };
    if (!result?.success || !result.data) {
      alert(result?.error || '保存映射失败');
      return;
    }
    addDataset(result.data);
    setCandidate(null);
  }, [candidate, mapping, addDataset]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;
      // 一次多选的所有文件归为一个数据集
      const fileNames = Array.from(files).map((f) => f.name);
      addDataset({
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 4),
        name: fileNames.length === 1 ? fileNames[0] : `${fileNames[0].split('.')[0]} 数据集`,
        samples: 0,
        format: fileNames.length === 1 ? fileNames[0].split('.').pop() || 'unknown' : 'multi',
        type: 'unlabeled',
        createdAt: new Date().toISOString(),
        files: fileNames,
      });
      // Reset so the same file can be selected again
      e.target.value = '';
    },
    [addDataset],
  );

  const handleTypeChange = useCallback(
    async (row: Dataset) => {
      const types: Dataset['type'][] = ['train', 'test', 'val', 'unlabeled'];
      const currentIndex = types.indexOf(row.type);
      const nextType = types[(currentIndex + 1) % types.length];
      updateDatasetType(row.id, nextType);
      const result = (await window.electronAPI?.datasetUpdateType?.(row.id, nextType)) as
        { success: boolean; error?: string } | undefined;
      if (result && !result.success) {
        updateDatasetType(row.id, row.type);
        alert(`更新类型失败: ${result.error || '未知错误'}`);
      }
    },
    [updateDatasetType],
  );

  const columns: Column<Dataset>[] = [
    {
      key: 'name',
      label: '名称',
      render: (row) =>
        editingId === row.id ? (
          <input
            autoFocus
            value={editName}
            placeholder="输入新名称"
            onChange={(e) => setEditName(e.target.value)}
            onBlur={() => commitRename(row)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitRename(row);
              if (e.key === 'Escape') {
                setEditingId(null);
                setEditName('');
              }
            }}
            className="text-sm px-2 py-1 rounded outline-none w-40"
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--accent)',
              color: 'var(--foreground)',
            }}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span style={{ color: 'var(--foreground)' }}>{row.name}</span>
        ),
    },
    { key: 'samples', label: '样本数' },
    { key: 'format', label: '格式' },
    {
      key: 'files',
      label: '文件数',
      render: (row) => (
        <span style={{ color: 'var(--foreground-muted)', fontSize: '12px' }}>
          {row.files && row.files.length > 0 ? `${row.files.length} 个文件` : '—'}
        </span>
      ),
    },
    {
      key: 'type',
      label: '类型',
      align: 'center',
      render: (row) => (
        <span
          onClick={(e) => {
            e.stopPropagation();
            handleTypeChange(row);
          }}
          className="px-2 py-0.5 rounded text-xs font-medium cursor-pointer hover:opacity-80 select-none"
          style={{
            background: 'var(--surface)',
            color: 'var(--accent)',
            border: '1px solid var(--border-light)',
          }}
          title="点击切换类型"
        >
          {row.type}
        </span>
      ),
    },
    { key: 'createdAt', label: '日期' },
    {
      key: 'actions',
      label: '操作',
      align: 'right',
      render: (row) => (
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              startRename(row);
            }}
            className="text-xs px-2 py-1 rounded hover:bg-white/10 transition-colors flex items-center gap-1"
            style={{ color: 'var(--foreground-muted)' }}
            title="重命名数据集"
          >
            <Pencil size={11} />
            重命名
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (
                confirm(
                  `确定删除数据集 "${row.name}" 吗？\n此操作只从软件中移除该记录，不会删除原文件。`,
                )
              ) {
                void (async () => {
                  const result = (await window.electronAPI?.datasetDelete?.(row.id)) as
                    { success: boolean; error?: string } | undefined;
                  if (!result || result.success) deleteDataset(row.id);
                  else alert(`删除失败: ${result.error || '未知错误'}`);
                })();
              }
            }}
            className="text-xs px-2 py-1 rounded hover:bg-red-500/10 transition-colors"
            style={{ color: 'var(--danger)' }}
          >
            删除
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-5">
      {/* Hidden file input for browser fallback */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT}
        multiple
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      <PageHeader
        title="数据采集"
        description="管理训练和推理数据集"
        actions={
          <div className="flex items-center gap-3">
            <input
              type="text"
              placeholder="搜索数据集..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="px-3 py-1.5 rounded-lg text-sm outline-none"
              style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                color: 'var(--foreground)',
                width: '200px',
              }}
            />
            <button
              onClick={handleImport}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
              style={{
                background: 'var(--accent)',
                color: '#fff',
              }}
            >
              <Upload size={14} />
              导入数据集
            </button>
          </div>
        }
      />

      <Modal
        open={Boolean(candidate)}
        title="映射 C-MAPSS 数据文件"
        onClose={() => setCandidate(null)}
      >
        <div className="space-y-4">
          <label className="block text-sm">
            数据集名称
            <input
              aria-label="数据集名称"
              value={mapping.name}
              onChange={(e) => setMapping((m) => ({ ...m, name: e.target.value }))}
              className="w-full mt-1 p-2 rounded bg-transparent border"
            />
          </label>
          {(['trainFile', 'testFile', 'rulFile'] as const).map((key) => (
            <label key={key} className="block text-sm">
              {{ trainFile: '训练文件', testFile: '测试文件', rulFile: 'RUL 标签文件' }[key]}
              <select
                aria-label={
                  { trainFile: '训练文件', testFile: '测试文件', rulFile: 'RUL 标签文件' }[key]
                }
                value={mapping[key]}
                onChange={(e) => setMapping((m) => ({ ...m, [key]: e.target.value }))}
                className="w-full mt-1 p-2 rounded bg-transparent border"
              >
                <option value="">-- 请选择 --</option>
                {candidate?.files?.map((file) => (
                  <option key={file} value={file}>
                    {file.split(/[/\\]/).pop()}
                  </option>
                ))}
              </select>
            </label>
          ))}
          <button
            onClick={saveMapping}
            disabled={!mapping.name || !mapping.trainFile || !mapping.testFile || !mapping.rulFile}
            className="px-4 py-2 rounded text-white disabled:opacity-40"
            style={{ background: 'var(--accent)' }}
          >
            保存数据集映射
          </button>
        </div>
      </Modal>

      <GlassPanel>
        {filtered.length > 0 ? (
          <DataTable
            columns={columns}
            data={filtered}
            renderExpanded={(row) => (
              <div className="flex flex-col gap-1">
                <span
                  className="text-xs font-semibold mb-1"
                  style={{ color: 'var(--foreground-muted)' }}
                >
                  包含的文件（{row.files?.length ?? 0}）:
                </span>
                {row.files && row.files.length > 0 ? (
                  row.files.map((f) => {
                    const fileName = f.split(/[/\\]/).pop() || f;
                    return (
                      <div
                        key={f}
                        className="text-xs font-mono"
                        style={{ color: 'var(--foreground-dim)' }}
                      >
                        {fileName}
                      </div>
                    );
                  })
                ) : (
                  <span className="text-xs" style={{ color: 'var(--foreground-dim)' }}>
                    无文件记录
                  </span>
                )}
              </div>
            )}
          />
        ) : (
          <EmptyState
            icon={Database}
            title="暂无数据集"
            description="选择一个数据文件夹，并一次映射训练、测试和 RUL 标签文件；训练与推理会自动复用"
            action={{ label: '导入数据集', onClick: handleImport }}
          />
        )}
      </GlassPanel>
    </div>
  );
};

export default DataCollection;
