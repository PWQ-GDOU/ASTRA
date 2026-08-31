import React, { useEffect } from 'react';
import { Routes, Route, Navigate, HashRouter } from 'react-router-dom';
import AppShell from './components/layout/AppShell';
import Dashboard from './pages/Dashboard';
import DataCollection from './pages/DataCollection';
import Inference from './pages/Inference';
import InferenceHistory from './pages/InferenceHistory';
import Training from './pages/Training';
import ModelManagement from './pages/ModelManagement';
import Statistics from './pages/Statistics';
import { useDatasetStore } from './stores/dataset.store';
import { useModelStore, type Model } from './stores/model.store';
import type { Dataset } from './stores/dataset.store';

const App: React.FC = () => {
  // 应用启动时从 IPC 加载数据集和模型，供各页面选择
  useEffect(() => {
    const removeModelStatus = window.electronAPI?.onModelStatus?.((raw) => {
      const model = { ...(raw as Model), type: 'ASTRA checkpoint' };
      const current = useModelStore.getState().models;
      useModelStore
        .getState()
        .setModels([...current.filter((item) => item.id !== model.id), model]);
    });
    const loadData = async () => {
      try {
        if (window.electronAPI?.datasetList) {
          const res = (await window.electronAPI.datasetList()) as {
            success: boolean;
            data?: Dataset[];
          };
          if (res.success && res.data) {
            useDatasetStore.getState().setDatasets(res.data);
          }
        }
        if (window.electronAPI?.modelList) {
          const res = (await window.electronAPI.modelList()) as {
            success: boolean;
            data?: Model[];
          };
          if (res.success && res.data) {
            useModelStore.getState().setModels(res.data);
          }
        }
      } catch {
        // 非 Electron 环境或 IPC 不可用时静默跳过
      }
    };
    void loadData();
    return () => removeModelStatus?.();
  }, []);

  return (
    <HashRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/data" element={<DataCollection />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/inference-history" element={<InferenceHistory />} />
          <Route path="/training" element={<Training />} />
          <Route path="/models" element={<ModelManagement />} />
          <Route path="/statistics" element={<Statistics />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </HashRouter>
  );
};

export default App;
