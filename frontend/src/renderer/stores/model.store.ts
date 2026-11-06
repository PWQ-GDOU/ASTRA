import { create } from 'zustand';
import type { ModelPerformance } from '../../shared/types/model';

export type ModelStatus = 'available' | 'training' | 'error';
export type ModelSource = 'imported' | 'trained' | 'pretrained';

export interface Model {
  id: string;
  name: string;
  type: string;
  status: ModelStatus;
  source: ModelSource;
  isDefault: boolean;
  path: string;
  size?: string;
  createdAt: string;
  metrics?: ModelMetrics;
  performance?: ModelPerformance;
}

export type { ModelPerformance };

export interface ModelMetrics {
  accuracy: number;
  paramCount: string;
  framework: string;
  inputShape: string;
}

interface ModelState {
  models: Model[];
  selectedModel: Model | null;
  searchQuery: string;
  sourceFilter: ModelSource | 'all';
  setModels: (models: Model[]) => void;
  selectModel: (m: Model | null) => void;
  setSearchQuery: (q: string) => void;
  setSourceFilter: (f: ModelSource | 'all') => void;
  addModel: (m: Model) => void;
  deleteModel: (id: string) => void;
  setDefaultModel: (id: string) => void;
  getFilteredModels: () => Model[];
}

export const useModelStore = create<ModelState>((set, get) => ({
  models: [],
  selectedModel: null,
  searchQuery: '',
  sourceFilter: 'all',

  setModels: (models) =>
    set((state) => ({
      models,
      selectedModel: state.selectedModel
        ? (models.find((model) => model.id === state.selectedModel?.id) ?? null)
        : null,
    })),
  selectModel: (m) => set({ selectedModel: m }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setSourceFilter: (f) => set({ sourceFilter: f }),
  addModel: (m) => set((s) => ({ models: [...s.models, m] })),
  deleteModel: (id) =>
    set((s) => ({
      models: s.models.filter((m) => m.id !== id),
      selectedModel: s.selectedModel?.id === id ? null : s.selectedModel,
    })),
  setDefaultModel: (id) =>
    set((s) => ({
      models: s.models.map((m) => ({ ...m, isDefault: m.id === id })),
    })),

  getFilteredModels: () => {
    const { models, searchQuery, sourceFilter } = get();
    let filtered = models;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter((m) => m.name.toLowerCase().includes(q));
    }
    if (sourceFilter !== 'all') {
      filtered = filtered.filter((m) => m.source === sourceFilter);
    }
    return filtered;
  },
}));
