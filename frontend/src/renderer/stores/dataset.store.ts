import { create } from 'zustand';

export type DatasetType = 'train' | 'test' | 'val' | 'unlabeled';

export interface Dataset {
  id: string;
  name: string;
  samples: number;
  format: string;
  type: DatasetType;
  createdAt: string;
  /** 该数据集包含的文件路径列表（多选导入时） */
  files?: string[];
  rootPath?: string;
  trainFile?: string;
  testFile?: string;
  rulFile?: string;
  astraCompatible?: boolean;
}

interface DatasetState {
  datasets: Dataset[];
  searchQuery: string;
  selectedDataset: Dataset | null;
  setDatasets: (datasets: Dataset[]) => void;
  setSearchQuery: (q: string) => void;
  selectDataset: (ds: Dataset | null) => void;
  addDataset: (ds: Dataset) => void;
  updateDatasetType: (id: string, type: DatasetType) => void;
  renameDataset: (id: string, name: string) => void;
  deleteDataset: (id: string) => void;
  getFilteredDatasets: () => Dataset[];
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  datasets: [],
  searchQuery: '',
  selectedDataset: null,

  setDatasets: (datasets) => set({ datasets }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  selectDataset: (ds) => set({ selectedDataset: ds }),
  addDataset: (ds) => set((s) => ({ datasets: [...s.datasets, ds] })),
  updateDatasetType: (id, type) =>
    set((s) => ({
      datasets: s.datasets.map((d) => (d.id === id ? { ...d, type } : d)),
    })),
  renameDataset: (id, name) =>
    set((s) => ({
      datasets: s.datasets.map((d) => (d.id === id ? { ...d, name } : d)),
    })),
  deleteDataset: (id) =>
    set((s) => ({
      datasets: s.datasets.filter((d) => d.id !== id),
      selectedDataset: s.selectedDataset?.id === id ? null : s.selectedDataset,
    })),

  getFilteredDatasets: () => {
    const { datasets, searchQuery } = get();
    if (!searchQuery) return datasets;
    const q = searchQuery.toLowerCase();
    return datasets.filter((d) => d.name.toLowerCase().includes(q));
  },
}));
