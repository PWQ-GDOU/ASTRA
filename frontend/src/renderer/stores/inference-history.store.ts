import { create } from 'zustand';

export interface InferenceHistoryItem {
  id: string;
  taskName: string;
  model: string;
  dataset: string;
  status: 'completed' | 'failed';
  duration: string;
  time: string;
  metrics: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    latency?: number;
    throughput?: number;
    rmse?: number;
    mae?: number;
    score?: number;
  };
  log: string[];
}

interface HistoryFilters {
  model: string;
  status: string;
  dateRange: [string, string] | null;
  search: string;
}

interface InferenceHistoryState {
  items: InferenceHistoryItem[];
  filters: HistoryFilters;
  selectedItem: InferenceHistoryItem | null;
  isDrawerOpen: boolean;
  total: number;
  page: number;
  pageSize: number;
  setItems: (items: InferenceHistoryItem[]) => void;
  setFilters: (filters: Partial<HistoryFilters>) => void;
  selectItem: (item: InferenceHistoryItem | null) => void;
  openDrawer: () => void;
  closeDrawer: () => void;
  deleteItem: (id: string) => void;
  setPage: (page: number) => void;
}

export const useInferenceHistoryStore = create<InferenceHistoryState>((set) => ({
  items: [],
  filters: {
    model: '',
    status: '',
    dateRange: null,
    search: '',
  },
  selectedItem: null,
  isDrawerOpen: false,
  total: 0,
  page: 1,
  pageSize: 20,

  setItems: (items) => set({ items }),
  setFilters: (filters) => set((s) => ({ filters: { ...s.filters, ...filters } })),
  selectItem: (item) => set({ selectedItem: item }),
  openDrawer: () => set({ isDrawerOpen: true }),
  closeDrawer: () => set({ isDrawerOpen: false }),
  deleteItem: (id) =>
    set((s) => ({
      items: s.items.filter((i) => i.id !== id),
      selectedItem: s.selectedItem?.id === id ? null : s.selectedItem,
    })),
  setPage: (page) => set({ page }),
}));
