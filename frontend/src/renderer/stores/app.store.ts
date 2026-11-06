import { create } from 'zustand';

interface AppState {
  isDark: boolean;
  isSidebarCollapsed: boolean;
  toggleDark: () => void;
  toggleSidebar: () => void;
  setDark: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  isDark: true,
  isSidebarCollapsed: false,
  toggleDark: () => set((s) => ({ isDark: !s.isDark })),
  toggleSidebar: () => set((s) => ({ isSidebarCollapsed: !s.isSidebarCollapsed })),
  setDark: (v) => set({ isDark: v }),
}));
