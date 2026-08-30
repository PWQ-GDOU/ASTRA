import type { ElectronAPI } from './index';

export type { ElectronAPI };

declare global {
  interface Window {
    electronAPI: ElectronAPI;
  }
}

// Electron 自定义 CSS 属性（-webkit-app-region）的类型增强
declare module 'react' {
  interface CSSProperties {
    WebkitAppRegion?: 'drag' | 'no-drag';
  }
}
