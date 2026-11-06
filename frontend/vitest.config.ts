import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.{ts,tsx}'],
    globals: true,
    environment: 'jsdom',
    // 完整 App 渲染（含 ECharts 图表）在 jsdom 中较慢，
    // 提高默认测试超时避免并发时 CPU 竞争导致 flaky 超时
    testTimeout: 20000,
    hookTimeout: 20000,
  },
});
