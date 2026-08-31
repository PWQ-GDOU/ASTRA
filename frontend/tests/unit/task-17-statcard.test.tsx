/**
 * Task 17 测试：StatCard 组件 — 单个统计卡片
 *
 * 按 02-dashboard.md 任务 17 的功能点要求验证。
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { Activity } from 'lucide-react';

const loadStatCard = async () =>
  (await import('../../src/renderer/components/common/StatCard')).default;

describe('Task 17: StatCard 组件', () => {
  it('test_statcard_renders_label', async () => {
    const StatCard = await loadStatCard();
    render(React.createElement(StatCard, { label: '数据集', value: 10, icon: Activity }));
    expect(screen.getByText('数据集')).toBeDefined();
  });

  it('test_statcard_renders_value', async () => {
    const StatCard = await loadStatCard();
    render(React.createElement(StatCard, { label: '数据集', value: 42, icon: Activity }));
    expect(screen.getByText('42')).toBeDefined();
  });

  it('test_statcard_renders_dash_when_value_null', async () => {
    const StatCard = await loadStatCard();
    render(React.createElement(StatCard, { label: '数据集', value: null, icon: Activity }));
    expect(screen.getByText('—')).toBeDefined();
  });

  it('test_statcard_zero_renders_zero_not_dash', async () => {
    const StatCard = await loadStatCard();
    render(React.createElement(StatCard, { label: '数据集', value: 0, icon: Activity }));
    expect(screen.getByText('0')).toBeDefined();
    expect(screen.queryByText('—')).toBeNull();
  });

  it('test_statcard_renders_change_positive', async () => {
    const StatCard = await loadStatCard();
    render(
      React.createElement(StatCard, { label: '数据集', value: 10, change: 5, icon: Activity }),
    );
    expect(screen.getByText(/↑/)).toBeDefined();
    expect(screen.getByText(/5%/)).toBeDefined();
  });

  it('test_statcard_renders_change_negative', async () => {
    const StatCard = await loadStatCard();
    render(
      React.createElement(StatCard, { label: '数据集', value: 10, change: -3, icon: Activity }),
    );
    expect(screen.getByText(/↓/)).toBeDefined();
    expect(screen.getByText(/3%/)).toBeDefined();
  });

  it('test_statcard_no_change_hides_indicator', async () => {
    const StatCard = await loadStatCard();
    render(React.createElement(StatCard, { label: '数据集', value: 10, icon: Activity }));
    expect(screen.queryByText(/↑/)).toBeNull();
    expect(screen.queryByText(/↓/)).toBeNull();
  });

  it('test_statcard_change_zero_hidden', async () => {
    const StatCard = await loadStatCard();
    render(
      React.createElement(StatCard, { label: '数据集', value: 10, change: 0, icon: Activity }),
    );
    expect(screen.queryByText(/↑/)).toBeNull();
  });

  it('test_statcard_has_glass_card_class', async () => {
    const StatCard = await loadStatCard();
    const { container } = render(
      React.createElement(StatCard, { label: '数据集', value: 10, icon: Activity }),
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain('glass-card');
  });

  it('test_statcard_renders_icon', async () => {
    const StatCard = await loadStatCard();
    const { container } = render(
      React.createElement(StatCard, { label: '数据集', value: 10, icon: Activity }),
    );
    // lucide icon renders an svg
    expect(container.querySelector('svg')).not.toBeNull();
  });
});
