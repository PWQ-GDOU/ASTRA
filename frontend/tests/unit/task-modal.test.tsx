/**
 * Task: Modal 通用组件测试
 *
 * 覆盖：
 * - 遮罩层 + 居中卡片渲染
 * - 标题 + 内容渲染
 * - 关闭按钮 / 遮罩点击关闭 / Esc 关闭
 * - onClose 回调触发
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import React from 'react';

afterEach(cleanup);

describe('Modal 通用组件', () => {
  const loadModal = async () =>
    (await import('../../src/renderer/components/common/Modal')).default;

  it('test_modal_renders_children', async () => {
    const Modal = await loadModal();
    render(
      React.createElement(
        Modal,
        { open: true, onClose: vi.fn() },
        React.createElement('div', null, '弹窗内容'),
      ),
    );
    expect(screen.getByText('弹窗内容')).toBeDefined();
  });

  it('test_modal_renders_title', async () => {
    const Modal = await loadModal();
    render(
      React.createElement(
        Modal,
        { open: true, title: '配置服务器', onClose: vi.fn() },
        React.createElement('div'),
      ),
    );
    expect(screen.getByText('配置服务器')).toBeDefined();
  });

  it('test_modal_not_rendered_when_closed', async () => {
    const Modal = await loadModal();
    const { container } = render(
      React.createElement(
        Modal,
        { open: false, onClose: vi.fn() },
        React.createElement('div', null, '内容'),
      ),
    );
    expect(container.querySelector('.modal-overlay')).toBeNull();
  });

  it('test_modal_close_button_calls_onclose', async () => {
    const Modal = await loadModal();
    const onClose = vi.fn();
    render(React.createElement(Modal, { open: true, onClose }, React.createElement('div')));
    const closeBtn = screen.getByTitle('关闭');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });

  it('test_modal_overlay_click_closes', async () => {
    const Modal = await loadModal();
    const onClose = vi.fn();
    const { container } = render(
      React.createElement(Modal, { open: true, onClose }, React.createElement('div')),
    );
    const overlay = container.querySelector('.modal-overlay') as HTMLElement;
    fireEvent.click(overlay);
    expect(onClose).toHaveBeenCalled();
  });

  it('test_modal_esc_closes', async () => {
    const Modal = await loadModal();
    const onClose = vi.fn();
    render(React.createElement(Modal, { open: true, onClose }, React.createElement('div')));
    fireEvent.keyDown(document.body, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('test_modal_has_overlay_class', async () => {
    const Modal = await loadModal();
    const { container } = render(
      React.createElement(Modal, { open: true, onClose: vi.fn() }, React.createElement('div')),
    );
    expect(container.querySelector('.modal-overlay')).not.toBeNull();
  });

  it('test_modal_has_panel_class', async () => {
    const Modal = await loadModal();
    const { container } = render(
      React.createElement(Modal, { open: true, onClose: vi.fn() }, React.createElement('div')),
    );
    expect(container.querySelector('.modal-panel')).not.toBeNull();
  });
});
