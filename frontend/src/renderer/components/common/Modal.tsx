import React, { useEffect } from 'react';
import { X } from 'lucide-react';

interface ModalProps {
  open: boolean;
  title?: string;
  onClose: () => void;
  children: React.ReactNode;
  /** 弹窗宽度（默认 520px） */
  width?: number;
}

const Modal: React.FC<ModalProps> = ({ open, title, onClose, children, width = 520 }) => {
  // Esc 关闭
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="modal-overlay fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(4px)' }}
      onClick={(e) => {
        // 点击遮罩（非面板）关闭
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal-panel glass-panel flex flex-col"
        style={{
          width,
          maxWidth: '90vw',
          maxHeight: '85vh',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--bg-elevated)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4 flex-shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <span className="text-sm font-semibold" style={{ color: 'var(--foreground)' }}>
            {title}
          </span>
          <button
            onClick={onClose}
            title="关闭"
            className="flex items-center justify-center w-7 h-7 rounded hover:bg-white/10 transition-colors"
            style={{ color: 'var(--foreground-muted)' }}
          >
            <X size={16} />
          </button>
        </div>
        {/* Body */}
        <div className="px-5 py-4 overflow-y-auto flex-1">{children}</div>
      </div>
    </div>
  );
};

export default Modal;
