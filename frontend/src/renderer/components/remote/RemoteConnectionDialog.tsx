import React, { useState } from 'react';
import Modal from '../common/Modal';
import type { RemoteAuthType } from '../../../shared/types/remote';

interface RemoteConnectionDialogProps {
  open: boolean;
  onClose: () => void;
  onConnected: (info: { host: string; port: number; username: string }) => void;
}

/** 服务器状态：未连接 / 连接中 / 已连接 / 失败 */
export type RemoteStatus = 'idle' | 'connecting' | 'connected' | 'failed';

const RemoteConnectionDialog: React.FC<RemoteConnectionDialogProps> = ({
  open,
  onClose,
  onConnected,
}) => {
  const [host, setHost] = useState('192.168.1.100');
  const [port, setPort] = useState(22);
  const [username, setUsername] = useState('user');
  const [authType, setAuthType] = useState<RemoteAuthType>('password');
  const [password, setPassword] = useState('');
  const [privateKeyPath, setPrivateKeyPath] = useState('~/.ssh/id_rsa');
  const [testing, setTesting] = useState(false);
  const [status, setStatus] = useState<RemoteStatus>('idle');
  const [message, setMessage] = useState('');

  const handleTest = async () => {
    if (!host || !username) {
      setMessage('请填写服务器地址和用户名');
      setStatus('failed');
      return;
    }
    setTesting(true);
    setStatus('connecting');
    setMessage('');
    try {
      const config = {
        host,
        port,
        username,
        authType,
        ...(authType === 'password' ? { password } : { privateKeyPath }),
      };
      const res = (await window.electronAPI?.remoteTestConnection?.(config)) as {
        success: boolean;
        data?: {
          connected: boolean;
          message: string;
          serverInfo?: { host: string; port: number; username: string };
        };
        error?: string;
      };
      if (res?.success && res.data?.connected) {
        setStatus('connected');
        setMessage(res.data.message);
        onConnected({
          host: res.data.serverInfo?.host ?? host,
          port: res.data.serverInfo?.port ?? port,
          username: res.data.serverInfo?.username ?? username,
        });
      } else {
        setStatus('failed');
        setMessage(res?.data?.message || res?.error || '连接失败');
      }
    } catch (e) {
      setStatus('failed');
      setMessage(`连接失败：${String(e)}`);
    } finally {
      setTesting(false);
    }
  };

  return (
    <Modal open={open} title="服务器连接" onClose={onClose} width={460}>
      <div className="space-y-3">
        {/* 认证方式切换 */}
        <div
          className="flex rounded-lg overflow-hidden"
          style={{ border: '1px solid var(--border)' }}
        >
          {(['password', 'sshKey'] as RemoteAuthType[]).map((t) => (
            <button
              key={t}
              onClick={() => setAuthType(t)}
              className="flex-1 py-2 text-xs font-medium transition-colors"
              style={{
                background: authType === t ? 'var(--accent)' : 'transparent',
                color: authType === t ? '#fff' : 'var(--foreground-muted)',
              }}
            >
              {t === 'password' ? '密码' : 'SSH密钥'}
            </button>
          ))}
        </div>

        <Field label="服务器地址">
          <input
            type="text"
            placeholder="192.168.1.100"
            value={host}
            onChange={(e) => setHost(e.target.value)}
            className="text-right font-mono text-sm bg-transparent outline-none w-40"
            style={{ color: 'var(--foreground)' }}
          />
        </Field>
        <Field label="端口">
          <input
            type="number"
            value={port}
            onChange={(e) => setPort(Number(e.target.value))}
            className="text-right font-mono text-sm bg-transparent outline-none w-20"
            style={{ color: 'var(--foreground)' }}
          />
        </Field>
        <Field label="用户名">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="text-right font-mono text-sm bg-transparent outline-none w-32"
            style={{ color: 'var(--foreground)' }}
          />
        </Field>

        {authType === 'password' ? (
          <Field label="密码">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="text-right font-mono text-sm bg-transparent outline-none w-40"
              style={{ color: 'var(--foreground)' }}
            />
          </Field>
        ) : (
          <Field label="私钥路径">
            <input
              type="text"
              placeholder="~/.ssh/id_rsa"
              value={privateKeyPath}
              onChange={(e) => setPrivateKeyPath(e.target.value)}
              className="text-right font-mono text-sm bg-transparent outline-none w-40"
              style={{ color: 'var(--foreground)' }}
            />
          </Field>
        )}

        {/* 状态提示 */}
        {message && (
          <div
            className="p-2 rounded text-xs"
            style={{
              background: status === 'connected' ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)',
              color: status === 'connected' ? 'var(--success)' : 'var(--danger)',
              border: `1px solid ${status === 'connected' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
            }}
          >
            {message}
          </div>
        )}

        <button
          onClick={handleTest}
          disabled={testing}
          className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors disabled:opacity-50"
          style={{ background: 'var(--accent)', color: '#fff' }}
        >
          {testing ? '连接测试中...' : '测试连接'}
        </button>
      </div>
    </Modal>
  );
};

function Field({ label, children }: { label: string; children?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span style={{ color: 'var(--foreground-muted)' }}>{label}</span>
      {children}
    </div>
  );
}

export default RemoteConnectionDialog;
