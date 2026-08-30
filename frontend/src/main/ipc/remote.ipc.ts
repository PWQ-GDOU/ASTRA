import { ipcMain } from 'electron';
import { IPC_CHANNELS } from '../../shared/types/ipc';
import type { IPCResult } from '../../shared/types/ipc-result';
import type { RemoteConnectionConfig, RemoteConnectionResult } from '../../shared/types/remote';
import { readFileSync } from 'fs';
import { Client } from 'ssh2';

// 当前活动连接（用于断开）
let activeConnection: {
  client: Client;
  config: RemoteConnectionConfig;
} | null = null;

/** 展开 ~ 到用户主目录 */
function expandHome(p: string): string {
  if (p === '~' || p.startsWith('~/')) {
    return p.replace(/^~/, process.env.USERPROFILE || process.env.HOME || '');
  }
  return p;
}

/**
 * 测试 SSH 连接。
 * 支持密码和 SSH 私钥两种认证方式。
 */
function testConnection(config: RemoteConnectionConfig): Promise<RemoteConnectionResult> {
  return new Promise((resolve) => {
    const client = new Client();

    const cleanup = () => {
      if (activeConnection?.client === client) {
        activeConnection = null;
      }
      client.end();
    };

    client.on('ready', () => {
      activeConnection = { client, config };
      resolve({
        connected: true,
        message: `已连接到 ${config.host}:${config.port}（用户 ${config.username}）`,
        serverInfo: { host: config.host, port: config.port, username: config.username },
      });
      cleanup();
    });

    client.on('error', (err: Error) => {
      cleanup();
      resolve({
        connected: false,
        message: `连接失败：${err.message}`,
      });
    });

    const base = {
      host: config.host,
      port: config.port || 22,
      username: config.username,
      readyTimeout: 8000,
      keepaliveInterval: 10000,
    };

    if (config.authType === 'sshKey') {
      let privateKey: Buffer | undefined;
      try {
        privateKey = readFileSync(expandHome(config.privateKeyPath || ''));
      } catch (e) {
        cleanup();
        resolve({
          connected: false,
          message: `无法读取私钥文件：${(e as Error).message}`,
        });
        return;
      }
      client.connect({ ...base, privateKey });
    } else {
      // 密码方式
      client.connect({ ...base, password: config.password || '' });
    }
  });
}

export function registerRemoteHandlers(): void {
  ipcMain.handle(
    IPC_CHANNELS.REMOTE_TEST_CONNECTION,
    async (_e, config: RemoteConnectionConfig): Promise<IPCResult<RemoteConnectionResult>> => {
      try {
        // 参数校验
        if (!config || typeof config !== 'object') {
          return { success: false, error: '远程连接配置无效' };
        }
        if (!config.host || !config.username) {
          return { success: false, error: '请填写服务器地址和用户名' };
        }
        if (config.authType !== 'password' && config.authType !== 'sshKey') {
          return { success: false, error: '无效的连接认证方式' };
        }
        const result = await testConnection(config);
        return { success: true, data: result };
      } catch (e) {
        return { success: false, error: String(e) };
      }
    },
  );

  ipcMain.handle(IPC_CHANNELS.REMOTE_DISCONNECT, (): IPCResult<void> => {
    try {
      if (activeConnection) {
        activeConnection.client.end();
        activeConnection = null;
      }
      return { success: true, data: undefined };
    } catch (e) {
      return { success: false, error: String(e) };
    }
  });
}
