/**
 * IPC 调用客户端封装
 * 渲染进程通过 window.electronAPI 调用 IPC 方法
 */

export async function ipcCall<T>(channel: string, ...args: unknown[]): Promise<T> {
  const electronAPI = (
    window as unknown as {
      electronAPI?: Record<
        string,
        (...a: unknown[]) => Promise<{ success: boolean; data?: T; error?: string }>
      >;
    }
  ).electronAPI;

  if (!electronAPI) {
    throw new Error('window.electronAPI not available — not running in Electron?');
  }

  // Map channel names like "dataset:list" to camelCase method names
  const methodName = channelToMethod(channel);
  const fn = electronAPI[methodName];

  if (typeof fn !== 'function') {
    throw new Error(
      `IPC method "${methodName}" (channel: ${channel}) not exposed on window.electronAPI`,
    );
  }

  try {
    const result = await fn(...args);
    if (result.success) {
      return result.data as T;
    }
    throw new Error(result.error || 'Unknown IPC error');
  } catch (err) {
    if (err instanceof Error) {
      throw err;
    }
    throw new Error(String(err));
  }
}

function channelToMethod(channel: string): string {
  // "dataset:list" → "datasetList"
  // "inference:history-list" → "inferenceHistoryList"
  return channel
    .replace(/:(\w)/g, (_m, c) => c.toUpperCase())
    .replace(/-(\w)/g, (_m, c) => c.toUpperCase());
}
