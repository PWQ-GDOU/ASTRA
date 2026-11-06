/** 远程连接方式 */
export type RemoteAuthType = 'password' | 'sshKey';

/** 远程服务器连接配置 */
export interface RemoteConnectionConfig {
  host: string;
  port: number;
  username: string;
  /** 认证方式 */
  authType: RemoteAuthType;
  /** 密码方式：密码 */
  password?: string;
  /** SSH密钥方式：私钥路径 */
  privateKeyPath?: string;
}

/** 远程连接测试结果 */
export interface RemoteConnectionResult {
  connected: boolean;
  message: string;
  /** 服务器信息（连接成功时） */
  serverInfo?: {
    host: string;
    port: number;
    username: string;
  };
}
