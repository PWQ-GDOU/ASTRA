export type IPCResult<T> = { success: true; data: T } | { success: false; error: string };

export type IPCError = { success: false; error: string };
