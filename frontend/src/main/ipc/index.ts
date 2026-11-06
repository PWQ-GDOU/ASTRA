import { registerSystemHandlers } from './system.ipc';
import { registerDatasetHandlers } from './dataset.ipc';
import { registerInferenceHandlers } from './inference.ipc';
export { registerAllHandlers } from './register-all';

// Re-export for external use
export { registerSystemHandlers, registerDatasetHandlers, registerInferenceHandlers };
