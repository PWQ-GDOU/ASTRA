import { registerSystemHandlers } from './system.ipc';
import { registerDatasetHandlers } from './dataset.ipc';
import { registerInferenceHandlers } from './inference.ipc';
import { registerTrainingHandlers } from './training.ipc';
import { registerModelHandlers } from './model.ipc';
import { registerStatisticsHandlers } from './statistics.ipc';
import { registerRemoteHandlers } from './remote.ipc';
import type { BrowserWindow } from 'electron';

export function registerAllHandlers(mainWindow: BrowserWindow): void {
  registerSystemHandlers();
  registerDatasetHandlers();
  registerInferenceHandlers(mainWindow);
  registerTrainingHandlers(mainWindow);
  registerModelHandlers();
  registerStatisticsHandlers();
  registerRemoteHandlers();
}
