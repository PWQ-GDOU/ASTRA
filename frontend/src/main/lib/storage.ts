import { readFileSync, existsSync, writeFileSync } from 'fs';
import { join } from 'path';
import { app } from 'electron';

const userDataPath = app.getPath('userData');

export const DATASETS_FILE = 'datasets.json';
export const MODELS_FILE = 'models.json';
export const INFERENCE_HISTORY_FILE = 'inference-history.json';
export const TRAINING_CONFIG_FILE = 'training-configs.json';
export const STATS_CACHE_FILE = 'stats-cache.json';

export function readJSON<T>(filename: string): T[] | null {
  const filePath = join(userDataPath, filename);
  if (!existsSync(filePath)) return null;
  try {
    const raw = readFileSync(filePath, 'utf-8');
    return JSON.parse(raw) as T[];
  } catch {
    return null;
  }
}

export function writeJSON<T>(filename: string, data: T[]): void {
  const filePath = join(userDataPath, filename);
  writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

export function appendJSON<T>(filename: string, item: T): void {
  const existing = readJSON<T>(filename) || [];
  existing.push(item);
  writeJSON(filename, existing);
}

export function updateJSON<T extends { id: string }>(
  filename: string,
  id: string,
  updater: (item: T) => T,
): void {
  const existing = readJSON<T>(filename) || [];
  const idx = existing.findIndex((e) => e.id === id);
  if (idx !== -1) {
    existing[idx] = updater(existing[idx]);
    writeJSON(filename, existing);
  }
}

export function deleteJSON(filename: string, id: string): void {
  const existing = readJSON<{ id: string }>(filename) || [];
  writeJSON(
    filename,
    existing.filter((e) => e.id !== id),
  );
}
