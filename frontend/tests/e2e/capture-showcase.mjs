import { _electron as electron } from 'playwright';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'fs';
import { join, resolve } from 'path';
import { tmpdir } from 'os';
import { fileURLToPath } from 'url';

const frontendDir = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const outputDir = join(frontendDir, 'showcase');
const fixedNow = '2026-08-31T10:30:00.000Z';

const mockDatasets = [
  {
    id: 'fd001-showcase',
    name: 'C-MAPSS FD001',
    samples: 100,
    format: 'cmapss',
    type: 'test',
    createdAt: fixedNow,
    astraCompatible: true,
    trainFile: 'D:/datasets/FD001/train_FD001.txt',
    testFile: 'D:/datasets/FD001/test_FD001.txt',
    rulFile: 'D:/datasets/FD001/RUL_FD001.txt',
  },
  {
    id: 'fd002-showcase',
    name: 'C-MAPSS FD002',
    samples: 259,
    format: 'cmapss',
    type: 'train',
    createdAt: '2026-08-30T09:00:00.000Z',
    astraCompatible: true,
    trainFile: 'D:/datasets/FD002/train_FD002.txt',
    testFile: 'D:/datasets/FD002/test_FD002.txt',
    rulFile: 'D:/datasets/FD002/RUL_FD002.txt',
  },
  {
    id: 'fd003-showcase',
    name: 'C-MAPSS FD003',
    samples: 100,
    format: 'cmapss',
    type: 'validation',
    createdAt: '2026-08-29T08:00:00.000Z',
    astraCompatible: true,
    trainFile: 'D:/datasets/FD003/train_FD003.txt',
    testFile: 'D:/datasets/FD003/test_FD003.txt',
    rulFile: 'D:/datasets/FD003/RUL_FD003.txt',
  },
];

const mockModels = [
  {
    id: 'multiscale-showcase',
    name: 'ASTRA-MultiScale-TCN',
    status: 'available',
    source: 'trained',
    isDefault: true,
    path: 'D:/models/astra_multiscale_fd001.pt',
    size: '2.1 MB',
    createdAt: '2026-08-31T09:15:00.000Z',
    performance: {
      taskId: 'inference-showcase-1',
      datasetId: 'fd001-showcase',
      datasetName: 'C-MAPSS FD001',
      rmse: 13.537,
      mae: 10.284,
      score: 318.42,
      sampleCount: 100,
      seconds: 0.742,
      evaluatedAt: fixedNow,
    },
  },
  {
    id: 'tcn-showcase',
    name: 'TCN-Baseline',
    status: 'available',
    source: 'pretrained',
    isDefault: false,
    path: 'D:/models/tcn_baseline.pt',
    size: '1.6 MB',
    createdAt: '2026-08-28T11:20:00.000Z',
  },
  {
    id: 'lstm-showcase',
    name: 'BiLSTM-RUL',
    status: 'training',
    source: 'trained',
    isDefault: false,
    path: 'D:/models/bilstm_rul.pt',
    size: '3.4 MB',
    createdAt: '2026-08-27T16:45:00.000Z',
  },
];

const mockHistory = [
  {
    id: 'history-showcase-1',
    taskName: 'FD001 评估任务',
    model: 'ASTRA-MultiScale-TCN',
    dataset: 'C-MAPSS FD001',
    status: 'completed',
    duration: '0.74s',
    time: fixedNow,
    metrics: { rmse: 13.537, mae: 10.284, score: 318.42, latency: 0.742 },
    log: ['ASTRA 推理完成'],
  },
  {
    id: 'history-showcase-2',
    taskName: 'FD002 验证任务',
    model: 'TCN-Baseline',
    dataset: 'C-MAPSS FD002',
    status: 'completed',
    duration: '1.21s',
    time: '2026-08-30T15:20:00.000Z',
    metrics: { rmse: 17.862, mae: 13.419, score: 512.86, latency: 1.21 },
    log: ['ASTRA 推理完成'],
  },
  {
    id: 'history-showcase-3',
    taskName: 'FD003 回归检查',
    model: 'ASTRA-MultiScale-TCN',
    dataset: 'C-MAPSS FD003',
    status: 'completed',
    duration: '0.88s',
    time: '2026-08-29T13:10:00.000Z',
    metrics: { rmse: 14.926, mae: 11.307, score: 366.18, latency: 0.88 },
    log: ['ASTRA 推理完成'],
  },
];

function writeProfile(profileDir, withMockData) {
  if (!withMockData) return;
  writeFileSync(join(profileDir, 'datasets.json'), JSON.stringify(mockDatasets, null, 2));
  writeFileSync(join(profileDir, 'models.json'), JSON.stringify(mockModels, null, 2));
  writeFileSync(join(profileDir, 'inference-history.json'), JSON.stringify(mockHistory, null, 2));
}

async function openPage(page, hash, heading) {
  await page.evaluate((nextHash) => {
    window.location.hash = nextHash;
  }, hash);
  await page.getByRole('heading', { name: heading, exact: true }).waitFor();
  await page.waitForTimeout(250);
}

async function captureSet({ name, withMockData }) {
  const profileDir = mkdtempSync(join(tmpdir(), `spacecraft-showcase-${name}-`));
  const targetDir = join(outputDir, name);
  mkdirSync(targetDir, { recursive: true });
  writeProfile(profileDir, withMockData);

  let electronApp;
  try {
    electronApp = await electron.launch({
      args: ['.', `--user-data-dir=${profileDir}`],
      cwd: frontendDir,
      env: { ...process.env, ASTRA_PYTHON: 'python' },
      timeout: 30_000,
    });
    const page = await electronApp.firstWindow();
    await electronApp.evaluate(({ BrowserWindow }) => {
      BrowserWindow.getAllWindows()[0]?.setSize(1440, 1000);
    });
    await page.setViewportSize({ width: 1440, height: 956 });
    await page.reload();
    await page.getByRole('heading', { name: '工作台', exact: true }).waitFor();

    if (withMockData) {
      await openPage(page, '#/inference', '推理模块');
      await electronApp.evaluate(({ BrowserWindow }) => {
        BrowserWindow.getAllWindows()[0]?.webContents.send('inference:result', {
          taskId: 'dashboard-showcase-result',
          units: [1],
          predictions: [42.8],
          targets: [40],
          rmse: 2.8,
          mae: 2.8,
          score: 0.24,
          seconds: 0.12,
        });
      });
      await page.getByText('推理结果', { exact: true }).waitFor();
      await openPage(page, '#/inference-history', '推理历史');
      await page.getByText('FD001 评估任务', { exact: true }).waitFor();
      await openPage(page, '#/', '工作台');
      await page.screenshot({ path: join(targetDir, '01-dashboard-overview.png') });

      await openPage(page, '#/inference', '推理模块');
      await page.getByRole('button', { name: /C-MAPSS FD001/ }).click();
      await page.getByRole('button', { name: /ASTRA-MultiScale-TCN/ }).click();
      const units = Array.from({ length: 100 }, (_, index) => index + 1);
      const targets = units.map((unit) => Math.max(8, 132 - unit * 1.17));
      const predictions = targets.map(
        (target, index) => target + Math.sin(index / 6) * 7.2 + Math.cos(index / 13) * 3.1,
      );
      await electronApp.evaluate(
        ({ BrowserWindow }, result) => {
          BrowserWindow.getAllWindows()[0]?.webContents.send('inference:result', result);
        },
        {
          taskId: 'inference-showcase-1',
          units,
          predictions,
          targets,
          rmse: 13.537,
          mae: 10.284,
          score: 318.42,
          seconds: 0.742,
        },
      );
      const analysis = page.getByRole('region', { name: '推理结果分析' });
      await analysis.waitFor();
      await analysis.scrollIntoViewIfNeeded();
      await page.waitForTimeout(600);
      await page.screenshot({ path: join(targetDir, '02-inference-analysis.png') });

      await openPage(page, '#/models', '模型管理');
      await page.getByRole('button', { name: /ASTRA-MultiScale-TCN/ }).click();
      await page.getByText('最近实测效果', { exact: true }).waitFor();
      await page.screenshot({ path: join(targetDir, '03-model-performance.png') });
    } else {
      await page.screenshot({ path: join(targetDir, '01-dashboard-empty.png') });
      await openPage(page, '#/data', '数据采集');
      await page.getByText('暂无数据集', { exact: true }).waitFor();
      await page.screenshot({ path: join(targetDir, '02-datasets-empty.png') });
      await openPage(page, '#/models', '模型管理');
      await page.getByText('暂无模型', { exact: true }).waitFor();
      await page.screenshot({ path: join(targetDir, '03-models-empty.png') });
    }
  } finally {
    await electronApp?.close().catch(() => {});
    rmSync(profileDir, { recursive: true, force: true });
  }
}

mkdirSync(outputDir, { recursive: true });
await captureSet({ name: 'with-mock', withMockData: true });
await captureSet({ name: 'without-mock', withMockData: false });
console.log(`SHOWCASE_CAPTURE_OK: ${outputDir}`);
