import { _electron as electron } from 'playwright';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { join, resolve } from 'path';
import { tmpdir } from 'os';
import { fileURLToPath } from 'url';

const frontendDir = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const backendDir = resolve(frontendDir, '..', 'backend');
const cmapssDir = join(
  backendDir,
  'data',
  'processed',
  'cmapss',
  '6. Turbofan Engine Degradation Simulation Data Set',
);
const profileDir = mkdtempSync(join(tmpdir(), 'spacecraft-e2e-'));
let electronApp;

try {
  electronApp = await electron.launch({
    args: ['.', `--user-data-dir=${profileDir}`],
    cwd: frontendDir,
    env: {
      ...process.env,
      ASTRA_DIR: backendDir,
      ASTRA_PYTHON: 'python',
    },
    timeout: 30_000,
  });
  const userDataDir = await electronApp.evaluate(({ app }) => app.getPath('userData'));
  if (resolve(userDataDir) !== resolve(profileDir)) {
    throw new Error(`Electron 未使用隔离测试目录：${userDataDir}`);
  }

  writeFileSync(
    join(userDataDir, 'datasets.json'),
    JSON.stringify([
      {
        id: 'FD001-e2e',
        name: 'FD001 E2E',
        samples: 100,
        format: 'cmapss',
        type: 'test',
        createdAt: new Date().toISOString(),
        files: [
          join(cmapssDir, 'train_FD001.txt'),
          join(cmapssDir, 'test_FD001.txt'),
          join(cmapssDir, 'RUL_FD001.txt'),
        ],
        rootPath: cmapssDir,
        trainFile: join(cmapssDir, 'train_FD001.txt'),
        testFile: join(cmapssDir, 'test_FD001.txt'),
        rulFile: join(cmapssDir, 'RUL_FD001.txt'),
        astraCompatible: true,
      },
    ]),
  );
  writeFileSync(
    join(userDataDir, 'models.json'),
    JSON.stringify([
      {
        id: 'FD001-model-e2e',
        name: 'FD001 E2E checkpoint',
        status: 'available',
        source: 'imported',
        isDefault: true,
        path: join(
          backendDir,
          'outputs',
          'clean_fd001_multiscale_w60',
          'FD001_multiscale_all24_s42.pt',
        ),
        createdAt: new Date().toISOString(),
      },
    ]),
  );

  const page = await electronApp.firstWindow();
  await page.reload();
  await page.getByText('推理模块', { exact: true }).click();
  await page.getByText('FD001 E2E', { exact: true }).click();
  await page.getByText('FD001 E2E checkpoint', { exact: true }).click();
  await page.getByRole('button', { name: '开始推理' }).click();
  await page.getByText('推理结果', { exact: true }).waitFor({ timeout: 60_000 });
  await page.getByText('100', { exact: true }).waitFor();
  await page.getByText('预测与真实 RUL', { exact: true }).waitFor();
  await page.getByText('样本误差明细', { exact: true }).waitFor();
  await page.getByText('最大绝对误差', { exact: true }).waitFor();

  const persistedModels = JSON.parse(readFileSync(join(userDataDir, 'models.json'), 'utf8'));
  const evaluatedModel = persistedModels.find((model) => model.id === 'FD001-model-e2e');
  if (
    evaluatedModel?.performance?.datasetName !== 'FD001 E2E' ||
    evaluatedModel.performance.sampleCount !== 100 ||
    typeof evaluatedModel.performance.rmse !== 'number'
  ) {
    throw new Error('真实推理结果未正确写回模型效果');
  }

  await page.getByText('模型管理', { exact: true }).click();
  await page.getByRole('heading', { name: '模型管理', exact: true }).waitFor();
  const modelCard = page.getByText(/^FD001 E2E checkpoint/).first();
  try {
    await modelCard.waitFor({ timeout: 5_000 });
  } catch {
    throw new Error(`模型页未同步推理后的模型状态：\n${await page.locator('body').innerText()}`);
  }
  await modelCard.click();
  await page.getByText('最近实测效果', { exact: true }).waitFor();
  await page.getByText('FD001 E2E', { exact: true }).waitFor();
  await page.getByText('100 个样本', { exact: true }).waitFor();

  await page.getByText('推理历史', { exact: true }).click();
  await page.getByText('推理-FD001 E2E', { exact: true }).waitFor();
  console.log(
    'REAL_E2E_OK: renderer -> Electron IPC -> ASTRA Python -> result analysis/model performance/history',
  );
} finally {
  await electronApp?.close().catch(() => {});
  rmSync(profileDir, { recursive: true, force: true });
}
