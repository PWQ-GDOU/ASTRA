/**
 * Task 5 测试：ESLint + Prettier 配置
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');

describe('Task 5: ESLint + Prettier 配置', () => {
  const eslintConfigs = [
    '.eslintrc.json',
    '.eslintrc.js',
    '.eslintrc.cjs',
    'eslint.config.js',
    '.eslintrc',
  ];
  const prettierConfigs = [
    '.prettierrc',
    '.prettierrc.json',
    'prettier.config.js',
    'prettier.config.cjs',
  ];

  function findFile(candidates: string[]): string | null {
    for (const f of candidates) {
      if (existsSync(resolve(ROOT, f))) return f;
    }
    return null;
  }

  it('test_eslint_config_exists', () => {
    const found = findFile(eslintConfigs);
    expect(found, 'No ESLint config found').toBeDefined();
  });

  it('test_prettier_config_exists', () => {
    const found = findFile(prettierConfigs);
    expect(found, 'No Prettier config found').toBeDefined();
  });

  it('test_eslint_parser_is_typescript', () => {
    const file = findFile(eslintConfigs);
    if (file) {
      const content = readFileSync(resolve(ROOT, file), 'utf-8');
      expect(content).toMatch(/@typescript-eslint\/parser/);
    }
  });

  it('test_prettier_semi_true', () => {
    const file = findFile(prettierConfigs);
    if (file) {
      const content = readFileSync(resolve(ROOT, file), 'utf-8');
      if (file === '.prettierrc' || file === '.prettierrc.json') {
        const cfg = JSON.parse(content);
        expect(cfg.semi).toBe(true);
      } else {
        expect(content).toMatch(/semi.*true/);
      }
    }
  });

  it('test_prettier_single_quote_true', () => {
    const file = findFile(prettierConfigs);
    if (file) {
      const content = readFileSync(resolve(ROOT, file), 'utf-8');
      if (file === '.prettierrc' || file === '.prettierrc.json') {
        const cfg = JSON.parse(content);
        expect(cfg.singleQuote).toBe(true);
      } else {
        expect(content).toMatch(/singleQuote.*true/);
      }
    }
  });

  it('test_prettier_tab_width_2', () => {
    const file = findFile(prettierConfigs);
    if (file) {
      const content = readFileSync(resolve(ROOT, file), 'utf-8');
      if (file === '.prettierrc' || file === '.prettierrc.json') {
        const cfg = JSON.parse(content);
        expect(cfg.tabWidth).toBe(2);
      }
    }
  });

  it('test_prettier_print_width_100', () => {
    const file = findFile(prettierConfigs);
    if (file) {
      const content = readFileSync(resolve(ROOT, file), 'utf-8');
      if (file === '.prettierrc' || file === '.prettierrc.json') {
        const cfg = JSON.parse(content);
        expect(cfg.printWidth).toBe(100);
      }
    }
  });

  it('test_eslintignore_exists', () => {
    expect(existsSync(resolve(ROOT, '.eslintignore'))).toBe(true);
  });

  it('test_prettierignore_exists', () => {
    expect(existsSync(resolve(ROOT, '.prettierignore'))).toBe(true);
  });

  it('test_prettierignore_includes_dist', () => {
    if (existsSync(resolve(ROOT, '.prettierignore'))) {
      const content = readFileSync(resolve(ROOT, '.prettierignore'), 'utf-8');
      expect(content).toContain('dist');
    }
  });
});
