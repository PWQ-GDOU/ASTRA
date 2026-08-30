/**
 * Task 4 测试：Tailwind CSS 3 配置 + 自定义主题扩展
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');

function readConfig(): string {
  return readFileSync(resolve(ROOT, 'tailwind.config.ts'), 'utf-8');
}

describe('Task 4: Tailwind CSS 3 配置', () => {
  it('test_tailwind_config_exists', () => {
    expect(existsSync(resolve(ROOT, 'tailwind.config.ts'))).toBe(true);
  });

  it('test_tailwind_content_includes_src_ts_tsx', () => {
    const config = readConfig();
    expect(config).toMatch(/content.*src/);
    expect(config).toMatch(/\{ts,tsx\}/);
  });

  it('test_tailwind_colors_has_accent', () => {
    const config = readConfig();
    expect(config).toContain('accent');
  });

  it('test_tailwind_colors_has_surface', () => {
    const config = readConfig();
    expect(config).toContain('surface');
  });

  it('test_tailwind_font_sans_is_inter', () => {
    const config = readConfig();
    expect(config).toContain('Inter');
  });

  it('test_tailwind_font_mono_is_jetbrains', () => {
    const config = readConfig();
    expect(config).toContain('JetBrains Mono');
  });

  it('test_tailwind_border_radius_has_custom', () => {
    const config = readConfig();
    expect(config).toMatch(/borderRadius/);
    expect(config).toContain('radius');
  });

  it('test_tailwind_backdrop_blur_has_glass', () => {
    const config = readConfig();
    expect(config).toContain('20px');
  });

  it('test_postcss_config_exists', () => {
    expect(existsSync(resolve(ROOT, 'postcss.config.js'))).toBe(true);
  });

  it('test_postcss_config_has_tailwind_plugin', () => {
    const config = readFileSync(resolve(ROOT, 'postcss.config.js'), 'utf-8');
    expect(config).toContain('tailwindcss');
    expect(config).toContain('autoprefixer');
  });
});
