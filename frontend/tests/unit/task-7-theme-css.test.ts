/**
 * Task 7 测试：theme.css — 方案 C 全部 CSS 变量定义
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');
const THEME_PATH = resolve(ROOT, 'src', 'renderer', 'styles', 'theme.css');

function readTheme(): string {
  return readFileSync(THEME_PATH, 'utf-8');
}

describe('Task 7: theme.css CSS 变量定义', () => {
  it('test_theme_css_exists', () => {
    expect(existsSync(THEME_PATH)).toBe(true);
  });

  it('test_theme_defines_bg_deep_light', () => {
    const css = readTheme();
    expect(css).toMatch(/--bg-deep\s*:\s*#e5e7eb/);
  });

  it('test_theme_defines_accent', () => {
    const css = readTheme();
    expect(css).toMatch(/--accent\s*:\s*#[0-9A-Fa-f]+/);
  });

  it('test_theme_defines_foreground', () => {
    const css = readTheme();
    expect(css).toMatch(/--foreground\s*:\s*#[0-9A-Fa-f]+/);
  });

  it('test_theme_defines_radius_default', () => {
    const css = readTheme();
    expect(css).toMatch(/--radius\s*:\s*16px/);
  });

  it('test_theme_defines_font_sans', () => {
    const css = readTheme();
    expect(css).toMatch(/--font-sans/);
    expect(css).toContain('Inter');
  });

  it('test_theme_defines_sidebar_width', () => {
    const css = readTheme();
    expect(css).toMatch(/--sidebar-width\s*:\s*240px/);
  });

  it('test_theme_defines_titlebar_height', () => {
    const css = readTheme();
    expect(css).toMatch(/--titlebar-height\s*:\s*44px/);
  });

  it('test_theme_defines_easing', () => {
    const css = readTheme();
    expect(css).toMatch(/--easing\s*:\s*cubic-bezier/);
  });

  it('test_theme_is_valid_css_syntax', () => {
    const css = readTheme();
    // Must have :root { and closing }
    expect(css).toMatch(/:root\s*\{/);
    // Must not have unbalanced braces
    const openBraces = (css.match(/\{/g) || []).length;
    const closeBraces = (css.match(/\}/g) || []).length;
    expect(openBraces).toBe(closeBraces);
    expect(openBraces).toBeGreaterThan(0);
  });
});
