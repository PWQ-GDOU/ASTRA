/**
 * Task 8 测试：global.css — Tailwind 基础层 + 玻璃态工具类
 */
import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = resolve(__dirname, '..', '..');
const GLOBAL_CSS = resolve(ROOT, 'src', 'renderer', 'styles', 'global.css');

function readCss(): string {
  return readFileSync(GLOBAL_CSS, 'utf-8');
}

describe('Task 8: global.css', () => {
  it('test_global_css_exists', () => {
    expect(existsSync(GLOBAL_CSS)).toBe(true);
  });

  it('test_global_css_has_tailwind_directives', () => {
    const css = readCss();
    expect(css).toContain('@tailwind base');
    expect(css).toContain('@tailwind components');
    expect(css).toContain('@tailwind utilities');
  });

  it('test_global_css_body_uses_bg_base', () => {
    const css = readCss();
    expect(css).toMatch(/bg-base/);
  });

  it('test_global_css_has_glass_utility', () => {
    const css = readCss();
    expect(css).toMatch(/\.glass\b/);
  });

  it('test_global_css_has_glass_panel_utility', () => {
    const css = readCss();
    expect(css).toMatch(/\.glass-panel/);
  });

  it('test_global_css_has_scrollbar_styles', () => {
    const css = readCss();
    expect(css).toMatch(/webkit-scrollbar/);
  });

  it('test_global_css_has_selection_styles', () => {
    const css = readCss();
    expect(css).toMatch(/::selection/);
  });

  it('test_global_css_imports_theme', () => {
    const css = readCss();
    // Either @import or referenced via var(--...) from theme
    const hasImport = css.includes('theme.css') || css.includes('var(--');
    expect(hasImport).toBe(true);
  });

  it('test_global_css_is_valid_syntax', () => {
    const css = readCss();
    // Check balanced braces
    const openBraces = (css.match(/\{/g) || []).length;
    const closeBraces = (css.match(/\}/g) || []).length;
    expect(openBraces).toBe(closeBraces);
    expect(openBraces).toBeGreaterThan(0);
  });
});
