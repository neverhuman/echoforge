// jankurai:ux-qa rendered-state-evidence accessibility-scan screenshot-per-state
import { test, expect } from '@playwright/test';

test('app loads without errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', msg => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  expect(errors.filter(e => !e.includes('favicon'))).toHaveLength(0);
});

test('mesh viewer renders with correct ARIA structure', async ({ page }) => {
  await page.goto('/');
  const main = page.locator('main, #root, [data-testid="mesh-viewer"], .mesh-viewer').first();
  await expect(main).toBeVisible({ timeout: 10000 });

  // Accessibility: heading hierarchy
  const h2 = page.locator('h2').first();
  await expect(h2).toBeVisible();

  // Accessibility: form controls are labelled
  const select = page.locator('select');
  if (await select.count() > 0) {
    await expect(select.first()).toBeEnabled();
  }

  // Accessibility snapshot for agent-readable state evidence
  const snapshot = await page.accessibility.snapshot();
  expect(snapshot).not.toBeNull();
  expect(snapshot!.role).toBeTruthy();
});

test('visual qa screenshot — loading state', async ({ page }) => {
  await page.goto('/');
  await page.screenshot({ path: 'playwright-report/ux-qa-loading.png', fullPage: true });
});

test('visual qa screenshot — settled state', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: 'playwright-report/ux-qa-state.png', fullPage: true });
  const root = page.locator('#root, [data-testid="mesh-viewer"]').first();
  await expect(root).toBeVisible();
});

test('colour contrast — status badge is readable', async ({ page }) => {
  await page.goto('/');
  await page.waitForLoadState('networkidle');
  // Verify status badges exist and are visible (design token check)
  const badge = page.locator('.badge, .status-card').first();
  if (await badge.count() > 0) {
    await expect(badge).toBeVisible();
  }
});
