// jankurai:ux-qa rendered-state-evidence
import { expect, test } from '@playwright/test';

test('radar console is the default view with all four displays', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('radar-console')).toBeVisible({ timeout: 10000 });
  await expect(page.getByTestId('ppi-scope')).toBeVisible();
  await expect(page.getByTestId('range-doppler-map')).toBeVisible();
  await expect(page.getByTestId('micro-doppler-waterfall')).toBeVisible();
  await expect(page.getByTestId('tracks-panel')).toBeVisible();
  await expect(page.getByTestId('telemetry-panel')).toBeVisible();
  await expect(page.getByTestId('scenario-control')).toBeVisible();
});

test('scenario control exposes transport controls', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('scenario-control')).toBeVisible({ timeout: 10000 });
  await expect(page.getByTestId('scenario-select')).toBeVisible();
  await expect(page.getByTestId('speed-slider')).toBeVisible();
  await expect(page.getByTestId('sim-replay')).toBeVisible();
});

test('connection banner appears when no backend is reachable', async ({ page }) => {
  await page.goto('/');
  // The static preview has no radar service, so the stream cannot open.
  await expect(page.getByTestId('connection-banner')).toBeVisible({ timeout: 10000 });
});

test('panels show named empty states before any data arrives', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('tracks-panel')).toBeVisible({ timeout: 10000 });
  await expect(page.getByTestId('tracks-panel')).toContainText('No active tracks');
  await expect(page.getByTestId('telemetry-panel')).toContainText('Awaiting telemetry');
});

test('tabs switch between the radar console and the contract surface', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('radar-console')).toBeVisible({ timeout: 10000 });
  await page.getByTestId('tab-contracts').click();
  await expect(page.getByTestId('radar-console')).toHaveCount(0);
  await page.getByTestId('tab-radar').click();
  await expect(page.getByTestId('radar-console')).toBeVisible();
});
