// jankurai:ux-qa rendered-state-evidence
import { expect, test } from "@playwright/test";

test("command center is the default Studio view", async ({ page }) => {
	await page.goto("/");
	await expect(page.getByTestId("command-center")).toBeVisible({
		timeout: 10000,
	});
	await expect(page.getByTestId("studio-shell")).toBeVisible();
});

test("live radar view exposes all displays and presets", async ({ page }) => {
	await page.goto("/");
	await page.getByTestId("tab-radar").click();
	await expect(page.getByTestId("radar-console")).toBeVisible({
		timeout: 10000,
	});
	await expect(page.getByTestId("live-preset-expert")).toBeVisible();
	await expect(page.getByTestId("live-preset-attribution")).toBeVisible();
	await expect(page.getByTestId("ppi-scope")).toBeVisible();
	await expect(page.getByTestId("range-doppler-map")).toBeVisible();
	await expect(page.getByTestId("micro-doppler-waterfall")).toBeVisible();
	await expect(page.getByTestId("tracks-panel")).toBeVisible();
	await expect(page.getByTestId("telemetry-panel")).toBeVisible();
	await expect(page.getByTestId("scenario-control")).toBeVisible();
});

test("scenario control exposes transport controls", async ({ page }) => {
	await page.goto("/");
	await page.getByTestId("tab-radar").click();
	await expect(page.getByTestId("scenario-control")).toBeVisible({
		timeout: 10000,
	});
	await expect(page.getByTestId("scenario-select")).toBeVisible();
	await expect(page.getByTestId("speed-slider")).toBeVisible();
	await expect(page.getByTestId("sim-replay")).toBeVisible();
});

test("connection banner appears when no backend is reachable", async ({
	page,
}) => {
	await page.goto("/");
	await page.getByTestId("tab-radar").click();
	// The static preview has no radar service, so the stream cannot open.
	await expect(page.getByTestId("connection-banner")).toBeVisible({
		timeout: 10000,
	});
});

test("panels show named empty states before any data arrives", async ({
	page,
}) => {
	await page.goto("/");
	await page.getByTestId("tab-radar").click();
	await expect(page.getByTestId("tracks-panel")).toBeVisible({
		timeout: 10000,
	});
	await expect(page.getByTestId("tracks-panel")).toContainText(
		"No active tracks",
	);
	await expect(page.getByTestId("telemetry-panel")).toContainText(
		"Awaiting telemetry",
	);
});

test("navigation switches between Studio surfaces", async ({ page }) => {
	await page.goto("/");
	await expect(page.getByTestId("command-center")).toBeVisible({
		timeout: 10000,
	});
	await page.getByTestId("tab-radar").click();
	await expect(page.getByTestId("radar-console")).toBeVisible();
	await page.getByTestId("tab-builder").click();
	await expect(page.getByTestId("monte-carlo-builder")).toBeVisible();
	await page.getByTestId("tab-runs").click();
	await expect(page.getByTestId("runs-view")).toBeVisible();
	await page.getByTestId("tab-artifacts").click();
	await expect(page.getByTestId("artifacts-view")).toBeVisible();
	await page.getByTestId("tab-api").click();
	await expect(page.getByTestId("api-headless-view")).toBeVisible();
	await page.getByTestId("tab-contracts").click();
	await expect(page.getByTestId("radar-console")).toHaveCount(0);
});

test("jobs tab exposes the ml pipeline composer and queue", async ({
	page,
}) => {
	await page.goto("/");
	await page.getByTestId("tab-lab").click();
	await expect(page.getByTestId("jobs-view")).toBeVisible({ timeout: 10000 });
	await expect(page.getByTestId("job-composer")).toBeVisible();
	await expect(page.getByTestId("job-board")).toBeVisible();
	await expect(page.getByTestId("job-launch")).toBeVisible();
});

test("monte carlo builder renders API and CLI equivalents", async ({
	page,
}) => {
	await page.goto("/");
	await page.getByTestId("tab-builder").click();
	await expect(page.getByTestId("monte-carlo-builder")).toBeVisible({
		timeout: 10000,
	});
	await expect(page.getByTestId("mc-api-example")).toContainText(
		"POST /api/runs",
	);
	await expect(page.getByTestId("mc-queue-run")).toBeVisible();
});
