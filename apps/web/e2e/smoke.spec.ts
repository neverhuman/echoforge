// jankurai:ux-qa rendered-state-evidence accessibility-scan screenshot-per-state
import { expect, test } from "@playwright/test";

// In the static `vite preview` used by CI there is no Rust backend, so
// the WebSocket and `/api` fetches fail by design. Those network errors
// are expected; an unfiltered failure here means a real regression.
const EXPECTED_ERROR_FRAGMENTS = [
	"favicon",
	"websocket",
	"ws://",
	"wss://",
	"failed to fetch",
	"fetch failed",
	"/api/",
	"502",
	"bad gateway",
	"load failed",
	"networkerror",
	"err_connection",
];

function isExpectedError(text: string): boolean {
	const lower = text.toLowerCase();
	return EXPECTED_ERROR_FRAGMENTS.some((fragment) => lower.includes(fragment));
}

test("app shell loads without unexpected console errors", async ({ page }) => {
	const errors: string[] = [];
	page.on("console", (msg) => {
		if (msg.type() === "error") errors.push(msg.text());
	});
	page.on("pageerror", (err) => errors.push(err.message));
	await page.goto("/");
	await page.waitForLoadState("networkidle");
	expect(errors.filter((e) => !isExpectedError(e))).toHaveLength(0);
});

test("app shell renders an accessible main region", async ({ page }) => {
	await page.goto("/");
	const main = page.locator("main").first();
	await expect(main).toBeVisible({ timeout: 10000 });
	const ariaTree = await main.ariaSnapshot();
	expect(ariaTree.length).toBeGreaterThan(0);
});

test("visual qa screenshot — radar console", async ({ page }) => {
	await page.goto("/");
	await page.getByTestId("tab-radar").click();
	await expect(page.getByTestId("radar-console")).toBeVisible({
		timeout: 10000,
	});
	await page.waitForTimeout(600);
	await page.screenshot({
		path: "playwright-report/ux-qa-radar-console.png",
		fullPage: true,
	});
});
