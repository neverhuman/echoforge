import { afterEach, describe, expect, test, vi } from "vitest";
import type { RunSummary } from "./radarContract";
import {
	archiveRun,
	createMonteCarloRun,
	duplicateRun,
	fetchRunQueueSummary,
	restoreRun,
} from "./radarControl";

const sampleRun: RunSummary = {
	run_id: "run-shahed-ingress-00",
	created_utc: "unix-1",
	status: "validated",
	config: {
		scenario_id: "shahed-ingress",
		scenario_label: "Shahed-class ingress",
		mode: "live",
		seed: 1592597009,
		scenario_hash: "abc123",
		object_source_card: "public-proxy",
		material_assumption_card: "public-proxy-material",
		solver_chain_version: "echoforge-studio-0.2.0",
	},
	validation: {
		tier: "V1 public-proxy",
		grade: "export-ready-with-limitations",
		export_gate_passed: true,
		source_confidence: "public source-card assumptions only",
		uncertainty_statement: "synthetic uncertainty only",
		known_limitations: [],
		leakage_guard_status: "pass",
		reproducibility_metadata: ["run_id", "seed"],
	},
	artifacts: [],
};

function stubFetch(payload: unknown) {
	const fetchMock = vi.fn(async () => ({
		ok: true,
		status: 200,
		json: async () => payload,
	}));
	vi.stubGlobal("fetch", fetchMock);
	return fetchMock;
}

describe("radarControl run adapters", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	test("archiveRun posts an archive request body", async () => {
		const fetchMock = stubFetch({ ...sampleRun, status: "archived" });
		await archiveRun(sampleRun.run_id, { reason: "review complete" });
		const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("/api/runs/run-shahed-ingress-00/archive");
		expect(init.method).toBe("POST");
		expect(JSON.parse(String(init.body))).toEqual({
			reason: "review complete",
		});
	});

	test("restoreRun posts to the restore endpoint", async () => {
		const fetchMock = stubFetch(sampleRun);
		await restoreRun(sampleRun.run_id);
		const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("/api/runs/run-shahed-ingress-00/restore");
		expect(init.method).toBe("POST");
	});

	test("duplicateRun sends seed and mode controls", async () => {
		const fetchMock = stubFetch({
			...sampleRun,
			run_id: "run-shahed-ingress-copy-01",
		});
		await duplicateRun(sampleRun.run_id, {
			seed: 2026052101,
			mode: "monte_carlo",
		});
		const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("/api/runs/run-shahed-ingress-00/duplicate");
		expect(init.method).toBe("POST");
		expect(JSON.parse(String(init.body))).toEqual({
			seed: 2026052101,
			mode: "monte_carlo",
		});
	});

	test("createMonteCarloRun posts a composed public-proxy request", async () => {
		const fetchMock = stubFetch({ ...sampleRun, status: "queued" });
		await createMonteCarloRun({
			scenario_id: "coastal-clutter",
			source_pack: "public-proxy-v1",
			object_pack: "airspace-objects-v1",
			hard_negatives: ["bird_flock_dense"],
			weather_profile: "uae_coastal_summer",
			detector_pipeline: "physics_cfar_track_fusion_v1",
			seed: 2026052101,
			run_count: 12,
			workers: 4,
			max_concurrent: 2,
			smoke: true,
			validation_target: "V1 public-proxy",
		});
		const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("/api/runs");
		expect(init.method).toBe("POST");
		expect(JSON.parse(String(init.body))).toMatchObject({
			scenario_id: "coastal-clutter",
			hard_negatives: ["bird_flock_dense"],
			validation_target: "V1 public-proxy",
		});
	});

	test("fetchRunQueueSummary reads queue metadata", async () => {
		const fetchMock = stubFetch({
			total: 3,
			active: 3,
			archived: 0,
			export_ready: 3,
			queued: 0,
			newest_created_utc: "unix-1",
			validation_tiers: ["V1 public-proxy"],
		});
		const summary = await fetchRunQueueSummary();
		expect(summary.total).toBe(3);
		expect(fetchMock.mock.calls[0][0]).toBe("/api/runs/queue/summary");
	});
});
