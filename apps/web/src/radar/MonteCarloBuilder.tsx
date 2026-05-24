import { useCallback, useMemo, useState } from "react";
import type { MonteCarloRunRequest, RunSummary } from "./radarContract";
import { createMonteCarloRun } from "./radarControl";

const SOURCE_PACKS = [
	"public-proxy-v1",
	"public-proxy-clutter-v1",
	"airspace-objects-v1",
];
const OBJECT_PACKS = [
	"airspace-objects-v1",
	"public-proxy-uav-v1",
	"coastal-confusers-v1",
];
const WEATHER_PROFILES = [
	"uae_coastal_summer",
	"black_sea_autumn_duct",
	"ukraine_winter",
	"gulf_monsoon_heavy_rain",
];
const DETECTOR_PIPELINES = [
	"physics_cfar_track_fusion_v1",
	"tensor_microdoppler_fusion_v1",
	"raw_iq_ssl_research_v1",
];
const HARD_NEGATIVES = [
	"bird_flock_dense",
	"rain_cell",
	"wind_turbine_large",
	"rfi_burst_emitter",
	"vegetation_canopy",
];

type Preset = "smoke" | "attribution" | "full";

function presetLabel(preset: Preset): string {
	if (preset === "smoke") return "Smoke";
	if (preset === "attribution") return "Attribution";
	return "Full";
}

function requestToCli(request: MonteCarloRunRequest): string {
	return [
		"rtk cargo run -p echoforge-cli -- monte-carlo",
		`--scenario ${request.scenario_id}`,
		`--source-pack ${request.source_pack}`,
		`--object-pack ${request.object_pack}`,
		`--weather ${request.weather_profile}`,
		`--detector ${request.detector_pipeline}`,
		`--seed ${request.seed}`,
		`--runs ${request.run_count}`,
		`--workers ${request.workers}`,
		`--max-concurrent ${request.max_concurrent}`,
		`--validation-target "${request.validation_target}"`,
		request.smoke ? "--smoke" : "--full",
	].join(" ");
}

function ChoiceGrid({
	label,
	value,
	choices,
	onChange,
}: {
	label: string;
	value: string;
	choices: string[];
	onChange: (value: string) => void;
}) {
	return (
		<div className="radar-field">
			<span>{label}</span>
			<div className="choice-grid">
				{choices.map((choice) => (
					<button
						type="button"
						key={choice}
						className={choice === value ? "is-active" : ""}
						onClick={() => onChange(choice)}
					>
						{choice}
					</button>
				))}
			</div>
		</div>
	);
}

export default function MonteCarloBuilder() {
	const [preset, setPreset] = useState<Preset>("smoke");
	const [scenarioId, setScenarioId] = useState("coastal-clutter");
	const [sourcePack, setSourcePack] = useState(SOURCE_PACKS[0]);
	const [objectPack, setObjectPack] = useState(OBJECT_PACKS[0]);
	const [weatherProfile, setWeatherProfile] = useState(WEATHER_PROFILES[0]);
	const [detectorPipeline, setDetectorPipeline] = useState(
		DETECTOR_PIPELINES[0],
	);
	const [hardNegatives, setHardNegatives] = useState<string[]>([
		"bird_flock_dense",
		"rain_cell",
	]);
	const [seed, setSeed] = useState(2026052101);
	const [runCount, setRunCount] = useState(12);
	const [workers, setWorkers] = useState(4);
	const [maxConcurrent, setMaxConcurrent] = useState(2);
	const [smoke, setSmoke] = useState(true);
	const [validationTarget, setValidationTarget] = useState("V1 public-proxy");
	const [busy, setBusy] = useState(false);
	const [lastRun, setLastRun] = useState<RunSummary | null>(null);
	const [error, setError] = useState<string | null>(null);

	const request = useMemo<MonteCarloRunRequest>(
		() => ({
			scenario_id: scenarioId,
			source_pack: sourcePack,
			object_pack: objectPack,
			hard_negatives: hardNegatives,
			weather_profile: weatherProfile,
			detector_pipeline: detectorPipeline,
			seed,
			run_count: runCount,
			workers,
			max_concurrent: maxConcurrent,
			smoke,
			validation_target: validationTarget,
		}),
		[
			detectorPipeline,
			hardNegatives,
			maxConcurrent,
			objectPack,
			runCount,
			scenarioId,
			seed,
			smoke,
			sourcePack,
			validationTarget,
			weatherProfile,
			workers,
		],
	);

	const applyPreset = useCallback((nextPreset: Preset) => {
		setPreset(nextPreset);
		if (nextPreset === "smoke") {
			setRunCount(12);
			setWorkers(4);
			setMaxConcurrent(2);
			setSmoke(true);
			setValidationTarget("V1 public-proxy");
		} else if (nextPreset === "attribution") {
			setRunCount(64);
			setWorkers(8);
			setMaxConcurrent(3);
			setSmoke(false);
			setValidationTarget("V1 public-proxy attribution");
		} else {
			setRunCount(256);
			setWorkers(16);
			setMaxConcurrent(4);
			setSmoke(false);
			setValidationTarget("V2 benchmarked public-proxy");
		}
	}, []);

	const toggleHardNegative = useCallback((id: string) => {
		setHardNegatives((current) =>
			current.includes(id)
				? current.filter((entry) => entry !== id)
				: [...current, id],
		);
	}, []);

	const queueRun = useCallback(() => {
		setBusy(true);
		setError(null);
		createMonteCarloRun(request)
			.then((run) => setLastRun(run))
			.catch((err: unknown) => {
				const message = err instanceof Error ? err.message : String(err);
				setError(message);
			})
			.finally(() => setBusy(false));
	}, [request]);

	const apiExample = `POST /api/runs\n${JSON.stringify(request, null, 2)}`;
	const cliExample = requestToCli(request);

	return (
		<section className="studio-view" data-testid="monte-carlo-builder">
			<div className="studio-view__header">
				<div>
					<h2>Scenario / Monte Carlo Builder</h2>
					<p>
						Compose public-proxy campaign requests with seeds, hard negatives,
						validation targets, and run limits.
					</p>
				</div>
				<div className="studio-segment">
					{(["smoke", "attribution", "full"] as Preset[]).map((entry) => (
						<button
							type="button"
							key={entry}
							className={preset === entry ? "is-active" : ""}
							onClick={() => applyPreset(entry)}
							data-testid={`mc-preset-${entry}`}
						>
							{presetLabel(entry)}
						</button>
					))}
				</div>
			</div>

			<div className="builder-layout">
				<section className="radar-panel">
					<h3 className="radar-panel__title">Scenario Inputs</h3>
					<label className="radar-field">
						<span>Scenario ID</span>
						<input
							type="text"
							value={scenarioId}
							onChange={(event) => setScenarioId(event.target.value)}
						/>
					</label>
					<ChoiceGrid
						label="Source pack"
						value={sourcePack}
						choices={SOURCE_PACKS}
						onChange={setSourcePack}
					/>
					<ChoiceGrid
						label="Object pack"
						value={objectPack}
						choices={OBJECT_PACKS}
						onChange={setObjectPack}
					/>
					<ChoiceGrid
						label="Weather / clutter"
						value={weatherProfile}
						choices={WEATHER_PROFILES}
						onChange={setWeatherProfile}
					/>
				</section>

				<section className="radar-panel">
					<h3 className="radar-panel__title">Expert Controls</h3>
					<ChoiceGrid
						label="Detector pipeline"
						value={detectorPipeline}
						choices={DETECTOR_PIPELINES}
						onChange={setDetectorPipeline}
					/>
					<label className="radar-field">
						<span>Run count - {runCount}</span>
						<input
							type="range"
							min="4"
							max="256"
							step="4"
							value={runCount}
							onChange={(event) => setRunCount(Number(event.target.value))}
						/>
					</label>
					<label className="radar-field">
						<span>Workers - {workers}</span>
						<input
							type="range"
							min="1"
							max="20"
							step="1"
							value={workers}
							onChange={(event) => setWorkers(Number(event.target.value))}
						/>
					</label>
					<label className="radar-field">
						<span>Max concurrent - {maxConcurrent}</span>
						<input
							type="range"
							min="1"
							max="6"
							step="1"
							value={maxConcurrent}
							onChange={(event) => setMaxConcurrent(Number(event.target.value))}
						/>
					</label>
					<label className="radar-field">
						<span>Seed</span>
						<input
							type="number"
							value={seed}
							onChange={(event) => setSeed(Number(event.target.value))}
						/>
					</label>
					<label className="radar-field">
						<span>Validation target</span>
						<input
							type="text"
							value={validationTarget}
							onChange={(event) => setValidationTarget(event.target.value)}
						/>
					</label>
					<label className="radar-field radar-field--inline">
						<input
							type="checkbox"
							checked={smoke}
							onChange={(event) => setSmoke(event.target.checked)}
						/>
						<span>Smoke mode</span>
					</label>
				</section>

				<section className="radar-panel">
					<h3 className="radar-panel__title">Hard Negatives</h3>
					<div className="check-grid">
						{HARD_NEGATIVES.map((id) => (
							<label className="check-tile" key={id}>
								<input
									type="checkbox"
									checked={hardNegatives.includes(id)}
									onChange={() => toggleHardNegative(id)}
								/>
								<span>{id}</span>
							</label>
						))}
					</div>
					<button
						type="button"
						className="radar-btn radar-btn--go"
						onClick={queueRun}
						disabled={busy}
						data-testid="mc-queue-run"
					>
						Queue smoke run
					</button>
					{lastRun ? (
						<p className="radar-control__status">
							Queued {lastRun.run_id} with scenario hash{" "}
							{lastRun.config.scenario_hash}
						</p>
					) : null}
					{error ? (
						<p className="studio-error">Monte Carlo API unavailable: {error}</p>
					) : null}
				</section>

				<section className="radar-panel builder-layout__wide">
					<h3 className="radar-panel__title">API Equivalent</h3>
					<pre className="code-surface" data-testid="mc-api-example">
						{apiExample}
					</pre>
					<h3 className="radar-panel__title">CLI Equivalent</h3>
					<pre className="code-surface">{cliExample}</pre>
				</section>
			</div>
		</section>
	);
}
