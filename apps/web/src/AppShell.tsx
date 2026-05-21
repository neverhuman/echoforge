import { useState } from "react";
import App from "./App";
import ApiHeadlessPanel from "./radar/ApiHeadlessPanel";
import ArtifactsPanel from "./radar/ArtifactsPanel";
import CommandCenter from "./radar/CommandCenter";
import DetectorComparisonPanel from "./radar/DetectorComparisonPanel";
import JobsPanel from "./radar/JobsPanel";
import MonteCarloBuilder from "./radar/MonteCarloBuilder";
import RadarConsole from "./radar/RadarConsole";
import RunsPanel from "./radar/RunsPanel";

type StudioView =
	| "command"
	| "live"
	| "builder"
	| "runs"
	| "lab"
	| "artifacts"
	| "api"
	| "contracts";

const VIEWS: Array<[StudioView, string]> = [
	["command", "Command Center"],
	["live", "Live Radar"],
	["builder", "Scenario / Monte Carlo"],
	["runs", "Runs & Queue"],
	["lab", "Benchmark / Detector Lab"],
	["artifacts", "Artifacts"],
	["api", "API / Headless"],
	["contracts", "Contracts"],
];

function viewTestId(id: StudioView): string {
	return id === "live" ? "tab-radar" : `tab-${id}`;
}

/** Top-level Studio host: navigation, strict-open boundary chips, and the
 *  product work surfaces backed by the Rust origin. */
export default function AppShell() {
	const [view, setView] = useState<StudioView>("command");

	return (
		<div className="appshell" data-testid="studio-shell">
			<header className="appshell__bar">
				<div>
					<span className="appshell__brand">
						EchoForge <span className="appshell__brand-sub">Studio</span>
					</span>
					<p className="appshell__tagline">
						Strict-open synthetic radar artifacts, queue control, and validation
						evidence.
					</p>
				</div>
				<div className="appshell__badges">
					<span className="job-chip">public-proxy</span>
					<span className="job-chip">uncertainty-scored</span>
					<span className="job-chip job-chip--gold">not measured truth</span>
				</div>
			</header>
			<div className="appshell__workspace">
				<nav className="appshell__rail" aria-label="Studio views">
					{VIEWS.map(([id, label]) => (
						<button
							type="button"
							key={id}
							className={view === id ? "is-active" : ""}
							onClick={() => setView(id)}
							data-testid={viewTestId(id)}
						>
							{label}
						</button>
					))}
				</nav>
				<main className="appshell__body" data-testid="studio-main">
					{view === "command" ? <CommandCenter /> : null}
					{view === "live" ? <RadarConsole /> : null}
					{view === "builder" ? <MonteCarloBuilder /> : null}
					{view === "runs" ? <RunsPanel /> : null}
					{view === "lab" ? (
						<>
							<DetectorComparisonPanel />
							<JobsPanel />
						</>
					) : null}
					{view === "artifacts" ? <ArtifactsPanel /> : null}
					{view === "api" ? <ApiHeadlessPanel /> : null}
					{view === "contracts" ? <App /> : null}
				</main>
			</div>
		</div>
	);
}
