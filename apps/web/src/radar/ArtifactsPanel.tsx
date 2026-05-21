import { useCallback, useEffect, useMemo, useState } from "react";
import type { RunArtifact, RunSummary } from "./radarContract";
import { fetchRunArtifacts, fetchRuns } from "./radarControl";

type ArtifactsState =
	| { phase: "loading" }
	| { phase: "ready"; runs: RunSummary[]; artifacts: RunArtifact[] }
	| { phase: "error"; message: string };

function ArtifactsErrorState({ message }: { message: string }) {
	return <p className="studio-error">Artifact API unavailable: {message}</p>;
}

function ArtifactsEmptyState() {
	return <p className="studio-empty">No run artifacts are available yet.</p>;
}

export default function ArtifactsPanel() {
	const [state, setState] = useState<ArtifactsState>({ phase: "loading" });
	const [activeRunId, setActiveRunId] = useState("");

	const load = useCallback(
		(runId?: string) => {
			setState({ phase: "loading" });
			fetchRuns()
				.then(async (runs) => {
					const nextRunId = runId || activeRunId || runs[0]?.run_id || "";
					const artifacts = nextRunId ? await fetchRunArtifacts(nextRunId) : [];
					setActiveRunId(nextRunId);
					setState({ phase: "ready", runs, artifacts });
				})
				.catch((err: unknown) => {
					const message = err instanceof Error ? err.message : String(err);
					setState({ phase: "error", message });
				});
		},
		[activeRunId],
	);

	useEffect(() => {
		load();
	}, [load]);

	const activeRun = useMemo(
		() =>
			state.phase === "ready"
				? state.runs.find((run) => run.run_id === activeRunId)
				: undefined,
		[activeRunId, state],
	);

	return (
		<section className="studio-view" data-testid="artifacts-view">
			<div className="studio-view__header">
				<div>
					<h2>Artifacts</h2>
					<p>
						Review validation-gated bundles, dataset cards, and dossiers before
						downloading metadata fixtures.
					</p>
				</div>
				<button
					type="button"
					className="radar-btn"
					onClick={() => load(activeRunId)}
				>
					Refresh
				</button>
			</div>

			{state.phase === "loading" ? (
				<p className="studio-empty">Loading artifacts...</p>
			) : null}
			{state.phase === "error" ? (
				<ArtifactsErrorState message={state.message} />
			) : null}
			{state.phase === "ready" ? (
				<div className="artifact-layout">
					<section className="radar-panel">
						<h3 className="radar-panel__title">Run</h3>
						{state.runs.length > 0 ? (
							<div className="run-picker" data-testid="artifact-run-picker">
								{state.runs.map((run) => (
									<button
										type="button"
										key={run.run_id}
										className={run.run_id === activeRunId ? "is-active" : ""}
										onClick={() => load(run.run_id)}
									>
										{run.run_id}
									</button>
								))}
							</div>
						) : (
							<ArtifactsEmptyState />
						)}
						{activeRun ? (
							<table className="radar-table">
								<tbody>
									<tr>
										<td>Scenario</td>
										<td>{activeRun.config.scenario_label}</td>
									</tr>
									<tr>
										<td>Seed</td>
										<td>{activeRun.config.seed}</td>
									</tr>
									<tr>
										<td>Hash</td>
										<td>{activeRun.config.scenario_hash}</td>
									</tr>
									<tr>
										<td>Validation</td>
										<td>{activeRun.validation.tier}</td>
									</tr>
								</tbody>
							</table>
						) : null}
					</section>

					<section className="radar-panel artifact-layout__wide">
						<h3 className="radar-panel__title">Available Artifacts</h3>
						{state.artifacts.length > 0 ? (
							<div className="artifact-grid">
								{state.artifacts.map((artifact) => (
									<article className="artifact-tile" key={artifact.id}>
										<div className="artifact-tile__head">
											<strong>{artifact.label}</strong>
											<span
												className="job-status-badge"
												data-status={artifact.ready ? "completed" : "queued"}
											>
												{artifact.ready ? "ready" : "pending"}
											</span>
										</div>
										<p>
											{artifact.requires_validation
												? "Validation gate required"
												: "No validation gate"}
										</p>
										<a
											className="radar-btn radar-btn--go"
											href={artifact.download_path}
										>
											Download {artifact.kind}
										</a>
									</article>
								))}
							</div>
						) : (
							<ArtifactsEmptyState />
						)}
					</section>

					<section className="radar-panel artifact-layout__wide">
						<h3 className="radar-panel__title">Review Language</h3>
						<p className="studio-copy">
							Artifact reports should describe public-proxy assumptions,
							uncertainty, validation tier, seed, and scenario hash. They should
							not describe exported fixtures as measured truth or
							proprietary-equivalent behavior.
						</p>
					</section>
				</div>
			) : null}
		</section>
	);
}
