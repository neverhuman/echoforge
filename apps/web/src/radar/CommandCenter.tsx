import { useCallback, useEffect, useMemo, useState } from "react";
import type { JobSummary, RunQueueSummary, RunSummary } from "./radarContract";
import { fetchJobs, fetchRunQueueSummary, fetchRuns } from "./radarControl";

type CommandState =
	| { phase: "loading" }
	| {
			phase: "ready";
			runs: RunSummary[];
			jobs: JobSummary[];
			queue: RunQueueSummary | null;
	  };

interface CommandModel {
	newest?: RunSummary;
	runningJobs: JobSummary[];
	totalRuns: number;
	activeRuns: number;
	archivedRuns: number;
	exportReady: number;
}

function latestRun(runs: RunSummary[]): RunSummary | undefined {
	return [...runs].sort((left, right) =>
		right.created_utc.localeCompare(left.created_utc),
	)[0];
}

function countRuns(runs: RunSummary[], status: string): number {
	return runs.filter((run) => run.status === status).length;
}

function CommandCenterLoadingState() {
	return <p className="studio-empty">Loading Studio control-plane state...</p>;
}

function buildCommandModel(
	runs: RunSummary[],
	jobs: JobSummary[],
	queue: RunQueueSummary | null,
): CommandModel {
	const runningJobs = jobs.filter(
		(job) => job.status === "running" || job.status === "queued",
	);
	return {
		newest: latestRun(runs),
		runningJobs,
		totalRuns: queue?.total ?? runs.length,
		activeRuns:
			queue?.active ?? runs.filter((run) => run.status !== "archived").length,
		archivedRuns: queue?.archived ?? countRuns(runs, "archived"),
		exportReady:
			queue?.export_ready ??
			runs.filter((run) => run.validation.export_gate_passed).length,
	};
}

export default function CommandCenter() {
	const [state, setState] = useState<CommandState>({ phase: "loading" });

	const load = useCallback(() => {
		setState({ phase: "loading" });
		Promise.all([
			fetchRuns().catch(() => [] as RunSummary[]),
			fetchJobs().catch(() => [] as JobSummary[]),
			fetchRunQueueSummary().catch(() => null),
		]).then(([runs, jobs, queue]) => {
			setState({ phase: "ready", runs, jobs, queue });
		});
	}, []);

	useEffect(() => {
		load();
	}, [load]);

	const model = useMemo(
		() =>
			state.phase === "ready"
				? buildCommandModel(state.runs, state.jobs, state.queue)
				: undefined,
		[state],
	);

	return (
		<section
			className="studio-view command-center"
			data-testid="command-center"
		>
			<div className="studio-view__header">
				<div>
					<h2>Command Center</h2>
					<p>
						Same-origin Studio status, queue posture, validation evidence, and
						strict-open export readiness.
					</p>
				</div>
				<button
					type="button"
					className="radar-btn"
					onClick={load}
					data-testid="command-refresh"
				>
					Refresh
				</button>
			</div>

			{state.phase === "loading" ? <CommandCenterLoadingState /> : null}
			{model ? (
				<>
					<div className="command-stat-grid">
						<section className="command-stat">
							<span>Total runs</span>
							<strong>{model.totalRuns}</strong>
						</section>
						<section className="command-stat">
							<span>Active runs</span>
							<strong>{model.activeRuns}</strong>
						</section>
						<section className="command-stat">
							<span>Export-ready</span>
							<strong>{model.exportReady}</strong>
						</section>
						<section className="command-stat">
							<span>Archived</span>
							<strong>{model.archivedRuns}</strong>
						</section>
					</div>

					<div className="command-layout">
						<section className="radar-panel">
							<h3 className="radar-panel__title">Current Evidence Boundary</h3>
							<div className="evidence-stack">
								<span className="job-chip job-chip--gold">
									public-proxy provenance
								</span>
								<span className="job-chip">uncertainty-scored artifacts</span>
								<span className="job-chip">archive-first deletion model</span>
								<span className="job-chip">not measured truth</span>
							</div>
							<p className="studio-copy">
								Studio exports carry source cards, seed, scenario hash,
								validation tier, leakage guard status, and an uncertainty
								statement before download links are useful.
							</p>
						</section>

						<section className="radar-panel">
							<h3 className="radar-panel__title">Newest Run</h3>
							{model.newest ? (
								<table className="radar-table">
									<tbody>
										<tr>
											<td>Run</td>
											<td>{model.newest.run_id}</td>
										</tr>
										<tr>
											<td>Scenario</td>
											<td>{model.newest.config.scenario_label}</td>
										</tr>
										<tr>
											<td>Seed</td>
											<td>{model.newest.config.seed}</td>
										</tr>
										<tr>
											<td>Tier</td>
											<td>{model.newest.validation.tier}</td>
										</tr>
									</tbody>
								</table>
							) : (
								<p className="studio-empty">
									No runs are available from the preview service.
								</p>
							)}
						</section>

						<section className="radar-panel">
							<h3 className="radar-panel__title">Queue</h3>
							{model.runningJobs.length > 0 ? (
								<div className="job-list">
									{model.runningJobs.slice(0, 4).map((job) => (
										<div className="job-card job-card--static" key={job.job_id}>
											<div className="job-card__head">
												<strong>{job.job_id}</strong>
												<span
													className="job-status-badge"
													data-status={job.status}
												>
													{job.status}
												</span>
											</div>
											<p>{job.message}</p>
										</div>
									))}
								</div>
							) : (
								<p className="studio-empty">No active ML processing jobs.</p>
							)}
						</section>
					</div>
				</>
			) : null}
		</section>
	);
}
