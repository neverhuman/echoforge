import { useCallback, useEffect, useMemo, useState } from "react";
import type { RunQueueSummary, RunSummary } from "./radarContract";
import {
	archiveRun,
	duplicateRun,
	fetchRunQueueSummary,
	fetchRuns,
	replayRun,
	restoreRun,
} from "./radarControl";

type LoadState =
	| { phase: "loading" }
	| { phase: "ready"; runs: RunSummary[]; queue: RunQueueSummary | null }
	| { phase: "error"; message: string };

type StatusFilter = "all" | "active" | "queued" | "validated" | "archived";
type ViewMode = "table" | "cards";

const FILTERS: StatusFilter[] = [
	"all",
	"active",
	"queued",
	"validated",
	"archived",
];

function RunsEmptyState() {
	return <p className="studio-empty">No run receipts are available yet.</p>;
}

function RunsErrorState({ message }: { message: string }) {
	return <p className="studio-error">Run history unavailable: {message}</p>;
}

function statusMatches(run: RunSummary, filter: StatusFilter): boolean {
	if (filter === "all") return true;
	if (filter === "active") return run.status !== "archived";
	return run.status === filter;
}

function searchMatches(run: RunSummary, query: string): boolean {
	const haystack = [
		run.run_id,
		run.status,
		run.config.scenario_label,
		run.config.scenario_id,
		String(run.config.seed),
		run.config.scenario_hash,
		run.validation.tier,
		run.validation.grade,
	]
		.join(" ")
		.toLowerCase();
	return haystack.includes(query.toLowerCase());
}

function sortRuns(runs: RunSummary[]): RunSummary[] {
	return [...runs].sort((left, right) =>
		right.created_utc.localeCompare(left.created_utc),
	);
}

export default function RunsPanel() {
	const [state, setState] = useState<LoadState>({ phase: "loading" });
	const [busyRunId, setBusyRunId] = useState<string | null>(null);
	const [activeRunId, setActiveRunId] = useState<string | null>(null);
	const [markedRunIds, setMarkedRunIds] = useState<Set<string>>(
		() => new Set(),
	);
	const [filter, setFilter] = useState<StatusFilter>("active");
	const [query, setQuery] = useState("");
	const [viewMode, setViewMode] = useState<ViewMode>("table");

	const loadRuns = useCallback(() => {
		setState({ phase: "loading" });
		Promise.all([fetchRuns(), fetchRunQueueSummary().catch(() => null)])
			.then(([runs, queue]) => {
				const sorted = sortRuns(runs);
				setState({ phase: "ready", runs: sorted, queue });
				setActiveRunId((current) =>
					current && sorted.some((run) => run.run_id === current)
						? current
						: (sorted[0]?.run_id ?? null),
				);
			})
			.catch((err: unknown) => {
				const message = err instanceof Error ? err.message : String(err);
				setState({ phase: "error", message });
			});
	}, []);

	useEffect(() => {
		loadRuns();
	}, [loadRuns]);

	const runs = state.phase === "ready" ? state.runs : [];
	const filteredRuns = useMemo(
		() =>
			runs.filter(
				(run) => statusMatches(run, filter) && searchMatches(run, query),
			),
		[filter, query, runs],
	);
	const activeRun = useMemo(
		() =>
			runs.find((run) => run.run_id === activeRunId) ?? filteredRuns[0] ?? null,
		[activeRunId, filteredRuns, runs],
	);
	const compareRuns = useMemo(
		() => runs.filter((run) => markedRunIds.has(run.run_id)).slice(0, 4),
		[markedRunIds, runs],
	);

	const withBusyRun = useCallback(
		(runId: string, action: () => Promise<unknown>) => {
			setBusyRunId(runId);
			action()
				.then(() => loadRuns())
				.catch((err: unknown) => {
					const message = err instanceof Error ? err.message : String(err);
					setState({ phase: "error", message });
				})
				.finally(() => setBusyRunId(null));
		},
		[loadRuns],
	);

	const toggleMarked = useCallback((runId: string) => {
		setMarkedRunIds((current) => {
			const next = new Set(current);
			if (next.has(runId)) {
				next.delete(runId);
			} else {
				next.add(runId);
			}
			return next;
		});
	}, []);

	const batchArchive = useCallback(() => {
		const ids = Array.from(markedRunIds);
		if (ids.length === 0) return;
		setBusyRunId("batch");
		Promise.all(
			ids.map((id) =>
				archiveRun(id, { reason: "batch archive from Studio UI" }),
			),
		)
			.then(() => {
				setMarkedRunIds(new Set());
				loadRuns();
			})
			.catch((err: unknown) => {
				const message = err instanceof Error ? err.message : String(err);
				setState({ phase: "error", message });
			})
			.finally(() => setBusyRunId(null));
	}, [loadRuns, markedRunIds]);

	const apiExample = activeRun
		? `GET /api/runs/${activeRun.run_id}\nPOST /api/runs/${activeRun.run_id}/replay\nPOST /api/runs/${activeRun.run_id}/archive`
		: "GET /api/runs";
	const cliExample = activeRun
		? `rtk curl http://127.0.0.1:8080/api/runs/${activeRun.run_id}`
		: "rtk curl http://127.0.0.1:8080/api/runs";

	return (
		<section className="studio-view" data-testid="runs-view">
			<div className="studio-view__header">
				<div>
					<h2>Runs & Queue</h2>
					<p>
						Search, inspect, replay, duplicate, archive, restore, compare, and
						download validation-gated artifacts.
					</p>
				</div>
				<div className="job-detail__actions">
					<button type="button" className="radar-btn" onClick={loadRuns}>
						Refresh
					</button>
					<button
						type="button"
						className="radar-btn"
						onClick={batchArchive}
						disabled={markedRunIds.size === 0 || busyRunId === "batch"}
						data-testid="run-batch-archive"
					>
						Archive marked
					</button>
				</div>
			</div>

			<div className="run-toolbar">
				<label className="radar-field run-toolbar__search">
					<span>Search</span>
					<input
						type="search"
						value={query}
						onChange={(event) => setQuery(event.target.value)}
						data-testid="run-search"
					/>
				</label>
				<div className="studio-segment">
					{FILTERS.map((entry) => (
						<button
							type="button"
							key={entry}
							className={filter === entry ? "is-active" : ""}
							onClick={() => setFilter(entry)}
							data-testid={`run-filter-${entry}`}
						>
							{entry}
						</button>
					))}
				</div>
				<div className="studio-segment">
					<button
						type="button"
						className={viewMode === "table" ? "is-active" : ""}
						onClick={() => setViewMode("table")}
					>
						Table
					</button>
					<button
						type="button"
						className={viewMode === "cards" ? "is-active" : ""}
						onClick={() => setViewMode("cards")}
					>
						Cards
					</button>
				</div>
			</div>

			{state.phase === "loading" ? (
				<p className="studio-empty">Loading run history...</p>
			) : null}
			{state.phase === "error" ? (
				<RunsErrorState message={state.message} />
			) : null}
			{state.phase === "ready" && runs.length === 0 ? <RunsEmptyState /> : null}
			{state.phase === "ready" && runs.length > 0 ? (
				<div className="runs-layout">
					<section className="radar-panel runs-layout__list">
						<div className="run-summary-row">
							<span>{state.queue?.total ?? runs.length} total</span>
							<span>
								{state.queue?.queued ??
									runs.filter((run) => run.status === "queued").length}{" "}
								queued
							</span>
							<span>
								{state.queue?.archived ??
									runs.filter((run) => run.status === "archived").length}{" "}
								archived
							</span>
							<span>
								{state.queue?.export_ready ??
									runs.filter((run) => run.validation.export_gate_passed)
										.length}{" "}
								export-ready
							</span>
						</div>
						{filteredRuns.length === 0 ? <RunsEmptyState /> : null}
						{viewMode === "table" && filteredRuns.length > 0 ? (
							<div className="run-table-wrap">
								<table className="radar-table">
									<thead>
										<tr>
											<th>Mark</th>
											<th>Run</th>
											<th>Status</th>
											<th>Scenario</th>
											<th>Seed</th>
											<th>Validation</th>
											<th>Actions</th>
										</tr>
									</thead>
									<tbody>
										{filteredRuns.map((run) => (
											<tr
												key={run.run_id}
												className={
													run.run_id === activeRun?.run_id ? "is-active" : ""
												}
											>
												<td>
													<input
														type="checkbox"
														checked={markedRunIds.has(run.run_id)}
														onChange={() => toggleMarked(run.run_id)}
														aria-label={`mark run ${run.run_id}`}
													/>
												</td>
												<td>
													<button
														type="button"
														className="run-link-button"
														onClick={() => setActiveRunId(run.run_id)}
													>
														{run.run_id}
													</button>
												</td>
												<td>
													<span
														className="job-status-badge"
														data-status={run.status}
													>
														{run.status}
													</span>
												</td>
												<td>{run.config.scenario_label}</td>
												<td>{run.config.seed}</td>
												<td>{run.validation.tier}</td>
												<td>
													<div className="run-action-row">
														<button
															type="button"
															className="radar-btn"
															disabled={busyRunId === run.run_id}
															onClick={() =>
																withBusyRun(run.run_id, () =>
																	replayRun(run.run_id),
																)
															}
															data-testid={`replay-${run.run_id}`}
														>
															Replay
														</button>
														<button
															type="button"
															className="radar-btn"
															disabled={busyRunId === run.run_id}
															onClick={() =>
																withBusyRun(run.run_id, () =>
																	duplicateRun(run.run_id, { mode: "replay" }),
																)
															}
															data-testid={`duplicate-${run.run_id}`}
														>
															Duplicate
														</button>
														{run.status === "archived" ? (
															<button
																type="button"
																className="radar-btn"
																disabled={busyRunId === run.run_id}
																onClick={() =>
																	withBusyRun(run.run_id, () =>
																		restoreRun(run.run_id),
																	)
																}
																data-testid={`restore-${run.run_id}`}
															>
																Restore
															</button>
														) : (
															<button
																type="button"
																className="radar-btn"
																disabled={busyRunId === run.run_id}
																onClick={() =>
																	withBusyRun(run.run_id, () =>
																		archiveRun(run.run_id, {
																			reason: "archive from Studio UI",
																		}),
																	)
																}
																data-testid={`archive-${run.run_id}`}
															>
																Archive
															</button>
														)}
													</div>
												</td>
											</tr>
										))}
									</tbody>
								</table>
							</div>
						) : null}

						{viewMode === "cards" && filteredRuns.length > 0 ? (
							<div className="run-card-grid">
								{filteredRuns.map((run) => (
									<button
										type="button"
										className={`run-card${run.run_id === activeRun?.run_id ? " is-active" : ""}`}
										key={run.run_id}
										onClick={() => setActiveRunId(run.run_id)}
									>
										<span className="job-status-badge" data-status={run.status}>
											{run.status}
										</span>
										<strong>{run.run_id}</strong>
										<span>{run.config.scenario_label}</span>
										<span>{run.validation.grade}</span>
									</button>
								))}
							</div>
						) : null}
					</section>

					<section
						className="radar-panel runs-layout__detail"
						data-testid="run-detail"
					>
						<h3 className="radar-panel__title">Inspect</h3>
						{activeRun ? (
							<>
								<table className="radar-table">
									<tbody>
										<tr>
											<td>Run</td>
											<td>{activeRun.run_id}</td>
										</tr>
										<tr>
											<td>Source card</td>
											<td>{activeRun.config.object_source_card}</td>
										</tr>
										<tr>
											<td>Seed</td>
											<td>{activeRun.config.seed}</td>
										</tr>
										<tr>
											<td>Scenario hash</td>
											<td>{activeRun.config.scenario_hash}</td>
										</tr>
										<tr>
											<td>Uncertainty</td>
											<td>{activeRun.validation.uncertainty_statement}</td>
										</tr>
									</tbody>
								</table>
								<div className="run-download-list">
									{activeRun.artifacts.map((artifact) => (
										<a
											key={artifact.id}
											className="run-download"
											href={artifact.download_path}
										>
											{artifact.kind}
										</a>
									))}
								</div>
								<pre className="code-surface" data-testid="run-api-command">
									{apiExample}
								</pre>
								<pre className="code-surface">{cliExample}</pre>
							</>
						) : (
							<RunsEmptyState />
						)}
					</section>

					{compareRuns.length > 1 ? (
						<section
							className="radar-panel runs-layout__wide"
							data-testid="run-compare"
						>
							<h3 className="radar-panel__title">Compare Marked</h3>
							<table className="radar-table">
								<tbody>
									{compareRuns.map((run) => (
										<tr key={run.run_id}>
											<td>{run.run_id}</td>
											<td>{run.config.seed}</td>
											<td>{run.config.scenario_hash}</td>
											<td>{run.validation.grade}</td>
										</tr>
									))}
								</tbody>
							</table>
						</section>
					) : null}
				</div>
			) : null}
		</section>
	);
}
