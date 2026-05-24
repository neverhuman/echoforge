import { useEffect, useMemo, useState } from "react";
import ConnectionBanner from "./ConnectionBanner";
import MicroDopplerWaterfall from "./MicroDopplerWaterfall";
import PpiScope from "./PpiScope";
import RangeDopplerMap from "./RangeDopplerMap";
import { RadarSocket } from "./radarSocket";
import ScenarioControlPanel from "./ScenarioControlPanel";
import TelemetryPanel from "./TelemetryPanel";
import TracksPanel from "./TracksPanel";
import { useRadarSnapshot } from "./useRadarStore";
import "./radar.css";

type LayoutPreset = "expert" | "attribution" | "signal" | "benchmark";

const LAYOUT_PRESETS: Array<[LayoutPreset, string]> = [
	["expert", "Expert Live"],
	["attribution", "Attribution"],
	["signal", "Signal Lab"],
	["benchmark", "Benchmark Replay"],
];

/** The realtime radar-operator console — four live displays fed by the
 *  `/ws/radar` WebSocket stream. */
export default function RadarConsole() {
	const {
		artifact,
		backpressure,
		lifecycle,
		scanSeq,
		session,
		status,
		validation,
	} = useRadarSnapshot();
	const [layout, setLayout] = useState<LayoutPreset>("expert");

	useEffect(() => {
		const socket = new RadarSocket();
		socket.connect();
		return () => socket.close();
	}, []);

	const eventRows = useMemo(
		() =>
			[
				lifecycle
					? `Lifecycle ${lifecycle.phase} for ${lifecycle.run_id}`
					: null,
				validation
					? `Validation ${validation.grade} at ${validation.tier}`
					: null,
				artifact
					? `Artifact ${artifact.kind} ready at ${artifact.download_path}`
					: null,
				backpressure
					? `Backpressure ${backpressure.dropped_frames} dropped frames`
					: null,
				status ? `${status.code}: ${status.message}` : null,
			].filter((row): row is string => row !== null),
		[artifact, backpressure, lifecycle, status, validation],
	);

	return (
		<div className="radar-console" data-testid="radar-console">
			<ConnectionBanner />
			<div className="live-toolbar">
				<div className="studio-segment">
					{LAYOUT_PRESETS.map(([id, label]) => (
						<button
							type="button"
							key={id}
							className={layout === id ? "is-active" : ""}
							onClick={() => setLayout(id)}
							data-testid={`live-preset-${id}`}
						>
							{label}
						</button>
					))}
				</div>
			</div>
			<div className="studio-status-strip">
				<span>Run: {lifecycle?.run_id ?? "waiting"}</span>
				<span>Validation: {validation?.grade ?? "pending"}</span>
				<span>
					FPS: {session?.frame_rate_hz ?? 0} / scans {scanSeq}
				</span>
				<span>Dropped: {backpressure?.dropped_frames ?? 0}</span>
			</div>
			<div className={`radar-grid radar-grid--${layout}`}>
				<div className="radar-cell">
					<div className="radar-cell__title">Plan-Position Indicator</div>
					<div className="radar-cell__body">
						<PpiScope />
					</div>
				</div>
				<div className="radar-cell">
					<div className="radar-cell__title">Range · Doppler</div>
					<div className="radar-cell__body">
						<RangeDopplerMap />
					</div>
				</div>
				<div className="radar-cell">
					<div className="radar-cell__title">Micro-Doppler Spectrogram</div>
					<div className="radar-cell__body">
						<MicroDopplerWaterfall />
					</div>
				</div>
				<div className="radar-side">
					<ScenarioControlPanel />
					<TelemetryPanel />
					<TracksPanel />
					<section className="radar-panel" data-testid="event-log">
						<h3 className="radar-panel__title">Event Log</h3>
						{eventRows.length > 0 ? (
							<ol className="event-log">
								{eventRows.map((row) => (
									<li key={row}>{row}</li>
								))}
							</ol>
						) : (
							<p className="radar-empty">Awaiting stream events</p>
						)}
					</section>
				</div>
			</div>
		</div>
	);
}
