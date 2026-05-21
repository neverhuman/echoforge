import { useCallback, useState } from "react";

const ENDPOINTS = [
	["GET", "/api/runs", "List run receipts and validation summaries"],
	["GET", "/api/runs/{id}", "Inspect one run receipt"],
	[
		"POST",
		"/api/runs/{id}/replay",
		"Replay a run seed through the live radar stream",
	],
	["POST", "/api/runs/{id}/archive", "Archive a run without hard deletion"],
	["POST", "/api/runs/{id}/restore", "Restore an archived run"],
	[
		"POST",
		"/api/runs/{id}/duplicate",
		"Duplicate a run into a queued replay or Monte Carlo request",
	],
	["GET", "/api/runs/{id}/artifacts", "List validation-gated artifacts"],
	["GET", "/api/jobs", "List ML processing jobs"],
	["POST", "/api/jobs", "Queue a processing job"],
	["POST", "/api/jobs/{id}/cancel", "Cancel a queued or running job"],
	["WS", "/ws/radar", "Control JSON frames plus quantized binary scan frames"],
] as const;

const HEADLESS_EXAMPLE = `rtk cargo run -p echoforge-studio --locked
rtk curl http://127.0.0.1:8080/api/runs
rtk curl -X POST http://127.0.0.1:8080/api/runs/run-shahed-ingress-00/duplicate \\
  -H 'content-type: application/json' \\
  -d '{"seed":2026052101,"mode":"monte_carlo"}'`;

const WEBSOCKET_EXAMPLE = `const socket = new WebSocket('ws://127.0.0.1:8080/ws/radar');
socket.binaryType = 'arraybuffer';
socket.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const control = JSON.parse(event.data);
    console.log(control.type, control);
  } else {
    console.log('binary scan frame', event.data.byteLength);
  }
};`;

function CopyButton({ text, label }: { text: string; label: string }) {
	const [copied, setCopied] = useState(false);
	const copy = useCallback(() => {
		navigator.clipboard
			?.writeText(text)
			.then(() => {
				setCopied(true);
				window.setTimeout(() => setCopied(false), 1200);
			})
			.catch(() => setCopied(false));
	}, [text]);

	return (
		<button type="button" className="radar-btn" onClick={copy}>
			{copied ? "Copied" : label}
		</button>
	);
}

export default function ApiHeadlessPanel() {
	return (
		<section className="studio-view" data-testid="api-headless-view">
			<div className="studio-view__header">
				<div>
					<h2>API / Headless</h2>
					<p>
						Same-origin REST and WebSocket contracts for reproducible Studio
						automation.
					</p>
				</div>
				<CopyButton text={HEADLESS_EXAMPLE} label="Copy CLI" />
			</div>

			<div className="api-layout">
				<section className="radar-panel api-layout__wide">
					<h3 className="radar-panel__title">REST Surface</h3>
					<table className="radar-table">
						<tbody>
							{ENDPOINTS.map(([method, path, detail]) => (
								<tr key={`${method}-${path}`}>
									<td>{method}</td>
									<td>{path}</td>
									<td>{detail}</td>
								</tr>
							))}
						</tbody>
					</table>
				</section>

				<section className="radar-panel">
					<div className="panel-head-row">
						<h3 className="radar-panel__title">Headless CLI</h3>
						<CopyButton text={HEADLESS_EXAMPLE} label="Copy" />
					</div>
					<pre className="code-surface">{HEADLESS_EXAMPLE}</pre>
				</section>

				<section className="radar-panel">
					<div className="panel-head-row">
						<h3 className="radar-panel__title">WebSocket Stream</h3>
						<CopyButton text={WEBSOCKET_EXAMPLE} label="Copy" />
					</div>
					<pre className="code-surface">{WEBSOCKET_EXAMPLE}</pre>
				</section>

				<section className="radar-panel api-layout__wide">
					<h3 className="radar-panel__title">Export Boundary</h3>
					<p className="studio-copy">
						Clients should persist run ID, seed, scenario hash, validation tier,
						source cards, uncertainty statement, and artifact download paths
						with every report. Headless usage does not relax the public-proxy
						boundary.
					</p>
				</section>
			</div>
		</section>
	);
}
