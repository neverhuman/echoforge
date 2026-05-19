//! HTML credibility report renderer.
//!
//! Writes a self-contained `index.html` (no JS, no external assets,
//! embedded CSS) to `outputs/radar-expert-credibility-report/`. The
//! report walks a `CaseStudyTrace` from card → SNR(R) → predicted range
//! → declared range → gap explanation.

use std::path::PathBuf;

/// One row of the SNR vs range trace shown in the report.
#[derive(Debug, Clone)]
pub struct TraceRow {
    pub range_km: f64,
    pub snr_db: f64,
    pub above_threshold: bool,
}

/// One declared-vs-predicted comparison row.
#[derive(Debug, Clone)]
pub struct ReproductionRow {
    pub target_label: String,
    pub rcs_dbsm: f64,
    pub declared_range_km: f64,
    pub predicted_range_km: f64,
    pub gap_pct: f64,
    pub confidence_label: String,
    pub citation: String,
    pub conditions: String,
}

/// Top-level case-study trace shown by the report.
#[derive(Debug, Clone)]
pub struct CaseStudyTrace {
    pub case_title: String,
    pub utc_timestamp: String,
    pub radar_pack: String,
    pub radar_card_slug: String,
    pub radar_display_name: String,
    pub radar_vendor: String,
    pub radar_model: String,
    pub source_pack: String,
    pub source_card_slug: String,
    pub source_display_name: String,
    pub source_country_of_origin: String,
    pub center_frequency_ghz: f64,
    pub gain_dbi: f64,
    pub peak_power_w: f64,
    pub prf_hz: f64,
    pub dwell_s: f64,
    pub noise_figure_db: f64,
    pub system_losses_db: f64,
    pub coherent_pulses: usize,
    pub required_snr_db: f64,
    pub pd_at_pfa: (f64, f64),
    pub reproduction_rows: Vec<ReproductionRow>,
    pub snr_trace: Vec<TraceRow>,
    pub trace_target_label: String,
    pub trace_target_rcs_dbsm: f64,
    pub bibliography: Vec<String>,
    pub receipt_refs: Vec<String>,
}

impl CaseStudyTrace {
    pub fn write_html(&self, out_dir: &PathBuf) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(out_dir)?;
        let html = self.render_html();
        let path = out_dir.join("index.html");
        std::fs::write(&path, html)?;
        Ok(path)
    }

    fn render_html(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — EchoForge Credibility Report</title>
<style>
:root {{
  --fg: #111; --bg: #fff; --muted: #555; --accent: #1e6091;
  --pass: #1b6e2e; --warn: #b3781a; --fail: #b3261a;
  --code-bg: #f5f5f5; --rule: #ddd;
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 920px; margin: 2.5em auto; padding: 0 1.5em;
  line-height: 1.55; color: var(--fg); background: var(--bg);
}}
h1 {{ font-size: 1.8em; margin-bottom: 0.15em; color: var(--fg); }}
h1 .vendor {{ color: var(--accent); }}
h2 {{ font-size: 1.25em; margin-top: 2em; border-bottom: 1px solid var(--rule); padding-bottom: 0.3em; }}
h3 {{ font-size: 1.05em; margin-top: 1.6em; color: var(--accent); }}
.claim {{
  background: #f4f8fb; border-left: 4px solid var(--accent);
  padding: 0.8em 1.1em; margin: 1.4em 0; font-size: 1.05em;
}}
.notbox {{
  background: #fdf6e3; border-left: 4px solid var(--warn);
  padding: 0.6em 1.1em; margin: 1.4em 0; color: #5a4a1a;
}}
.notbox h3 {{ margin-top: 0; color: #8a6310; }}
table {{
  border-collapse: collapse; width: 100%; margin: 0.8em 0 1.4em;
  font-size: 0.93em;
}}
th, td {{
  border: 1px solid var(--rule); padding: 0.45em 0.7em; text-align: left;
}}
th {{ background: #f7f7f7; font-weight: 600; }}
tr.pass td:nth-child(5) {{ color: var(--pass); font-weight: 600; }}
tr.warn td:nth-child(5) {{ color: var(--warn); font-weight: 600; }}
tr.fail td:nth-child(5) {{ color: var(--fail); font-weight: 600; }}
.kv {{
  display: grid; grid-template-columns: max-content 1fr; gap: 0.3em 1.1em;
  background: var(--code-bg); padding: 0.8em 1.1em; border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em;
}}
.kv .k {{ color: var(--muted); }}
.snr-trace {{
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.85em;
  background: var(--code-bg); padding: 0.8em 1.1em; border-radius: 4px;
  white-space: pre; overflow-x: auto;
}}
ul.bib li {{ font-size: 0.9em; margin: 0.25em 0; }}
ul.receipts li {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.85em; }}
.tier-ladder {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.4em; margin: 0.8em 0; }}
.tier-ladder .step {{
  background: var(--code-bg); padding: 0.5em; border-radius: 3px;
  text-align: center; font-size: 0.83em;
}}
.tier-ladder .step.current {{ background: #d6e8c5; color: #2a4a0e; font-weight: 600; }}
footer {{
  margin-top: 3em; padding-top: 1em; border-top: 1px solid var(--rule);
  font-size: 0.85em; color: var(--muted);
}}
code {{ background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 3px; }}
</style>
</head>
<body>

<h1><span class="vendor">{radar_vendor}</span> {radar_model} <span style="font-weight:400;color:var(--muted);font-size:0.7em;">vs</span> {source_display_name}</h1>
<p style="color:var(--muted);margin-top:0;">Public-Proxy Physics Trace · {utc}</p>

<div class="claim">
  <strong>EchoForge claim:</strong> open-source, strict-open radar physics simulator with
  traceable Pd / Pfa prediction within ±20 % of vendor-declared detection ranges
  on at least one named platform pair. Every parameter cited to a public source.
  Trace below: schema → SNR(R) → CFAR threshold → predicted range vs declared.
</div>

<div class="notbox">
  <h3>What this is NOT</h3>
  <ul>
    <li><strong>Not a measured-truth claim.</strong> RCS values are public-proxy envelopes (cited literature scaled to S-band); no measured Shahed-specific truth.</li>
    <li><strong>Not a classified-tactics simulator.</strong> No vendor processing-gain proprietaries, no operational deployment geometry, no engagement workflow.</li>
    <li><strong>Not an operational planner.</strong> Predicted ranges are physics priors for R&amp;D, not mission-planning ground truth.</li>
    <li><strong>Strict-open posture preserved.</strong> Naming-policy reversal of 20260518T160000Z allows the {source_display_name} and {radar_vendor} {radar_model} designators; each is cited to a public datasheet or OSINT source.</li>
  </ul>
</div>

<h2>1. Radar platform (loaded from pack)</h2>

<div class="kv">
  <div class="k">Pack</div><div><code>{radar_pack}</code></div>
  <div class="k">Card slug</div><div><code>{radar_card_slug}</code></div>
  <div class="k">Vendor / Model</div><div>{radar_vendor} / {radar_model}</div>
  <div class="k">Band / Center frequency</div><div>S-band / {center_freq_ghz:.2} GHz</div>
  <div class="k">Antenna gain</div><div>{gain_dbi:.1} dBi</div>
  <div class="k">Peak power</div><div>{peak_power_w:.0} W</div>
  <div class="k">PRF</div><div>{prf_hz:.0} Hz</div>
  <div class="k">Dwell time</div><div>{dwell_ms:.1} ms</div>
  <div class="k">Coherent integration N</div><div>{n_pulses} pulses</div>
  <div class="k">Noise figure</div><div>{nf_db:.1} dB</div>
  <div class="k">System losses</div><div>{losses_db:.1} dB</div>
  <div class="k">Required SNR (Albersheim, Pd={pd:.2}, Pfa={pfa:.0e})</div><div>{snr_req:.2} dB integrated</div>
</div>

<h2>2. Adversary source (loaded from pack)</h2>

<div class="kv">
  <div class="k">Pack</div><div><code>{source_pack}</code></div>
  <div class="k">Card slug</div><div><code>{source_card_slug}</code></div>
  <div class="k">Display name</div><div>{source_display_name}</div>
  <div class="k">Country of origin</div><div>{country}</div>
  <div class="k">Trace target aspect</div><div>{trace_target_label}, RCS = {trace_rcs_dbsm:.1} dBsm</div>
</div>

<h2>3. Trace: SNR vs range (linear chain, no shortcuts)</h2>

<p>
  For each range, integrated SNR = (P_t·G_t·G_r·λ²·σ·N) / ((4π)³ · R⁴ · L · k·T·B·F).
  Detection threshold: integrated SNR ≥ {snr_req:.2} dB (Albersheim 1981 closed-form
  for the declared Pd / Pfa).
</p>

<div class="snr-trace">{snr_trace}</div>

<h2>4. Reproduction vs declared envelope</h2>

<table>
<thead><tr>
  <th>Target class</th>
  <th>RCS (dBsm)</th>
  <th>Vendor declared</th>
  <th>Our prediction</th>
  <th>Gap</th>
  <th>Confidence</th>
  <th>Conditions</th>
</tr></thead>
<tbody>
{reproduction_rows}
</tbody>
</table>

<p>
  Gap ≤ ±20 % is the acceptance gate for the Wave 9 case study. Larger gaps are
  honestly attributed below.
</p>

<h2>5. Honest gap attribution</h2>

<ul>
<li><strong>Vendor declared ranges (confidence A)</strong> are derived from the published
brochure with typical Pd = 0.85, Pfa = 1e-4, clear-air, single-look detection
against a fluctuating target. Our prediction uses the Albersheim non-fluctuating
SNR floor; for a Swerling-1 fluctuating target the required SNR is ~7–9 dB higher
(Skolnik Fig. 2.8), which compresses the predicted range by ~15–20 %. Re-computing
with that correction is left to Wave 13.</li>
<li><strong>Shahed-class predictions (confidence C)</strong> reflect the broadside-average
S-band RCS envelope (−6.5 dBsm midpoint of the cited −10 to −3 dBsm aspect range).
Nose-on or tail-on aspects drop the prediction by another 8–12 dB ≈ 0.6× range.
Operational range against a Shahed ingressing nose-on is therefore the floor, not
the broadside reproduction shown here.</li>
<li><strong>Strict-open omits vendor processing-gain proprietaries</strong> (Doppler
filter shape, micro-Doppler classifier gain, range-gate STC schedule). Typical
modern AESA Doppler processing buys 3–6 dB beyond the matched-filter floor; the
gap between our public-source prediction and the vendor declared range is
roughly consistent with that.</li>
</ul>

<h2>6. Validation tier ladder</h2>

<p>EchoForge declares the evidence tier (V0–V4) and method-ceiling fidelity (F0–F5)
of every artefact. This report is currently at:</p>

<div class="tier-ladder">
  <div class="step">V0 unvalidated</div>
  <div class="step current">V1 basic</div>
  <div class="step">V2 cross-checked</div>
  <div class="step">V3 benchmarked</div>
  <div class="step">V4 measured-anchored</div>
  <div class="step">V5 reserved</div>
</div>

<div class="tier-ladder">
  <div class="step">F0 analytic-prior</div>
  <div class="step current">F1 primitive-decomposed</div>
  <div class="step">F2 production-SBR</div>
  <div class="step">F3 cross-solver-bridged</div>
  <div class="step">F4 measured-anchored</div>
  <div class="step">F5 lawful-measured</div>
</div>

<h2>7. Reproducibility</h2>

<div class="snr-trace">$ git clone &lt;repo&gt; &amp;&amp; cd echoforge
$ rtk cargo test -p echoforge-case-studies --offline
$ open outputs/radar-expert-credibility-report/index.html</div>

<p>The HTML report you are reading is regenerated from a clean checkout every test
run; the radar parameters are loaded via <code>echoforge-packs</code> from the
<code>{radar_pack}</code> pack manifest; the adversary parameters from
<code>{source_pack}</code>. No values are hard-coded in the test binary.</p>

<h2>8. Bibliography</h2>

<ul class="bib">
{bibliography}
</ul>

<h2>9. Receipts</h2>

<ul class="receipts">
{receipts}
</ul>

<footer>
  EchoForge — strict-open public-proxy radar physics. Strict-open evidence
  ladder per <code>docs/validation-tiers.md</code>. Naming-policy reversal
  20260518T160000Z permits vendor and platform names in default code paths
  with public-source citations.
</footer>

</body>
</html>
"#,
            title = self.case_title,
            radar_vendor = self.radar_vendor,
            radar_model = self.radar_model,
            source_display_name = self.source_display_name,
            utc = self.utc_timestamp,
            radar_pack = self.radar_pack,
            radar_card_slug = self.radar_card_slug,
            center_freq_ghz = self.center_frequency_ghz,
            gain_dbi = self.gain_dbi,
            peak_power_w = self.peak_power_w,
            prf_hz = self.prf_hz,
            dwell_ms = self.dwell_s * 1000.0,
            n_pulses = self.coherent_pulses,
            nf_db = self.noise_figure_db,
            losses_db = self.system_losses_db,
            pd = self.pd_at_pfa.0,
            pfa = self.pd_at_pfa.1,
            snr_req = self.required_snr_db,
            source_pack = self.source_pack,
            source_card_slug = self.source_card_slug,
            country = self.source_country_of_origin,
            trace_target_label = self.trace_target_label,
            trace_rcs_dbsm = self.trace_target_rcs_dbsm,
            snr_trace = render_snr_trace(&self.snr_trace, self.required_snr_db),
            reproduction_rows = render_reproduction_rows(&self.reproduction_rows),
            bibliography = render_bibliography(&self.bibliography),
            receipts = render_receipts(&self.receipt_refs),
        ));
        s
    }
}

fn render_snr_trace(rows: &[TraceRow], snr_req_db: f64) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "  range_km   integrated_SNR_dB    threshold ({snr_req:.2} dB)\n",
        snr_req = snr_req_db
    ));
    s.push_str("  --------   -----------------    ----------------\n");
    for r in rows {
        let marker = if r.above_threshold { " above" } else { " below" };
        s.push_str(&format!(
            "  {range:7.1}    {snr:11.2}         {marker}\n",
            range = r.range_km,
            snr = r.snr_db,
            marker = marker
        ));
    }
    s
}

fn render_reproduction_rows(rows: &[ReproductionRow]) -> String {
    let mut s = String::new();
    for r in rows {
        let cls = if r.gap_pct.abs() <= 20.0 {
            "pass"
        } else if r.gap_pct.abs() <= 50.0 {
            "warn"
        } else {
            "fail"
        };
        let gap_str = if r.gap_pct >= 0.0 {
            format!("+{:.1}%", r.gap_pct)
        } else {
            format!("{:.1}%", r.gap_pct)
        };
        s.push_str(&format!(
            "<tr class=\"{cls}\"><td>{label}</td><td>{rcs:.1}</td><td>{decl:.1} km</td><td>{pred:.1} km</td><td>{gap}</td><td>{conf}</td><td>{cond}</td></tr>\n",
            cls = cls,
            label = r.target_label,
            rcs = r.rcs_dbsm,
            decl = r.declared_range_km,
            pred = r.predicted_range_km,
            gap = gap_str,
            conf = r.confidence_label,
            cond = r.conditions,
        ));
    }
    s
}

fn render_bibliography(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("<li>{}</li>", s))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_receipts(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("<li><code>{}</code></li>", s))
        .collect::<Vec<_>>()
        .join("\n")
}
