# Paper Feedback Coverage Matrix

This matrix maps the major reviewer concerns in `tips/paper_feedback/` to the sections and artifacts that address them. It is a coverage artifact, not a claim that every concern is fully solved in the measured sense.

| Theme | Feedback sources | Paper / evidence location | Status | Notes |
| --- | --- | --- | --- | --- |
| Claim boundary and named-platform wording | `tip1.txt`, `tip2.txt`, `tip3.txt` | Abstract, Claim Boundary, Limitations | covered | Paper-facing text stays on the fixed-wing pusher-prop public proxy boundary and avoids field-truth claims. |
| Radar model card and physical units | `tip1.txt`, `tip2.txt`, `tip3.txt` | Radar and Multimodal Generative Model; Table `tab:radar` | covered | The paper now states carrier bands, bandwidths, PRF/CPI, pulse counts, range bins, and nominal resolution assumptions. |
| Prior / parameter tables | `tip1.txt`, `tip2.txt` | Radar and Multimodal Generative Model; Evidence Ledger | covered | Model-card and prior summaries are surfaced as readable tables and evidence files. |
| Measured-anchor comparison | `tip1.txt`, `tip2.txt`, `tip3.txt` | Raw Samples and Compare-Only Anchor | covered | KTH is explicitly compare-only, with no positive-truth inference. |
| Leakage controls and split isolation | `tip1.txt`, `tip2.txt`, `tip3.txt` | Scenario Design and Detector Views; Validation Tiers; Evidence Ledger | covered | Group-locked split, denylist, canary, nearest-neighbor scan, and holdout isolation are explicit. |
| Confidence intervals and uncertainty | `tip2.txt`, `tip3.txt` | Evaluation Protocol; Primary KPI; holdout summary tables | covered | The primary KPI is the lower 95% group-block bootstrap bound of recall at FPR <= 1%. |
| Calibration and reliability | `tip1.txt`, `tip2.txt`, `tip3.txt` | Evaluation Protocol; KPI figure | covered | Brier score, ECE, and calibration bins remain visible as diagnostics. |
| Locked-candidate explanation | `tip1.txt`, `tip2.txt`, `tip3.txt` | Locked-Candidate Transparency and Ablation Results | covered | Component scores, aliases, ablations, and locked-candidate framing are explicit. |
| Low-FPR operating behavior | `tip2.txt`, `tip3.txt` | Primary KPI; Holdout summary; KPI figure | covered | Low-FPR recall is elevated to the headline KPI rather than buried in metric clutter. |
| False-alarm families and robustness | `tip1.txt`, `tip2.txt`, `tip3.txt` | Phase diagnostics; false-alarm family breakdown | covered | Birds, RC aircraft, weather, wind turbines, clutter, multipath, and RFI remain first-class robustness slices. |
| Limitations and non-claims | `tip1.txt`, `tip2.txt`, `tip3.txt` | Limitations and Prohibited Inferences | covered | The paper keeps the synthetic/public-proxy boundary intact and does not claim field truth. |
| Generative-origin audit | `tip1.txt`, `tip3.txt` | Evidence Ledger; `paper/docs/generative_origin_manifest.json` | covered | The LOC audit is an auditable self-classification, not authorship proof. |
