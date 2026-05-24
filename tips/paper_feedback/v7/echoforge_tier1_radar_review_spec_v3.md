# EchoForge Tier-1 Radar-Paper Review and Engineering Spec v3

## Executive diagnosis

The current EchoForge paper is much stronger than the earlier draft: the author typo is fixed, the abstract now leads with the main KPI, the accepted prior-fusion comparator is named, and the appendix now contains a richer public-proxy model card. The current 15-page PDF still falls short of a tier-1 radar paper in five ways:

1. **The radar simulation is still too qualitative in the main text.** The model has equations and a compact model card, but a radar reviewer still cannot reconstruct the data path from scenario parameters to IQ to range-Doppler summaries to detector-view features.
2. **The EI method is still described at a high level, but not enough to remove skepticism.** The paper says train/CV search, sparse nonnegative fusion, and geodesic-odds calibration, but it should show an explicit pipeline and pseudo-code/algorithm table.
3. **The statistics are honest but need sharper framing.** Eight positive holdout groups is a major reviewer concern. The paper correctly labels phase/family slices diagnostic, but the main experiment should also state exactly what would falsify or stress the claim.
4. **Figure placement and page flow still dilute the result.** Page 5 contains Fig. 2 plus Table VII and then the EI section. Page 6 contains the main result figure but immediately drops into component weights. The story should be: experiment roadmap -> radar model -> data processing -> main result -> error analysis -> EI mechanism -> appendix.
5. **The appendix is richer, but not yet a fully auditable data-processing appendix.** It should include file-level evidence rows, exact generated CSV names, schema fields, and a `reproduce_evidence.sh` command path.

The main result is compelling: EI improves LCB95 Recall@≤1%FPR from 0.083 to 0.699 (+742%, 8.42x), point Recall@≤1%FPR from 0.292 to 0.833 (+185%, 2.85x), and selected-threshold false positives from 49 to 1. The paper must protect that claim by being unusually clear about the synthetic boundary, the low positive-count limitation, the feature denylist, the score pipeline, and the exact benchmark artifacts.

---

## Section-by-section review

### Abstract

**Strong:** The abstract now states the comparator, dataset size, holdout size, primary KPI, exact gains, FP reduction, and ROC AUC trade-off.

**Weak / likely reviewer concern:** It still uses a lot of compressed benchmark vocabulary in the first half. A radar reviewer may understand the result, but the sentence about strict-open/generated arrays/measured files is less important than the experiment. Keep it, but shorten it.

**Change:** Make the first two sentences: “EchoForge is a strict-open synthetic benchmark for low-altitude UAS radar and multimodal sensing. The main experiment compares accepted human-engineered prior fusion against a locked EI sparse late-fusion candidate on the same blind group-locked holdout.” Then give the result.

### Claim Boundary and Contributions

**Strong:** The paper now says no measured-platform truth, no proprietary-equivalent behavior, no classified fidelity.

**Weak:** The “contributions” are still paper-centric rather than experiment-centric. It should explicitly state the null/baseline hypothesis: accepted human-engineered prior fusion is the comparator; EI is a challenger; the claim is low-FPR operating-point improvement on this declared synthetic run.

**Change:** Add a “Claim tested” paragraph and an “Out-of-scope” mini-list. This prevents reviewers from assuming sensor-parity or operational claims.

### Core Experiment Roadmap

**Strong:** Table I is valuable and should stay.

**Weak:** Table I is not enough. Add an experiment-ladder figure or compact algorithm table that shows the exact data flow: scenario group -> phase rows -> IQ/cue streams -> detector views -> train/CV calibration/search -> selection lock -> holdout scoring.

**Change:** Add `experiment_ladder.pdf` or convert Fig. 1 into a clearer experiment-ladder figure. The current Fig. 1 is a governance/signal-chain stack, not an experiment design figure.

### Radar and Multimodal Generative Model

**Strong:** The monostatic received-power equation, IQ equation, resolution equations, and radar model card are credible.

**Weak / red flags:**

1. FMCW/chirp and pulse/CPI terminology is mixed. The current text calls this a “proxy,” but a radar reviewer will still ask if this is pulsed Doppler, FMCW, or a generic range-Doppler tensor generator.
2. The IQ model is too compact. It needs an explicit data-processing chain.
3. No parameters are shown for clutter, RCS fluctuation, receiver impairments, or SNR/CNR distributions in the main text.
4. It is unclear whether the detector models see raw IQ, range-Doppler heatmaps, extracted scalar summaries, or all of the above.

**Change:** Add a new subsection `Data Processing and Detector-View Construction` directly after the generative model. This should include:

- Scenario schema: `scenario_group_id`, split, site/range/aspect/noise/hard_negative_role.
- Phase expansion: three rows per group.
- Signal product: complex IQ with 24 pulses x 20 range bins.
- Transform: range-Doppler proxy via FFT/log magnitude only for visual diagnostics.
- Detector features: scalar detector-view CSVs; raw IQ not passed to the main KPI model.
- Denylist: label, split role, group ID, time lock ID, audit headers excluded from model matrices.
- Evidence output: performance metrics, selected thresholds, curves, calibration bins, false-alarm family tables.

### Detector Archetypes

**Strong:** Public-source role envelopes and excluded truth fields are useful.

**Weak:** It does not explicitly distinguish “accepted best practice” from “vendor-like branch.” The accepted human baseline is `layered_fusion_c2`, not a vendor implementation.

**Change:** Add one paragraph: accepted prior fusion is a human-engineered score fusion/control baseline built from detector-view outputs and train/CV thresholding; it is not claimed to be an operational C2 system.

### Scenario Design and Detector Views

**Strong:** Group-locked split and leakage checks are strong.

**Weak:** The class balance is extreme (50 positive groups out of 10,000; 8 positive holdout groups). This is honest but a tier-1 reviewer will want to know why this is not a toy. The answer is: rare-positive low-FPR benchmark, but the single-seed/small-positive holdout limits generality.

**Change:** Move some balance diagnostics to appendix. Keep the main text focused on group lock, positive counts, and leakage tests. Add `why rare-positive` explanation.

### Fig. 2

**Strong:** The latest Fig. 2 is readable.

**Weak:** It still asks too much of the reader: split/site/range/hard-negative + leakage + imbalance. This is now acceptable, but tier-1 polish would reduce to: split, positive counts, leakage checks, and a pointer to appendix for full strata.

**Change:** Either simplify Fig. 2 or relabel it “Audit snapshot” and state that full marginal balance is in CSV. If kept, the caption should define “marginal imbalance” in one phrase.

### Evaluation Protocol

**Strong:** Primary KPI is appropriate for rare-positive false-alarm constrained detection. ROC AUC trade-off is correctly caveated.

**Weak:** The relationship between selected-threshold counts and swept fixed-FPR recall remains easy to misunderstand. The paper has table wording, but add a small equation/definition block.

**Change:** Add definitions:

- selected-threshold TP/FP/FN: threshold locked on train/CV or EI lock;
- Recall@≤1%FPR: swept threshold chosen to satisfy FPR constraint on holdout for diagnostic operating-point comparison;
- LCB95: lower 2.5% group-block bootstrap bound.

### Fig. 3

**Strong:** Now clearly the main result. Good.

**Weak:** The AP ranking panel still includes multiple bars but not exact AP labels; it is diagnostic. That is fine. The main KPI panel should not include too many text annotations in the plot area.

**Change:** Keep current, but move `selected-threshold FP 49->1` into subtitle or caption rather than tiny plot text if it overlaps at final size.

### Engineered Intelligence section

**Strong:** Good high-level explanation of train/CV search, sparse fusion, calibration, lock.

**Weak / red flags:** “EI” can sound like marketing. It needs a more transparent algorithm sketch and one sentence saying it is not a magic model: it is a constrained meta-fusion/search layer over detector scores.

**Change:** Add `Algorithm 1: EI candidate discovery and locked scoring` as pseudo-code in text/table form:

1. Read train/CV detector-view scores.
2. Generate component candidates from detector/modality score families.
3. Optimize nonnegative sparse weights on train/CV only.
4. Fit monotone geodesic-odds calibrator on train/CV.
5. Write selection lock.
6. Score holdout once.
7. Emit main KPI and guardrail diagnostics.

### Fig. 4

**Strong:** The latest version is much improved: fixed-FPR recall by phase and clear false-alarm legend.

**Weak:** It still mixes two ideas: phase behavior and false-alarm family burden. Acceptable for space, but tier-1 would split if possible.

**Change:** If page count permits, split into `phase_kpi.pdf` and `false_alarm_burden.pdf`. If not, keep but ensure the legend never touches title area and the caption says near-threshold bars are not false positives.

### Fig. 5 and Fig. 6

**Strong:** Fig. 5 is clear; Fig. 6 has legends now.

**Weak:** Fig. 6 may still overemphasize passive RF, creating a reviewer question: why not use passive-RF-only if it has higher AP and Recall@≤1%FPR? The text explains F1/ECE; add a one-line callout to Fig. 6 caption or body.

**Change:** Add sentence: “Passive-RF-only is a post-hoc view control with higher AP but worse selected-threshold F1/ECE, so it is not the locked EI artifact.”

### Raw Samples and KTH Anchor

**Strong:** The captions correctly state qualitative diagnostics and compare-only anchor.

**Weak:** Fig. 8 heatmaps are not very discriminative visually; a radar reviewer may ask what they prove.

**Change:** Rename “Range-Doppler proxy diagnostics” to “Qualitative range-Doppler sanity panels.” Add micro-Doppler summary statistics or no longer use Fig. 8 as main evidence. The caption should be explicit: not detector input, not measured imagery, not evidence of platform truth.

### Evidence Ledger / Diagnostics

**Strong:** Solid audit culture.

**Weak:** Too many tables in main text. A tier-1 paper should move many tables to appendix/supplement and keep the main paper lean.

**Change:** Main text should keep only: experiment roadmap, compact radar model card, main KPI gain ledger, holdout summary, and maybe modality controls. Move leakage table, confusion matrix, false-alarm burden table, phase table, false-alarm family table to appendix or supplementary if page pressure exists.

### Rich Modeling Appendix

**Strong:** Much better now. Fig. 10 and Tables XVI-XVIII directly answer “how do you model the Iranian drone, noise, detector views?”

**Weak:** It still does not include enough numeric distributions. For tier-1, append exact priors/ranges from generated evidence CSVs, and give file names. It should be auditable from artifacts.

**Change:** Expand `public_proxy_model_detail_rows.csv` and `environment_impairment_model_rows.csv` to include `parameter`, `low`, `high`, `units`, `distribution`, `source_basis`, `claim_boundary` where possible. The current table has prose; reviewers will want reproducible numbers.

### References

**Strong:** Good radar fundamentals and micro-Doppler references.

**Weak / possible red flags:** Some product pages and future-year entries are fragile. The current references include public product pages and 2026 dates, which can look odd. If those are generated placeholders or current web pages, use access dates and avoid future publication-like years. Add stronger literature for synthetic radar simulation, clutter distributions, and calibration/uncertainty.

**Change:** Add or verify references for:

- FMCW radar signal model and range-Doppler processing;
- K-distribution/Weibull clutter and CFAR under non-Gaussian clutter;
- calibration and uncertainty under imbalanced classification;
- UAS radar datasets and drone/bird classification.

---

## Exact code-change plan

The accompanying patch file implements the following targeted changes:

1. `paper/echoforge_ieee.tex`
   - Add a formal claim-tested paragraph.
   - Add a `Data Processing and Detector-View Construction` subsection.
   - Add selected-threshold vs swept fixed-FPR definitions.
   - Add EI algorithm table.
   - Strengthen Fig. 8 and Fig. 10 captions.
   - Add text to explicitly differentiate post-hoc passive-RF-only controls from the locked EI artifact.
   - Add richer appendix tables and file-backed artifact references.

2. `detection/paper_evidence_major_upgrade_v1.py`
   - Add `data_processing_contract_rows` for the scenario-to-IQ-to-detector-feature flow.
   - Add `ei_algorithm_steps` for selection-lock auditability.
   - Add distribution-style fields to proxy/environment rows.
   - Write `data_processing_contract.csv/json` and `ei_algorithm_steps.csv/json`.
   - Add these rows to `paper_evidence_manifest.json`.

3. `paper/generate_figures_major_upgrade_v2.py`
   - Add `data_processing_flow.pdf/png` to show pipeline from scenario groups to evidence tables.
   - Add `ei_algorithm_map.pdf/png` if desired, or expand Fig. 5 with enough clarity.
   - Add these figures to `VECTOR_FIGURES` and `generate_all`.

4. `paper/validate_paper.py`
   - Require the data-processing section.
   - Require the EI algorithm language.
   - Require new evidence artifacts.
   - Fail if the paper includes unresolved “Table-IV-compatible” wording or stale `selected candidate` language.

---

## Acceptance criteria

A tier-1 radar reviewer should be able to answer the following from the paper without guessing:

- What is the experiment?
- What is the accepted-practice comparator?
- What does EI do, without marketing language?
- What data are generated, and what data are model-facing?
- What fields are blocked to prevent leakage?
- What exact metric is primary, and how is its CI computed?
- Why does ROC AUC decrease while the main KPI improves?
- Why is passive-RF-only not the main claimed artifact despite strong post-hoc AP?
- What is the Iranian-drone public proxy, and what is explicitly not claimed?
- What noise/clutter/RFI/receiver impairments are modeled, and at what level of abstraction?
- What are the limits of the single-seed, eight-positive-holdout-group result?

