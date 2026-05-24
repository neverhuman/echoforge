# EchoForge Tier-1 Radar Review and Engineering Specification

## Goal

Make the paper read like a tier-1 radar / sensing benchmark paper: one clean main experiment, visible low-FPR KPI gains, auditable data processing, radar-simulation assumptions that are explicit enough for a skeptical radar reviewer, and rich appendix material that does not inflate the central claim.

This spec assumes the current paper state already fixes the NeverHuman typo, colors the abstract KPI gains, and includes a richer appendix. The remaining Tier-1 gap is not raw effort; it is traceability, reviewer clarity, and reducing ambiguity around how synthetic radar data becomes detector-view scores and holdout metrics.

## Executive verdict

### Strong
- The claim boundary is now much safer: synthetic public-proxy benchmark evidence, not measured truth.
- The abstract finally states the primary KPI and direct EI-vs-prior-fusion gains.
- The main comparator is now visible: accepted human-engineered prior fusion, not a random list of baselines.
- The low-FPR operating regime is correctly emphasized over ROC AUC.
- Leakage checks, label shuffle, metadata baselines, and group-block bootstrap are strong review assets.
- The appendix now starts to explain fixed-wing pusher-prop public-proxy assumptions, noise/clutter/RFI, and detector-view boundaries.

### Still weak / likely expert-reviewer red flags
1. **Data processing is not explicit enough.** A radar reviewer can see equations and KPI tables, but still has to infer the path from scenario group -> phase record -> IQ/cue products -> detector-view features -> train/CV search -> holdout KPI.
2. **The radar simulation is still too compact in the main text.** It lists ingredients, but the transformations are not explicit: where target return, micro-Doppler, clutter, receiver impairments, and cue streams become detector-facing evidence.
3. **The appendix is rich but should be traceable to generated artifacts.** The appendix should not just contain prose/tables; it should point to generated CSV/JSON evidence rows that can be audited.
4. **Fig. 2–5 are much better, but visual validation must enforce this quality.** Without CI gates, future rebuilds can regress to overlapping labels, missing legends, or unclear red-line annotations.
5. **The paper still risks looking like a benchmark report rather than a radar paper.** The fastest fix is a short “Data Processing and Evidence Flow” subsection and a dedicated figure.

## Required code changes in this patch

### 1. Add a new figure: `data_processing_flow.pdf/png`

File: `paper/generate_figures_major_upgrade_v2.py`

Add:
- `data_processing_flow.png` to `VECTOR_FIGURES`.
- `figure_data_processing_flow(context)`.
- `figure_data_processing_flow(context)` in `generate_all()`.

Purpose:
- Shows end-to-end processing: scenario groups -> synthetic sensing -> detector-view schema -> train/CV branch -> blind holdout.
- Displays counts from `split_summary` when available.
- Explicitly states where leakage guards, selection lock, and metric separation occur.

This is the missing bridge between the radar equations and the KPI result.

### 2. Add evidence rows for data-processing traceability

File: `detection/paper_evidence_major_upgrade_v1.py`

Add:
- `_data_processing_trace_rows()`
- `data_processing_trace_rows` in the evidence manifest.
- Writes:
  - `outputs/paper-evidence/major-upgrade-v1/data_processing_trace_rows.csv`
  - `outputs/paper-evidence/major-upgrade-v1/data_processing_trace_rows.json`

Required columns:
- `stage`
- `input_artifacts`
- `output_artifacts`
- `reviewer_check`
- `claim_boundary`

Purpose:
- Makes the processing flow auditable without making reviewers read generator internals.
- Clarifies the exact pipeline and claim boundary at every step.

### 3. Add a new paper subsection: Data Processing and Evidence Flow

File: `paper/echoforge_ieee.tex`

Insert after Figure 1 and before the radar-equation section:
- `\subsection{Data Processing and Evidence Flow}`
- `figure*` using `data_processing_flow.pdf`
- `Data Processing Traceability Checklist` table.

This section should explicitly say:
- scenario groups are sampled first;
- phase expansion inherits group split;
- synthetic IQ/cue products are generated;
- detector-view materialization applies the denylist;
- train/CV handles EI search, weights, calibration, and selection lock;
- blind holdout is scored once;
- selected-threshold counts are not the same as swept Recall@≤1%FPR or LCB95.

### 4. Expand the appendix with radar-simulation processing detail

File: `paper/appendix_modeling_details.tex`

Add:
- `Synthetic Radar Simulation Processing Detail` table.

Rows:
- Target return
- Micro-Doppler
- Clutter/noise
- Receiver effects
- Cue streams
- Feature contract

Purpose:
- A radar expert can see exactly what is modeled, how it affects evidence, and what cannot be inferred.

### 5. Strengthen validation gates

Files:
- `paper/validate_visuals.py`
- `paper/validate_paper.py`

Add:
- `data_processing_flow.pdf/png` as required artifacts.
- Visual term checks for `Data-processing path`, `Synthetic sensing`, `Detector-view schema`, `Train/CV branch`, `Blind holdout`.
- Paper text checks for `Data Processing and Evidence Flow` and `data_processing_trace_rows.csv`.
- Evidence manifest/CSV checks for `data_processing_trace_rows`.

### 6. Run visual validation in the build

File: `paper/build.sh`

Ensure:
- `python3 paper/validate_visuals.py` runs immediately after strict figure generation and before LaTeX compilation.

## Section-by-section review

### Abstract

Strong:
- States strict-open definition.
- States primary KPI.
- States comparator.
- States +185%, +742%, and 98% fewer false alarms.
- States ROC AUC caveat.

Weak:
- Very dense; this is acceptable for IEEE but should not get longer.
- Do not add more appendix detail into the abstract.

Required policy:
- Keep all gain values colored via `\kpigain{}`.
- Keep the ROC AUC caveat via `\kpicaveat{}`.
- Validation should fail if the direct percentage gains disappear.

### Claim Boundary and Contributions

Strong:
- Clear claim boundary.
- Core Experiment Roadmap helps.

Potential red flag:
- “Accepted prior fusion” must be justified as human-engineered best-practice, not simply declared.

Recommended text:
- Add one sentence in Core Experiment: “This comparator is not a straw baseline; it is the strongest non-EI late-fusion lane assembled from radar/cue branches under the same split and train/CV thresholding policy.”

### Radar and Multimodal Generative Model

Strong:
- Includes radar equation and range/Doppler formulas.
- States public-proxy and synthetic boundary.

Weak:
- Needs a process path figure.
- Needs more detail on how IQ/cues become detector-view features.
- Needs explicit “not measured imagery” wording near raw sample figures.

Patch response:
- Add `Data Processing and Evidence Flow`.
- Add data-processing figure.
- Add appendix simulation-processing table.

### Public-Source Detector Archetypes

Strong:
- Good boundary language around public product pages and no Pd/Pfa equivalence.

Weak:
- “On par” language is dangerous even with caveat. It can trigger reviewer suspicion.

Recommended edit:
- Replace “on par” with “role-compatible public envelope.” If “on par” remains, validation should not allow it without the no-sensitivity/no-ECCM/no-Pd/Pfa caveat in the same paragraph.

### Scenario Design and Detector Views

Strong:
- Group-locked split is a real strength.
- The 8 positive holdout groups limitation is honest.

Weak:
- The processing order is not explicit enough.
- Figure 2 is now better, but the page-level placement can still make it feel like a diagnostics report rather than a methods narrative.

Patch response:
- New data-processing figure absorbs the pipeline burden.
- Figure 2 can remain a balance/leakage figure.

### Evaluation Protocol and Calibration

Strong:
- Low-FPR primary KPI is correct.
- Group-block bootstrap is correct.
- The ROC AUC trade-off is stated.

Weak:
- The geodesic-odds transform should be labeled as monotone calibration, not a physics claim.
- “LCB95” should be defined once and then used consistently.

Recommended edit:
- Add “LCB95 is the 2.5th percentile of the group-block bootstrap distribution unless otherwise stated.”

### Figure 3

Strong:
- The KPI-first layout is the right story.
- Red ticks and 1% FPR cap are now much clearer.

Remaining risk:
- Small inset labels are still hard in a printed two-column reduction.

Recommended code:
- Preserve vector PDF output.
- Keep validation terms for red tick, LCB95, 1% FPR cap, +742%, +185%.

### Figure 4

Strong:
- Family legend is now present.
- Near-threshold context is a major improvement.

Remaining risk:
- The left panel still has four bar families if the earlier build is used. Prefer the simplified “fixed-FPR recall by phase” variant if page pressure returns.
- The x-axis title “False-alarm count by top family” can confuse because the stacked colors are families and the methods are y-axis rows.

Recommended title:
- “Selected-threshold false alarms, stacked by family.”

### Figure 5

Strong:
- Far better than the old cramped diagram.
- The workflow now makes the holdout boundary visible.

Remaining risk:
- “Geodesic-odds” may sound over-fancy. Add one text phrase: “monotone odds calibration.”

### Figure 6

Strong:
- Component weights and comparable controls are useful.
- The passive-RF-only caveat is very important.

Potential red flag:
- Passive-RF-only has higher AP and swept fixed-FPR recall than the full EI candidate. Reviewers will ask why EI is the main claim.

Required explanation:
- The paper must say EI is the locked calibrated selected-threshold artifact, not a post-hoc ranking winner. This is already present; keep it.

### Raw samples and KTH anchor

Strong:
- Clear compare-only language.
- KTH z-score normalization avoids unit mixing.

Weak:
- Range-Doppler panels are visually qualitative and may not persuade radar reviewers.

Recommended improvement:
- Add a sentence: “The panels are sanity checks for signal morphology; no metric is computed directly from the rasterized figure.”

### Compact diagnostics

Strong:
- Good audit trail.
- Table XI leakage diagnostics are valuable.

Weak:
- Too many tables in the main paper; some belong in appendix if page pressure is tight.

Priority:
- Keep Main KPI table, confusion matrix, false-positive burden.
- Move detailed EI false-alarm families and full leakage table to appendix if final page budget is strict.

### Appendix

Strong:
- The new appendix is now in the right direction.
- The Iranian-drone public-proxy modeling card is essential and should stay.

Weak:
- It needs to feel artifact-backed, not merely narrative.

Patch response:
- Add `data_processing_trace_rows.csv`.
- Add `Synthetic Radar Simulation Processing Detail`.
- Add data-processing artifact map.

## References: likely improvements

The existing canonical radar/ML references are solid. To make this feel more tier-1, consider adding or checking:
- more explicit clutter references for Weibull/K-distribution and sea clutter;
- a reference for FMCW range-Doppler processing if reviewers think the FMCW/pulse wording is mixed;
- a UAS detection review paper for low-slow-small radar context;
- a calibration / low-FPR operating-point citation if the primary KPI is challenged.

Do not over-cite public product pages. They support role envelopes only; they should not be used as technical validation.

## Final build/test plan

Run:

```bash
git apply echoforge_tier1_review_code_changes.diff
python3 -m detection.paper_evidence_major_upgrade_v1 --force
python3 paper/generate_figures_major_upgrade_v2.py --strict
python3 paper/validate_visuals.py
bash paper/build.sh --copy-tracked
python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

Expected artifact additions:
- `paper/figures/data_processing_flow.pdf`
- `paper/figures/data_processing_flow.png`
- `outputs/paper-evidence/major-upgrade-v1/data_processing_trace_rows.csv`
- `outputs/paper-evidence/major-upgrade-v1/data_processing_trace_rows.json`

## Bottom line

The paper is now close, but the remaining Tier-1 gap is reviewer traceability. Add a dedicated data-processing flow, make the radar simulation transformations explicit in appendix artifacts, and make CI enforce visual/text clarity. That turns the paper from “dense synthetic benchmark report” into a radar-reviewable benchmark contribution.
