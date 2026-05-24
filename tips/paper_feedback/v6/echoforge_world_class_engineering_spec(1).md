# EchoForge World-Class Paper Revision Engineering Spec

## Executive decision

The paper should read like a radar evaluation paper, not like a ledger dump. The main paper must answer one question fast:

> At a <=1% false-positive-rate operating cap, does the EI sparse calibrated late-fusion candidate improve the main KPI over an accepted-practice, human-engineered prior-fusion baseline?

The answer is the headline:

- Primary KPI: LCB95 Recall@<=1%FPR.
- Prior fusion baseline: 0.083.
- EI candidate: 0.699.
- Gain: +0.616 absolute, +61.6 percentage points, +742% relative lift, 8.42x baseline.
- Point Recall@<=1%FPR: 0.292 -> 0.833, +54.1 pp, +185%, 2.85x.
- Selected-threshold false positives: 49 -> 1, 98.0% fewer false alarms.
- AP: 0.128 -> 0.825, +0.697, about +545%.
- Caveat: EI ROC AUC is lower, 0.917 vs 0.938, so this is a low-FPR operating-point claim, not universal ranking dominance.

Everything else must support that claim, or move to appendix/evidence artifacts.

## Current-state review

### Page 1 / Abstract

The updated PDF fixes the old `NeverHumqn` typo in the rendered author block, but the fix needs to be enforced in validation so it cannot regress. The abstract now includes gains, but it can be even more direct by making the first experimental question explicit and by using consistent comparator wording: `accepted-practice, human-engineered prior fusion` rather than a vague human-performance claim.

### Section I / Claim Boundary and Contributions

The contributions list is still too governance-heavy. It should lead with the core experiment and result, then say how leakage checks, modeling detail, hard negatives, and compare-only anchors support the result. The proposed patch adds a result-first core experiment statement and a short `reader-facing answer` paragraph.

### Section II / Radar and Multimodal Generative Model

The section has the right starting equations, but it undersells the radar modeling rigor. It should explicitly tie the model to detection theory, RCS/aspect modeling, Swerling-like fluctuations, clutter models, and low-angle clutter. The patch adds citations to `kay1998`, `vantrees2002`, `swerling1954`, `ward1990`, `knott2004`, and `billingsley2002`, and it adds a sentence that moves the detailed positive-proxy and noise modeling into appendix ledgers.

### Section III / Public-Source Detector Archetypes

The detector archetype table is valuable, but the main text should not imply product parity. It should emphasize role envelopes and detector-visible products. Keep the table, but make sure appendix source ledgers spell out what is not proved by product pages.

### Section IV / Scenario Design and Detector Views

This section is still one of the strongest parts. It should remain compact. Figure 2 should be the visual audit map, not another wall of text. The patch hardens the figure layout with larger margins, fewer x-axis ticks, an explicit audit note, and a bottom reading guide.

### Section V / Evaluation Protocol and Calibration

This is where the paper should become decisive. The KPI text is good, but the gain row should be visually and textually emphasized. The patch keeps the table, adds colored gain emphasis in the prose, and updates Figure 3 so the KPI gain is computed from source values rather than hard-coded.

### Section VI / Engineered Intelligence

This section should not have a process-status table in the main paper. The old Table VII had low information value and could float far away from its section, making the reader ask why it exists. The patch removes that table and replaces it with a two-paragraph explanation: what EI does, what it does not receive, and what claim is actually being made.

### Section VII / Raw Samples and Compare-Only Anchor

The raw sample figures are useful but should be clearly marked as diagnostic. Figure 7 should say `positive proxy assumptions`, not just `public-proxy assumptions`, and it should use the paper-facing proxy language. Figure 8 should remain visual diagnostic only. Figure 9 should remain compare-only.

### Section VIII / Compact Diagnostics Snapshot

These tables are valuable for audit, but they should not interrupt the main result flow. If page pressure grows, keep only the confusion matrix and false-positive burden in the main paper and move the leakage table, phase table, and EI false-alarm family table to the appendix.

### Evidence Ledger and Appendices

The appendix is currently too thin for the modeling claim the user wants. It says there are artifacts, but it does not show enough of the modeling structure. The patch adds:

- A new appendix figure: `proxy_noise_model_card.pdf`.
- A positive-proxy modeling ledger table.
- A noise/clutter/RFI/receiver modeling ledger table.
- Evidence-generation CSV/JSON files for `positive_proxy_model_rows` and `noise_clutter_model_rows`.

The appendix language uses `Iranian-style fixed-wing pusher-prop public proxy` carefully: it is a modeling shorthand for the open-source visual/kinematic envelope, not a measured Iranian-platform signature or operational claim.

## Figure-by-figure remediation plan

### Figure 1: Radar evidence stack

Current status: visually clean but generic.

A+ target:

- Keep as high-level claim-boundary figure.
- Caption should say exactly what the figure does and does not prove.
- Do not add more blocks; this figure is already a good opener.

Patch status: no major change, because it is not the bottleneck.

### Figure 2: Scenario balance and leakage diagnostics

Current problem:

- Earlier rendered versions had a title collision and dense x-axis/tick text.
- Current version is better, but the function should be hardened against layout regressions.

Patch changes:

- Taller canvas: 4.86 -> 5.08 inches.
- Top margin moved down: `top=0.835`.
- Bottom margin increased: `bottom=0.165`.
- Wider panel spacing: `hspace=0.68`, `wspace=0.48`.
- Fewer x-axis ticks: `MaxNLocator(nbins=4)`.
- Explicit reading guide at the bottom: counts on left, leakage/sanity checks on right.
- Clarifies that split/group identifiers are audit context, not detector features.

A+ target:

- The figure should answer: Is the split group-locked, balanced enough to audit, and leakage-guarded?
- The reader should not need the caption to decode axes.

### Figure 3: Holdout KPI and supporting diagnostics

Current problem:

- The red vertical line is visible but still easy to miss at small IEEE scale.
- The headline KPI gain should be visible inside the figure, not just in text.

Patch changes:

- Figure subtitle is computed from evidence values, not hard-coded.
- Primary panel includes a green gain callout with LCB95 and point-recall deltas.
- ROC panel uses a pale red low-FPR operating region, a thick red line at 1%, and a white-background callout with arrow.
- Visual validation requires `Primary KPI gain`, `+742%`, `+185%`, `LCB95`, and `1% FPR operating cap` in the vector figure text.

A+ target:

- Reviewer remembers the metric and the gain after 10 seconds.
- ROC caveat remains visible: AP/PR/calibration are diagnostics, not the headline.

### Figure 4: Phase behavior and false-alarm burden

Current problem:

- Too many legends compete with each other.
- False-alarm colors need a readable family key.
- Near-threshold bars need to be explained visually and in caption.

Patch changes:

- Figure height increases to 4.35 inches.
- More bottom margin for legends.
- Phase legend moves below left panel.
- False-alarm family legend moves below right panel.
- Legend entries become dynamic: only families present in false alarms or near-threshold counts are shown.
- Caption explicitly says stacked colors are false-alarm families and hatched/pale bars are near-threshold negatives.

A+ target:

- The reader can tell exactly what blue/black/orange/green mean.
- The figure explains both selected false alarms and near misses.
- Phase results stay caveated because there are only eight positive holdout groups.

### Figure 5: EI workflow

Current problem:

- Earlier versions had text spilling out of boxes.
- Current layout is better, but the workflow should foreground what EI actually does.

Patch changes:

- Keeps clipped text safety in `_add_box`.
- Uses a clear high-level subtitle: train/CV discovery -> sparse fusion -> calibration -> locked holdout score.
- Adds card boundaries for group-lock, feature denylist, claim boundary, and code boundary.

A+ target:

- A non-author reviewer can explain EI in one sentence: train/CV candidate discovery plus sparse calibrated late fusion, locked before holdout.

### Figure 6: EI components and ablations

Current problem:

- Useful but potentially too detailed for the main paper.
- It should be clearly labeled as interpretability/control evidence, not the main claim.

Patch status:

- No major geometry change in this patch.
- Caption/text changes keep component weights as interpretability controls.

A+ target:

- The figure should answer what the EI fusion uses and what happens under comparable controls.
- It should not imply the passive-RF-only view is the true final winner when its selected-threshold behavior/calibration are weaker.

### Figure 7: Radar model card

Current problem:

- Useful, but it does not yet satisfy the user request to show how the Iranian-style proxy and noise are modeled.

Patch changes:

- Changes the card title to `Positive proxy assumptions`.
- Uses `Iranian-style fixed-wing pusher-prop public proxy` as the paper-facing proxy label.
- Adds a new appendix figure for the richer breakdown rather than overloading Figure 7.

A+ target:

- Figure 7 stays compact; the appendix figure/table pair carries the full modeling ledger.

### Figure 8: Range-Doppler samples

Current problem:

- It is a diagnostic snapshot, not evidence of realism by itself.

Patch status:

- No major code change.
- Keep caption strict: axes and color scale are explained; no detector-feature or measured-truth claim.

A+ target:

- The figure should not invite a reviewer to ask whether this is real measured data.

### Figure 9: KTH anchor overlay

Current problem:

- Good concept, but should stay compare-only.

Patch status:

- No major code change.
- Appendix and limitations text continue to prohibit positive-class truth transfer.

A+ target:

- The reader understands that KTH normalizes observable shape, not platform truth.

### New Appendix Figure: Public proxy and noise/clutter/RFI modeling card

Purpose:

- Directly satisfy the user requirement: show how the Iranian-style proxy, noise, clutter, RFI, impairments, cues, and non-claims are modeled.

Patch changes:

- Adds `figure_proxy_noise_model_card(context)`.
- Adds `proxy_noise_model_card.png/pdf` to `VECTOR_FIGURES`, `generate_all`, `validate_visuals.py`, and `validate_paper.py`.
- Sources values from `public_proxy_positive_class_card` and `radar_model_card` when present.

A+ target:

- Reviewer can see the decomposition without reading JSON: positive proxy body, kinematics, radar observables, clutter/noise/RFI, receiver impairments, fusion cues, and non-claims.

## Evidence-generation changes

The patch updates `detection/paper_evidence_major_upgrade_v1.py` so appendix content is evidence-backed, not just manually typed prose.

New/extended artifacts:

- `public_proxy_positive_class_card.json`: expanded with radar observables and noise/environment model fields.
- `positive_proxy_model_rows.csv/json`: geometry, kinematics, RCS/aspect, micro-Doppler, and claims boundary rows.
- `noise_clutter_model_rows.csv/json`: clutter/noise/RFI/receiver/hard-negative modeling rows.

Validation requires these artifacts so the appendix cannot become stale prose.

## TeX/content strategy

### Main paper flow

1. Abstract: state experiment, metric, baseline, EI gains, caveat.
2. Claim boundary: one page maximum; no operational claims.
3. Core experiment: directly name accepted-practice prior fusion vs EI.
4. Radar model: equations, resolution, clutter/noise summary, then point to appendix.
5. Evaluation: KPI, Figure 3, Table VI, gain row.
6. EI: plain-language workflow; no table that merely says `pass`.
7. Diagnostics: only the diagnostics that support trust.
8. Appendix: rich modeling ledgers and evidence artifacts.

### Language rules

Use:

- `accepted-practice, human-engineered prior fusion` for the comparator.
- `Iranian-style fixed-wing pusher-prop public proxy` only as a modeling shorthand.
- `synthetic public-proxy evidence` for the claim level.

Do not use:

- `human best practice` as if humans were the evaluated subjects.
- `measured Iranian drone signature`.
- `sensor parity`, `operational parity`, or `classified fidelity`.
- `universal rank dominance`.

## Radar-review gaps closed by this patch

1. **RCS/aspect support**: adds RCS reference support and appendix rows.
2. **Target fluctuation**: connects Swerling-like behavior to standard fluctuation models.
3. **Clutter/noise specificity**: distinguishes Weibull/K-like clutter, terrain/sea/weather clutter, RFI, thermal/SNR buckets, receiver impairments.
4. **Low-FPR clarity**: primary KPI and red line are visually and mathematically emphasized.
5. **Calibration honesty**: maintains Brier/ECE as diagnostics and keeps ROC AUC caveat.
6. **Hard-negative interpretation**: false-alarm families get a legible legend and appendix context.
7. **Public-proxy boundary**: appendix tables explicitly list non-claims.

## Test plan

Run in this order:

```bash
python detection/paper_evidence_major_upgrade_v1.py --force
python paper/generate_figures_major_upgrade_v2.py --strict
python paper/validate_visuals.py
./paper/build.sh
python paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

Manual visual QA checklist:

- Page 1: no `NeverHumqn`; colored abstract gains visible in PDF output.
- Figure 2: no title overlap; x-axis tick labels do not collide; right-side cards readable.
- Figure 3: 1% FPR red line and shaded operating cap obvious; gain callout present.
- Figure 4: separate phase legend and family legend; family colors readable; near-threshold hatch explained.
- Figure 5: no clipped/spilled text; workflow understandable without caption.
- Figure 7/new appendix figure: positive proxy and noise/clutter/RFI modeling visible.
- Tables: Table VII no longer exists as a low-value process table; appendix tables carry rich modeling detail.

## Risk and review notes

- Adding a new appendix figure and tables may push the PDF beyond 14 pages. The patch updates validation to allow 18 pages because the user explicitly wants a richer appendix. If a strict conference page limit applies, move the rich appendix to a supplementary PDF while keeping the main paper lean.
- The phrase `Iranian-style` must be handled carefully. It should describe open-source geometric/kinematic similarity only, never measured radar truth or operational behavior.
- Product-page citations should remain source-ledger anchors for public role envelopes, not scientific proof of detector performance.
- The figure generator still depends on local output artifacts. Use `--strict` to catch missing artifacts before figures silently fall back.
