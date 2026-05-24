# EchoForge world-class paper repair engineering spec

## Scope

This spec and patch target the current EchoForge paper lane in `paper/echoforge_ieee.tex`, `paper/generate_figures_major_upgrade_v2.py`, `detection/paper_evidence_major_upgrade_v1.py`, and `paper/validate_paper.py`.

The paper is much closer than the earlier draft: the author block now says `NeverHuman`, the abstract already contains the main EI gains, and Fig. 3/Fig. 4 are substantially better. The remaining issue is that the main paper still has too many explanatory tables, the primary KPI figure still has a small legend/text collision, the phase/false-alarm legend needs more whitespace, and the appendix is not rich enough to satisfy a radar reviewer who asks, “How exactly are the Iranian/Shahed-style public proxy, radar observables, clutter, RFI, receiver impairment, and hard negatives modeled?”

## Executive narrative target

The first five pages should make one argument, cleanly:

1. **Experiment:** same synthetic public-proxy corpus, group-locked train/CV and blind holdout, rare-positive low-FPR regime.
2. **Comparator:** accepted human-engineered prior fusion baseline over radar/cue detector views.
3. **New method:** EI = train/CV score search + sparse nonnegative fusion + odds calibration + written selection lock + one blind holdout pass.
4. **Result:** LCB95 Recall@<=1%FPR improves from 0.083 to 0.699, +0.616 absolute, +742% relative; point Recall@<=1%FPR improves +185%; AP improves +545%; false positives fall from 49 to 1.
5. **Tradeoff:** ROC AUC falls from 0.938 to 0.917, so this is a low-FPR operating gain, not universal rank dominance.
6. **Boundary:** synthetic public-proxy evidence only; no measured Iranian-drone radar signature, platform equivalence, or operational performance claim.

## Rendered-PDF findings

### Abstract and page 1

The abstract now contains the correct `NeverHuman` author block and the main numerical gains. Keep the colored gain macros and validation checks, but the phrasing should stay concise. The abstract should not become a mini-methodology section.

### Figure 2: scenario balance and leakage diagnostics

Current status: readable in the latest render. It no longer has the catastrophic title collision seen in the earlier draft. Keep the full-width two-row layout. Acceptance checks:

- Title and subtitle remain above all panels.
- Top row does not collide with leakage cards.
- Bottom x-axis labels do not collide with caption.
- Hard-negative labels are readable at final IEEE size.
- Leakage card text remains short; do not add more values into the plot.

### Figure 3: holdout KPI

Current issue: the primary KPI panel has a small in-axis legend/text conflict at the lower-left corner. This is a credibility problem because Fig. 3 is the paper’s most important figure.

Patch action:

- Move the primary KPI legend out of the axis and into a figure-level legend above the plot.
- Add a direct LCB95 gain annotation between baseline and EI lower-bound ticks: `LCB95 +742% (+0.616)`.
- Keep the shaded low-FPR ROC region and explicit `1% FPR operating cap` arrow.
- Keep AP, ROC, PR, and calibration as supporting diagnostics, not co-equal claims.

Acceptance checks:

- No legend overlaps any bar, tick, or value label.
- The red vertical ticks are explained visually and in the caption.
- The reader can identify the primary KPI without reading the full caption.

### Figure 4: phase behavior and false-alarm burden

Current issue: the chart is much better than the original, but the two legends still sit close to the figure title and compete with the plot. This is especially risky after PDF scaling.

Patch action:

- Increase figure height and reserve a larger top band for legends.
- Move phase legend and family legend higher with more whitespace.
- Keep the family legend title and near-threshold hatch marker.
- Keep the caption wording explaining stacked family colors, near-threshold bars, FP/near/R@1% labels, and the small positive holdout caveat.

Acceptance checks:

- The phase legend does not touch the title.
- The family legend does not cover the right-panel title.
- The false-alarm stacks are interpretable without guessing what colors mean.

### Figure 5: EI workflow

Current status: acceptable. The workflow boxes are now readable and the rails do not visibly overflow. Keep the short stage labels. Do not reintroduce long prose inside boxes.

Acceptance checks:

- No text extends beyond a box.
- The workflow reads left-to-right as detector views -> train/CV search -> sparse fusion -> odds calibration -> selection lock -> blind holdout.
- Audit rails remain one or two short lines each.

### Figure 6: component weights and controls

Current status: useful. It should remain an interpretability figure, not a second leaderboard. The right panel must continue to label bars as AP deltas and dots as recall deltas.

Acceptance checks:

- Keep the “bars AP, dots recall” title.
- Keep raw component handles out of the main figure.
- Keep modality controls framed as diagnostics, not as alternate final models.

### Figure 7: model card / public proxy

Current issue: useful but too shallow for the user’s “show HOW we model the Iranian drone/noise/all of it” request. The main figure can remain compact, but the appendix must expand this into tables.

Patch action:

- Add rich appendix tables in TeX.
- Add evidence CSV/JSON rows for positive public-proxy modeling and noise/clutter/RFI modeling.

### Figure 8 and Figure 9

Current status: acceptable as diagnostic figures. Fig. 8 is visually dense but clear enough as a proxy sample panel. Fig. 9 is improved because it uses unitless measured-anchor z-scores. Keep the compare-only boundary explicit.

## Text-level changes included

### `paper/echoforge_ieee.tex`

1. Tightens contribution bullets so the paper gets to the experiment and KPI faster.
2. Removes the low-value “Primary KPI Selection” table from main text.
3. Removes the low-value “Main Experiment Lanes” table from main text. The EI concept is now explained in prose and figures.
4. Adds a formal rich modeling appendix after the conclusion with:
   - `Positive Public-Proxy Modeling Card for the Fixed-Wing Pusher-Prop Class`
   - `Noise, Clutter, RFI, and Receiver Modeling Detail`
   - `Accepted Best-Practice Comparator Lane Used in the Main Experiment`
5. Adds radar-supporting citations already present in the repository bibliography: Swerling target fluctuation, Ward sea clutter, Kay/Van Trees detection theory, standard radar and ML references.
6. Keeps the claim boundary explicit: no measured Iranian-drone radar signature, no route/tactics model, no payload inference, no deployment-performance claim.

### `paper/generate_figures_major_upgrade_v2.py`

1. Reworks Fig. 3 layout:
   - primary KPI panel is wider;
   - legend moves to figure-level position;
   - LCB95 relative gain is annotated directly;
   - axis height is expanded so annotation does not collide.
2. Reworks Fig. 4 layout:
   - larger figure height;
   - more top whitespace;
   - phase and family legends lifted away from plot titles.

### `detection/paper_evidence_major_upgrade_v1.py`

Adds two evidence emitters:

1. `positive_proxy_modeling_rows`
   - geometry;
   - kinematics;
   - launch/take-up phase;
   - radar scattering;
   - prop micro-Doppler;
   - acoustic/passive-RF cue modeling;
   - detector-visible effect and claim boundary for each row.

2. `noise_clutter_rfi_model_rows`
   - Weibull/K-like clutter;
   - multipath/ghosting;
   - weather/sea/terrain stress;
   - receiver impairments;
   - RFI/passive-RF missingness.

These are written as both `.json` and `.csv` artifacts and included in `paper_evidence_manifest.json`.

### `paper/validate_paper.py`

1. Forbids the removed “Primary KPI Selection” table from returning.
2. Requires the rich appendix markers.
3. Requires the new evidence manifest keys and CSVs.
4. Raises the page-count ceiling from 14 to 16 to allow the richer appendix while preserving a bounded paper length.

## Full rebuild commands

Run from repository root:

```bash
git apply /mnt/data/echoforge_world_class_paper_patch.diff

rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/generate_figures.py --strict
rtk bash paper/build.sh --copy-tracked

rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Acceptance criteria

### Main paper

- The abstract contains colored KPI gains and the ROC AUC caveat.
- `NeverHumqn` does not appear anywhere.
- The first two sections establish experiment, comparator, EI method, and claim boundary quickly.
- The main KPI is always LCB95 Recall@<=1%FPR.
- The main paper does not waste space on pass/pass tables.
- The EI method is explained at a high level before the reader sees component weights.

### Figures

- Fig. 2 has no title/subtitle/axis overlap.
- Fig. 3 has no legend overlap and directly labels the LCB95 gain.
- Fig. 3 ROC inset clearly marks the 1% FPR operating cap.
- Fig. 4 has readable phase and family legends.
- Fig. 4 false-alarm colors are interpretable, including near-threshold hatching.
- Fig. 5 text stays inside every box.

### Appendix

- The appendix explains how the positive fixed-wing pusher-prop / Shahed-style public proxy is modeled.
- The appendix explains how clutter, noise, RFI, multipath, weather, and receiver impairments are modeled.
- The appendix explicitly states what each modeling row is not allowed to prove.
- The evidence generator emits matching JSON/CSV artifacts for those appendix claims.

### Radar-review hard gates

- No measured-platform truth claim.
- No named-platform radar signature claim.
- No sensor-parity or operational-parity claim.
- No field false-alarm-rate claim.
- No implication that passive-RF-only post-hoc controls supersede the pre-registered EI lock.

## Recommended future work beyond this patch

- Add a true multi-seed evaluation and report seed-to-seed confidence intervals.
- Increase positive holdout groups beyond eight; this is the largest statistical weakness.
- Add a track-level false-alarm metric separate from record-level false positives.
- Add a measured positive-class validation lane when legally and ethically available.
- Add a hardware-in-the-loop or calibrated measured clutter validation lane.
- Split the rich appendix into a companion technical report if the target venue has a hard page limit.
