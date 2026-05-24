# EchoForge paper repair engineering spec

## Goal
Make the paper unambiguous about the main KPI improvement, fix the reader-facing typography/plot-legibility failures, and add validation gates so the same problems cannot silently re-enter the paper lane.

## Current failures observed

1. **Identity typo**: the PDF author block says `NeverHumqn Research Group`; it must be `NeverHuman Research Group`.
2. **Abstract undersells the main result**: the abstract reports raw values but not the reader-facing percentage gains. The main KPI should be explicit: LCB95 Recall@≤1%FPR improves from 0.083 to 0.699, a +0.616 absolute / +742% relative gain.
3. **Fig. 2 layout**: the title blocks panel titles, and the caption/footer text overlaps x-axis tick labels. Root cause: `constrained_layout=True`, a nearly top-edge `suptitle(y=0.995)`, and footer text drawn at `y=0.02` inside a tight figure.
4. **Fig. 3 visual encoding**: red vertical marks are used but are not decoded clearly enough. Root cause: the KPI panel draws LCB95 ticks without a legend or inline annotation, and the ROC inset uses another red vertical line for the 1% FPR limit.
5. **Fig. 4 legends**: the phase panel has four bars, and the false-alarm panel has stacked family colors plus near-threshold context, but the right panel lacks a family-color legend. Root cause: stacked bars are drawn from `family_palette` but no handles are emitted.
6. **Fig. 5 overflow**: EI workflow boxes contain long text in fixed-size boxes. Root cause: `_add_box` always wraps to width 24 regardless of actual box width, and `figure_ei_workflow` packs long rail sentences into short boxes.
7. **Fig. 6 legend gap**: component weights and ablation deltas use colors and dots, but the visual encodings are not labeled enough.
8. **Table VII value problem**: the generic “pass/pass/pass” EI transparency table does not explain why EI matters. Replace it with a KPI gain ledger.

## Main KPI math to surface

| Metric | Prior fusion baseline | EI candidate | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| LCB95 Recall@≤1%FPR | 0.083 | 0.699 | +0.616 | +742% |
| Point Recall@≤1%FPR | 0.292 | 0.833 | +0.541 | +185% |
| Average precision | 0.128 | 0.825 | +0.697 | +544% |
| Selected-threshold false positives | 49 | 1 | -48 | 98% reduction |
| F1 | 0.241 | 0.837 | +0.596 | +247% |
| ROC AUC | 0.938 | 0.917 | -0.021 | -2.2% |

ROC AUC must remain visible as the guardrail: EI improves the low-FPR operating KPI and false-alarm-controlled behavior, not universal rank dominance.

## Code changes in the patch

### `paper/echoforge_ieee.tex`

- Add `xcolor` and macros `\kpiGain{}` / `\kpiWarn{}`.
- Fix `NeverHumqn` → `NeverHuman`.
- Rewrite abstract result sentence to include colored KPI values and gains.
- Add a main-KPI paragraph after the Primary KPI subsection.
- Replace the current Table VII with “Main-KPI Gain Ledger Versus the Prior Fusion Baseline”.
- Improve Fig. 3, Fig. 4, and Fig. 5 captions so the red lines, family colors, and workflow boxes are explicitly decoded.

### `paper/generate_figures_major_upgrade_v2.py`

- Import `Patch` and `Line2D` for real legends.
- Make `_add_box` accept per-box `wrap_width` and `linespacing`.
- Rebuild Fig. 2 layout with explicit margins and title/subtitle text instead of a top-edge `suptitle`.
- Add per-panel x-axis labels and move the note to safe figure-bottom space.
- Add a KPI-panel legend and inline text explaining red LCB95 ticks in Fig. 3.
- Change the ROC inset label to `1% FPR limit`.
- Add a right-panel false-alarm family legend and near-threshold legend item in Fig. 4.
- Add component-weight and delta-control legends in Fig. 6.
- Shorten and rewrap EI workflow stage and rail boxes in Fig. 5.

### `detection/paper_evidence_major_upgrade_v1.py`

- Add `_main_kpi_gain_rows()` to compute the KPI gain ledger from the evaluation summary.
- Add `main_kpi_gain_rows` to the manifest.
- Write `main_kpi_gain_table.csv` and `main_kpi_gain_table.json` for auditability.

### `paper/validate_paper.py`

- Ban `NeverHumqn`.
- Require the gain markers in the paper text.
- Require caption language that decodes the red LCB95 ticks and false-alarm family legend.
- Require `main_kpi_gain_rows` in the evidence manifest and `main_kpi_gain_table.csv` in paper evidence output.

## Validation commands

Run the repo’s paper lane after applying the patch:

```bash
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

## Review acceptance criteria

- The author block uses `NeverHuman` everywhere.
- The abstract visually highlights the main values and the percentage gains.
- Table VII is a KPI gain ledger, not a generic transparency/pass table.
- Fig. 2 has no title overlap and no x-axis/caption collision.
- Fig. 3 states what the red LCB95 tick and 1% FPR line mean.
- Fig. 4 contains a family-color legend and near-threshold legend item.
- Fig. 5 has no text outside boxes.
- Fig. 6 explains both component-weight colors and AP/recall delta encodings.
- `paper/validate_paper.py` fails if the typo, missing gain statements, or missing gain evidence table reappear.
