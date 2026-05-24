# EchoForge EI Evolution “Money Plot” Engineering Spec

## 1. Purpose

This spec implements the reviewer-safe EI evolution plot requested for the paper: a figure that shows Engineered Intelligence improving through incremental train/CV candidate testing, then shows a single locked holdout endpoint. The goal is to turn the EI narrative from “we found a final model” into “we can audit the search trajectory that produced the final locked model.”

The current paper already states the correct split discipline: candidate discovery, thresholding, and calibration are train/CV-only, followed by one blind group-locked holdout pass after the selection lock. The uploaded paper also already contains the EI workflow and component/control evidence, but not the iteration-by-iteration evolution curve that would make the EI result visually obvious to a skeptical reviewer.

## 2. What the diff delivers

The diff adds one new evidence-producing module, instruments the advanced runner, adds the paper-facing figure, inserts the LaTeX section/table, and tightens validation.

Changed files:

| Path | Change |
|---|---|
| `detection/advanced_main_run_detectors.py` | Emits full candidate-order `evolution_trace.jsonl` and `evolution_trace.csv`; marks the selected row after train/CV selection; records trace metadata in `selection_lock.json` and `fusion_quality_report.json`. |
| `detection/ei_evolution_trace.py` | New normalizer that builds `ei_evolution_trace.csv` and `ei_evolution_summary.json` for the paper evidence lane. It fails in strict mode if only leaderboard rank is available. |
| `detection/paper_evidence_major_upgrade_v1.py` | Pulls the normalized EI evolution evidence into `paper_evidence_manifest.json`. |
| `paper/generate_figures_major_upgrade_v2.py` | Adds `figure_ei_evolution_money_plot`, a three-panel “money plot.” |
| `paper/echoforge_ieee.tex` | Adds `Evolution Trace and Locked Holdout Endpoint` subsection, figure, and audit-contract table. |
| `paper/validate_paper.py` | Requires the new PDF/PNG figure and required paper phrases. |
| `README.md` | Adds the advanced-evolution and strict trace commands to the paper lane. |

## 3. Reviewer-safe rule

The paper must not show holdout performance climbing across internal search attempts unless every plotted point is a separately locked, pre-declared holdout evaluation. Otherwise, a strong reviewer can fairly call it holdout shopping.

Therefore the plot contract is:

1. **Left panel:** candidate evaluation order versus train/CV objective, with every evaluated candidate as a light point and the running best as a bold line.
2. **Middle panel:** best train/CV AP by search stage: base mathematical lifts, surface controls, and meta-fusion.
3. **Right panel:** exactly one holdout endpoint, using the final selected candidate after `selection_lock.json`.

The figure caption and the body text must explicitly say: **the holdout is not used to draw the evolution curve.**

## 4. Data contract

### 4.1 Advanced runner output

The advanced runner must write:

```text
outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution/
  candidate_leaderboard.csv
  evolution_trace.jsonl
  evolution_trace.csv
  selection_lock.json
  fusion_quality_report.json
  performance_metrics.csv
  performance_summary.json
```

### 4.2 Required trace columns

`evolution_trace.csv` must contain these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Trace schema version, currently `ei-evolution-trace-v1`. |
| `candidate_index` | Actual candidate evaluation order, starting at 1. |
| `generation` | Coarse stage bucket for plotting. Not a biological claim. |
| `stage` | `base_candidate_search`, `surface_control`, or `meta_fusion_search`. |
| `candidate_id` | Full candidate handle. |
| `base_candidate_id` | Base candidate without final calibrator where applicable. |
| `candidate_type` | Advanced candidate, surface control, or meta-fusion candidate. |
| `family` | Candidate family. |
| `subset_name` | Feature/component subset label. |
| `head` | Candidate head or optimizer family. |
| `calibrator` | `raw`, `beta`, `geodesic_odds`, or `monotone_binning`. |
| `feature_count` | Count of source features/components used. |
| `component_count` | For meta-fusion, number of fused components. |
| `selection_split` | Must be `train_cv`. |
| `holdout_rows_used_for_selection` | Must be `0`. |
| `train_cv_objective` | Internal selection objective. |
| `train_cv_average_precision` | Internal CV AP. |
| `train_cv_roc_auc` | Internal CV ROC AUC. |
| `train_cv_f1` | Internal CV selected-threshold F1. |
| `train_cv_false_positive_rate` | Internal CV selected-threshold FPR. |
| `initial_take_up_average_precision` | Hardest-phase internal AP diagnostic. |
| `threshold` | Train/CV threshold for that candidate. |
| `running_best_candidate_id` | Best candidate seen so far under the selection tuple. |
| `running_best_objective` | Running-best internal objective. |
| `running_best_average_precision` | Running-best internal AP. |
| `running_best_roc_auc` | Running-best internal ROC AUC. |
| `selected_by_cv` | True only for the selected row. |
| `final_selected` | True only for the final locked row. |
| `trace_basis` | `true_evaluation_order` for valid evolution plot; `leaderboard_rank_only` for fallback. |
| `paper_note` | Guardrail text for review traceability. |

### 4.3 Paper evidence output

`detection/ei_evolution_trace.py` writes:

```text
outputs/paper-evidence/major-upgrade-v1/
  ei_evolution_trace.csv
  ei_evolution_summary.json
```

`ei_evolution_summary.json` contains:

| Field | Meaning |
|---|---|
| `trace_basis` | Whether the trace is true evaluation order or fallback. |
| `true_evaluation_order_available` | Must be `true` for the paper’s strict figure. |
| `candidate_evaluation_count` | Number of candidates in the trace. |
| `selected_candidate_id` | Candidate selected by train/CV. |
| `selected_row` | The selected candidate’s trace row. |
| `best_by_stage` | Best candidate per stage. |
| `holdout_endpoint` | One locked holdout endpoint. |
| `review_guardrail` | Text stating that holdout is not plotted as a search curve. |

## 5. Figure design

### Figure name

```text
paper/figures/ei_evolution_money_plot.png
paper/figures/ei_evolution_money_plot.pdf
```

### Panel A: Internal candidate evolution

- x-axis: `candidate_index`
- y-axis: `train_cv_objective`
- all candidates: light points, colored by stage
- running best: dark line using `running_best_objective`
- selected candidate: red star and annotation

### Panel B: Best stage result

- horizontal bars by stage
- value: best `train_cv_average_precision` within that stage
- label includes candidate count per stage

### Panel C: Locked holdout endpoint

- selected candidate ID
- holdout AP
- holdout ROC AUC
- selected-threshold FPR
- trace basis
- red guardrail note: no iterative holdout curve

## 6. Commands

Run the full lane in this order:

```bash
rtk python3 detection/generate_main_run.py \
  --profile fixed-wing-pusher-proxy-v2 \
  --out-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --scenario-groups 10000 \
  --seed 202605210136 \
  --force

rtk python3 detection/run_main_run_detectors.py \
  --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run \
  --folds 5 \
  --seed 202605210136 \
  --force

rtk python3 detection/run_advanced_main_run_detectors.py \
  --data-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --out-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --folds 5 \
  --seed 202605210136 \
  --search-profile v2_aggressive \
  --candidate-limit 128 \
  --evolution-rounds 5 \
  --evolution-sample-rows 4500 \
  --write-component-scores \
  --force

rtk python3 -m detection.ei_evolution_trace \
  --advanced-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --out-root outputs/paper-evidence/major-upgrade-v1 \
  --strict-trace

rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
rtk python3 paper/validate_visuals.py
rtk bash paper/build.sh --copy-tracked
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## 7. Acceptance criteria

The diff is accepted only if all checks pass:

```bash
test -s outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution/evolution_trace.csv
test -s outputs/paper-evidence/major-upgrade-v1/ei_evolution_trace.csv
test -s outputs/paper-evidence/major-upgrade-v1/ei_evolution_summary.json
test -s paper/figures/ei_evolution_money_plot.pdf
test -s paper/figures/ei_evolution_money_plot.png
```

And these semantic checks must hold:

1. `holdout_rows_used_for_selection == 0` for every trace row.
2. `selection_split == train_cv` for every trace row.
3. Exactly one row has `selected_by_cv == true`.
4. The selected row’s `candidate_id` equals `selection_lock.json:selected_candidate_id`.
5. `ei_evolution_summary.json:true_evaluation_order_available == true`.
6. `paper/generate_figures_major_upgrade_v2.py --strict` must fail if only `candidate_leaderboard.csv` exists and no true trace exists.
7. The paper must say the curve is train/CV-only and the holdout is a single endpoint after lock.

## 8. Reviewer red flags addressed

| Red flag | Mitigation in this diff |
|---|---|
| “This is just a final model, not evolution.” | Candidate-order trace and running-best line. |
| “You tuned on holdout.” | Trace fields explicitly record zero holdout rows for selection; figure shows only one endpoint. |
| “This is a sorted leaderboard pretending to be time.” | Strict mode rejects leaderboard-only fallback. |
| “Where did the selected candidate come from?” | Selection row is marked and matched to `selection_lock.json`. |
| “EI is a black box.” | Candidate family/head/calibrator/component counts are recorded for every point. |
| “The plot overclaims field performance.” | Caption and table keep the strict-open synthetic public-proxy claim boundary. |

## 9. Relationship to the existing paper

The existing paper already has the right discipline and core numbers: train/CV selection lock, one blind holdout pass, and the final low-FPR EI result. The gap is presentation and auditability. This patch does not change the claim boundary. It adds the missing search-trajectory evidence so the EI result becomes legible at first glance.

## 10. Relationship to the broader requested rewrite

This diff is intentionally scoped to the EI evolution “money plot.” The broader paper rebuild requested separately should follow as a second patch series:

1. Reorder the paper into a clearer tutorial flow: problem, simulation, scenarios, detector views, human baselines, fusion, EI, limitations.
2. Expand the radar simulation section with Monte Carlo details, waveform assumptions, clutter/RFI/noise stressors, and phase setup.
3. Add a sensor-fusion section before EI so the paper shows the natural climb from individual detectors to fusion to EI.
4. Move long source-code appendices into a one-column appendix or supplemental artifact.
5. For high-IP code, use a redacted sealed appendix with cryptographic hashes rather than implying the paper reveals operationally sensitive source.

The current patch creates the core EI trajectory infrastructure that the larger rewrite should build around.

## 11. Suggested final paper wording

Use this language in the EI section:

> The EI curve is an internal train/CV search trace. Each point is a candidate evaluated without holdout rows. The bold line is the running best under the declared selection objective. The blind holdout appears only once, after the selected candidate is written to `selection_lock.json`. This makes the evolution auditable without converting the holdout into a tuning surface.

This phrasing is short, reviewer-safe, and directly answers the expected critique.
