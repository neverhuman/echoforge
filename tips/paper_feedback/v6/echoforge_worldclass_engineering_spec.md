# EchoForge world-class paper revision engineering spec

## Executive goal

Make the paper read like a serious radar/evaluation paper, not an evidence dump. The paper should lead with a tight experiment story:

1. What is the benchmark and claim boundary?
2. What is the core experiment?
3. What is the accepted human-engineered best-practice comparator?
4. What does EI change?
5. How much does EI improve the primary KPI?
6. Which diagnostic slices explain or limit that result?
7. Where can reviewers audit the Iranian-drone public proxy, noise/clutter/RFI, detector schemas, hard negatives, and source boundaries?

The main text should be short and direct. The appendix should be rich, explicit, and auditable.

## Current PDF review, by section and figure

### Page 1 / Abstract

Status: much better than the older version, but this should still be build-gated.

Required fixes:
- Keep `NeverHuman`, never `NeverHumqn`.
- Keep the headline KPI values in the abstract and color the values/gains.
- Use accepted best-practice fusion language instead of raw `layered_fusion_c2` language.
- Preserve the ROC AUC caveat: EI improves low-FPR operation, not universal rank dominance.

Required abstract values:
- Point Recall@≤1%FPR: 0.292 → 0.833, +0.541 pp, +185%, 2.85x.
- LCB95 Recall@≤1%FPR: 0.083 → 0.699, +0.616 pp, +742%, 8.42x.
- AP: 0.128 → 0.825, +0.697, about +545%.
- Selected-threshold FP: 49 → 1, 98.0% fewer false alarms.
- ROC AUC caveat: 0.917 versus baseline 0.938.

Code changes:
- `paper/echoforge_ieee.tex`: add xcolor KPI macros and abstract rewrite.
- `paper/validate_paper.py`: fail if NeverHumqn reappears or if the abstract loses the required gain statements.

### Figure 1 / Evidence stack

Assessment: readable, but it is still a process diagram rather than a radar figure. It is acceptable as the claim-boundary figure, but it should not be the main-result figure.

Recommended treatment:
- Keep as Fig. 1.
- Caption must explicitly say “synthetic products are not measured truth.”
- Do not add more detail; this is already enough.

### Figure 2 / Scenario balance and leakage diagnostics

Problem seen in the current/older page render:
- In the bad render, the title collides with the center top panel and the subtitle or bottom explanatory text crosses the x-axis area.
- Long hard-negative labels create cramped axes.
- The figure is visually useful, but it should not feel like a screen dump.

Required code fix:
- Disable overly tight constrained layout for this figure.
- Give the title and subtitle their own reserved top band.
- Remove bottom figtext that can overlap the x-axis.
- Increase hspace/wspace.
- Shorten hard-negative labels and add tick padding.

Patch target:
- `paper/generate_figures_major_upgrade_v2.py::figure_monte_carlo_split_flow()`.

Validation:
- `paper/validate_visuals.py` should render/check the figure and fail on banned labels or unreadable dimensions.

### Figure 3 / Main KPI

Problem:
- In the older version, the red vertical marker is visually unexplained; it is not obvious whether it is a threshold, a cap, a confidence bound, or a warning.
- The caption needs to define the red tick and the low-FPR line.
- The paper should not force readers to compute the gains from Table VI.

Required code fix:
- Add an annotation pointing to the red LCB marker: `red tick = LCB95 lower bound`.
- Shade the 1% FPR operating band in the ROC inset and label it `1% FPR operating limit`.
- Add a green headline: `gain vs best practice: +185% point recall and +742% LCB95; selected FP 49→1`.
- Keep AP/PR/calibration secondary.

Patch target:
- `paper/generate_figures_major_upgrade_v2.py::figure_kpi_ranking()`.

Validation:
- `paper/validate_visuals.py` should require the terms `gain vs best practice`, `red tick = LCB95 lower bound`, and `1% FPR operating limit` in `kpi_ranking.pdf`.

### Figure 4 / Phase behavior and false-alarm burden

Problem:
- The four-bar legend is too much: AP and Recall are both plotted for both methods, which creates four legend entries and hides the main point.
- The false-alarm panel stacks colors without a complete readable family legend in the older render.
- The reader cannot know what blue/black/orange represent.

Required code fix:
- Left panel should show only the operating KPI by phase: Recall@≤1%FPR for accepted fusion versus EI.
- Print AP as small diagnostic text per phase, not as bars.
- Right panel must include a visible family legend for every stacked color.
- Pale/hatched near-threshold totals must be labeled.
- Caption must explicitly state that bars are families, pale bars are near-threshold negatives, and values are diagnostic because each phase has only eight positive holdout records.

Patch target:
- `paper/generate_figures_major_upgrade_v2.py::figure_phase_kpi()`.

Validation:
- `paper/validate_visuals.py` should require `False-alarm family legend`, `Near-threshold total`, and `Fixed-FPR recall` in `phase_kpi.pdf`.

### Figure 5 / EI workflow

Problem:
- Text spills out of boxes in the older render.
- The stages are conceptually good, but the graphic is too compressed.

Required code fix:
- Shorten stage labels: `Detector views`, `Train/CV search`, `Sparse fusion`, `Odds calibration`, `EI selection lock`, `Blind holdout`.
- Increase canvas height and spacing.
- Replace long rail text in boxes with title/body cards.
- Clip text and add a `wrap_width` argument to `_add_box()`.

Patch target:
- `paper/generate_figures_major_upgrade_v2.py::_add_box()` and `figure_ei_workflow()`.

Validation:
- `paper/validate_visuals.py` should require `Blind holdout` and `Feature denylist` in `ei_workflow.pdf`.

### Figure 6 / EI components and controls

Assessment:
- This is useful, but it must remain diagnostic.
- The passive-RF-only view beats full EI on AP and swept low-FPR recall, so the paper must explain why the selected EI artifact is still reported: selected-threshold F1/ECE and pre-registered lock behavior.

Recommended text:
- Keep the “not universal dominator” caveat near Table IX and Fig. 6.
- Do not headline Fig. 6.

### Figure 7 / Radar model card

Assessment:
- This is strong as a compact assumptions panel.
- It should be paired with a richer appendix table, because the figure alone does not fully explain the Iranian-drone public proxy, RCS/aspect, launch/take-up, noise, RFI, clutter, and receiver impairment modeling.

Code/content changes:
- Add `paper/appendix_modeling.tex` with rich public-proxy and noise tables.
- Add evidence CSVs: `public_proxy_model_detail_rows.csv` and `environment_impairment_model_rows.csv`.
- Add `appendix_modeling_map.pdf/png` as an appendix review surface.

### Figure 8 / Range-Doppler proxy samples

Assessment:
- Useful, but the images are diagnostic, not proof.
- The caption correctly states axes and normalized log magnitude.

Recommended improvement:
- Keep the caption conservative.
- Do not claim visible separation from these samples alone.
- In a future upgrade, add a measured-anchor comparison panel or show multiple positive and hard-negative examples in appendix, not the main paper.

### Figure 9 / KTH anchor overlay

Assessment:
- The z-score normalization is the right idea.
- It must remain compare-only.

Required text:
- Make clear KTH is drone/bird/human at 77 GHz and is not a positive fixed-wing truth source.
- Keep it as an anchor sanity check only.

### Table VII / Confusion

Problem:
- In the older version, Table VII was a generic pass/fail EI transparency table with low value.
- In the newer version, `Main Experiment Lanes` is better, but it floats too late and still risks confusing readers.

Required code fix:
- Replace it with `Core Experiment Roadmap` near the beginning of the paper.
- It should answer: “What are the experiments? What is the accepted comparator? What does EI change?”
- Keep component/ablation details in the appendix and Fig. 6.

Patch target:
- `paper/echoforge_ieee.tex`.

## Rich appendix additions

### New appendix section: Rich Modeling Appendix

Add `paper/appendix_modeling.tex` and input it from `paper/echoforge_ieee.tex`.

Tables:
1. Fixed-Wing Pusher-Prop / Iranian-Drone Public-Proxy Modeling Card
   - Airframe geometry
   - Launch/take-up
   - Kinematics
   - Propulsion micro-Doppler
   - RCS/aspect envelope
   - Phase labels
   - Claim boundaries

2. Noise, Clutter, RFI, and Receiver Modeling Details
   - Thermal/SNR stress
   - Weibull/K-like clutter
   - Weather/sea/terrain clutter
   - Multipath ghosts
   - RFI/passive-RF missingness
   - Receiver impairment

3. Detector-View Modeling Contract
   - X/Ku radar
   - S-band radar
   - GBAD cueing
   - Acoustic cue
   - Passive RF
   - Fusion

4. Artifact map
   - `public_proxy_model_detail_rows.csv`
   - `environment_impairment_model_rows.csv`
   - `detector_view_schema_evidence.csv`
   - `false_alarm_by_method_family.csv`
   - `selected_component_human_weights.csv`

### Evidence generator additions

Patch `detection/paper_evidence_major_upgrade_v1.py` to write:
- `public_proxy_model_detail_rows.json/csv`
- `environment_impairment_model_rows.json/csv`

These files give reviewers a programmatic audit surface instead of forcing them to trust prose-only appendix tables.

### Appendix figure addition

Patch `paper/generate_figures_major_upgrade_v2.py` to generate:
- `appendix_modeling_map.pdf`
- `appendix_modeling_map.png`

The figure is a three-column review surface:
- Fixed-wing pusher-prop public proxy
- Environment + receiver stress
- Detector views + fusion

## Radar-expert gaps to keep explicit

These are not fatal, but the paper must be honest:

1. **Small positive holdout**: only 8 positive holdout groups and 24 positive holdout records. Group-block CIs are required, but broader claims need more seeds and more positive groups.
2. **Synthetic RCS and micro-Doppler priors**: the RCS range and prop cadence are public-proxy envelopes, not measured signatures.
3. **No track-level FAR**: current false-alarm reporting is record/group level, not operational track-level false alarms per hour/area.
4. **KTH mismatch**: KTH is useful but at 77 GHz and target-mix mismatch; it is compare-only.
5. **No hardware-in-loop / measured positive truth**: this must remain in limitations.
6. **Passive-RF-only diagnostic tension**: passive-RF-only has higher AP/swept low-FPR recall than full EI, so report EI as the locked calibrated selected-threshold artifact, not a universal dominator.
7. **Product-page references**: public sensor pages should be described as role-envelope anchors only. Product-page years should be treated as access/update metadata, not peer-reviewed evidence.

## Validation gates added

### `paper/validate_paper.py`

Add checks for:
- no `NeverHumqn` anywhere;
- `\kpigain{}` exists;
- abstract contains `+185%`, `+742%`, and `98% fewer false alarms`;
- the generic EI pass/fail table is replaced;
- rich appendix table titles are present;
- new appendix evidence manifest keys exist;
- page window widened to 8–18 pages because the rich appendix is intentional.

### `paper/validate_visuals.py`

Add checks for:
- figure text explains the red LCB marker and 1% FPR operating limit;
- Fig. 4 has a false-alarm family legend and near-threshold label;
- Fig. 5 contains blind holdout and feature-denylist language;
- appendix modeling map is generated and readable.

### `paper/build.sh`

Run visual validation before LaTeX compilation/validation, so unreadable figures fail CI.

## Apply/build commands

```bash
git apply echoforge_worldclass_paper_full_code_changes.diff
python3 -m detection.paper_evidence_major_upgrade_v1 --force
python3 paper/generate_figures.py --strict
python3 paper/validate_visuals.py
bash paper/build.sh --copy-tracked
python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Expected result

The revised paper should read as:

> EchoForge is a strict-open synthetic radar benchmark. The main experiment compares accepted best-practice fusion with EI sparse calibrated fusion on one blind group-locked holdout. EI improves the primary low-FPR KPI by +742% LCB95 and +185% point recall, with 98% fewer selected-threshold false alarms. The result is synthetic public-proxy evidence only, with detailed public-proxy, noise/clutter/RFI, detector, hard-negative, and source-boundary assumptions audited in the appendix.

