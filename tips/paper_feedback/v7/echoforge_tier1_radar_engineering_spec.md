# EchoForge Tier-1 Radar Paper Engineering Spec

## Executive finding

The current paper is much stronger than the first draft: the abstract now states the KPI gains, the author block is corrected, and the appendix begins to document the Iranian-drone public proxy and environment assumptions. The remaining gap is reviewer confidence: a tier-1 radar reviewer needs to see the full data-processing path, what the detector sees, what is blocked, how the synthetic radar assumptions propagate into detector views, and how the passive-RF-only diagnostic is handled.

This spec pairs with `echoforge_tier1_radar_review_patch.diff`.

## Strengths to preserve

1. **Abstract**: It correctly states the main KPI, comparator, holdout size, positive group count, +742% LCB95 gain, +185% point-recall gain, +545% AP gain, and 98% false-positive reduction.
2. **Claim boundary**: It repeatedly states synthetic public-proxy evidence only.
3. **Primary KPI**: LCB95 Recall@<=1%FPR is the right headline for rare-positive, false-alarm-constrained radar.
4. **Group-locked split**: This is a credible leakage defense.
5. **False-alarm family analysis**: This makes the paper feel like radar/system evaluation instead of generic ML.
6. **ROC AUC tradeoff**: The paper admits ROC AUC is lower for EI, which makes the low-FPR claim more credible.

## Weaknesses that an expert reviewer will flag

1. **Data processing path is not explicit enough**. The reviewer needs a single figure showing scenario sampling -> phase records -> radar/cue synthesis -> detector views -> denylist -> baseline/EI scoring -> evidence output.
2. **Radar simulation is still too abstract**. Equations alone are not enough. The paper needs a contract that names generated artifacts, model-visible outputs, and blocked fields.
3. **Passive-RF-only control is a red flag**. Since passive-RF-only has higher AP and swept low-FPR recall than full EI, the paper must say this is a diagnostic warning, not hide it.
4. **Main text still has too many explanatory tables**. Generic "Primary/Secondary" tables should be prose or appendix. Main tables should be only the result and the critical modeling cards.
5. **Appendix must be richer, not longer**. It needs explicit positive-proxy, RCS/aspect, launch/take-up, micro-Doppler, clutter, weather, RFI, multipath, receiver impairment, and detector-view contracts.

## Code changes included

### `paper/echoforge_ieee.tex`
- Rewrites contribution bullets around the single main experiment.
- Adds `Data Processing Path`.
- Adds `data_processing_pipeline.pdf`.
- Adds `Signal-Chain Data Processing Contract`.
- Removes the low-value primary KPI selection table.
- Adds explicit text treating passive-RF-only performance as a reviewer red flag.
- Adds a rich appendix subsection with positive proxy and environment modeling tables.

### `paper/generate_figures_major_upgrade_v2.py`
- Adds `data_processing_pipeline.png/.pdf` to vector outputs.
- Adds `figure_data_processing_pipeline`.
- Includes it in `generate_all`.

### `detection/paper_evidence_major_upgrade_v1.py`
- Adds `main_kpi_gain_rows`.
- Adds `data_processing_pipeline_rows`.
- Writes `main_kpi_gain_table.csv`.
- Writes `data_processing_pipeline_rows.csv/json`.

### `paper/validate_paper.py`
- Requires the new pipeline figure.
- Requires the new tier-1 clarity phrases.
- Requires the new evidence rows and CSV artifacts.

## Additional reviewer-facing recommendations

1. Keep Fig. 3 as the central result figure. It should always show the LCB95 red ticks, the 1% FPR operating cap, and direct gain language.
2. Keep Fig. 4 simple: fixed-FPR recall by phase on the left and false-alarm family burden on the right. Do not mix four legends or overloaded AP bars if page space is tight.
3. Add one sentence wherever passive-RF-only is discussed: this diagnostic must be tested under multi-seed and alternative missingness/RFI policies.
4. Do not add more main-text governance tables. Put audit files in appendix.
5. The next scientific upgrade is not prose; it is data: multi-seed runs, larger positive holdout, measured positive-class validation, hardware-in-loop hooks, and track-level FAR.

## Build commands

```bash
git apply /mnt/data/echoforge_tier1_radar_review_patch.diff

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
