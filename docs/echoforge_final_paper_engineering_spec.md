# EchoForge Final Paper Engineering Spec

## Goal

Replace the default paper lane with a focused final manuscript. The paper reads as a radar-first synthetic benchmark and then as an Engineered Intelligence (EI) result. It avoids a parallel `*_final` paper source.

## Implemented Lane

- `paper/echoforge_ieee.tex` is the default manuscript.
- `paper/build.sh` regenerates evidence, metric macros, the runtime code appendix, focused figures, the PDF, and validation.
- `target/paper/generated_metrics.tex` is generated from `outputs/paper-evidence/tier1-final` before LaTeX runs.
- The claim boundary remains synthetic public-proxy benchmark evidence only.

## Narrative Structure

The manuscript uses this order:

1. Introduction
2. Radar Processing Background
3. EchoForge Simulator
4. Benchmark Design
5. Human Detection Ladder
6. Engineered Intelligence
7. Results
8. False-Positive Burden
9. Why It Matters
10. Limitations
11. Conclusion
12. Appendices

The main body avoids generated CSV/JSON filename lists. Reproduction commands and generated paths appear only in the appendix or validation/build tooling.

## Metric Source

`paper/generate_metric_macros.py` reads:

- `main_kpi_gain_table.csv`
- `selected_threshold_confusion_matrix.csv`
- `split_summary.json`

from `outputs/paper-evidence/tier1-final` and writes LaTeX macros used by the abstract and results table. This keeps the title-page claims and KPI ledger aligned with regenerated evidence.

## Focused Figures

`paper/generate_figures_focused.py` generates the final figure set:

- `architecture_stack.pdf/png`
- `phase_method_ladder.pdf/png`
- `ei_evolution_money_plot.pdf/png`
- `false_alarm_breakdown.pdf/png`
- `radar_positive_vs_false_positive.pdf/png`
- `appendix_radar_samples.png`

Legacy diagnostic collages are no longer referenced by the manuscript.

## Runtime Code Appendix

`paper/runtime_code_appendix.py` contains only appendix-safe runtime math:

- robust scaling
- CFAR/range-Doppler score
- MTD/Doppler score
- track continuity score
- acoustic cadence score
- passive-RF provenance score
- accepted prior fusion
- geodesic-odds calibration
- EI sparse geodesic fusion

`paper/source_appendix_manifest.json` points only at these routines. `paper/generate_source_appendix.py` renders syntax-highlighted Python blocks, hides repository paths in the paper, and uses color-coded human/generated/mixed-origin blocks for EI line groups.

## Validation

The final validators require:

- final narrative sections in order
- generated metric macros
- focused figures and PNG previews
- no main-body artifact-file clutter
- no legacy clutter figure references
- no source-path labels or writer/report plumbing in the rendered code appendix
- strict public-proxy claim-boundary language

Proof commands:

```bash
rtk python3 -m py_compile paper/*.py
rtk bash paper/build.sh --copy-tracked
rtk just paper
rtk just fast
rtk git diff --check
```

## Remaining Boundary

The final paper does not claim measured platform truth, proprietary-equivalent behavior, classified fidelity, or operational performance. EI is claimed only as an improved low-FPR operating point on the declared synthetic public-proxy benchmark.
