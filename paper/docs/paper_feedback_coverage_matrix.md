# Paper Feedback V5 Coverage Matrix

This matrix summarizes how the paper repair addresses actionable feedback in
`tips/paper_feedback/v5/`. The generated evidence lane also emits the full
machine-readable matrix under
`outputs/paper-evidence/major-upgrade-v1/feedback_coverage_matrix.*`.

| Tip | Action area | Status | Paper / evidence location |
| --- | --- | --- | --- |
| `tip1.txt` / `tip3.txt` / `tip5.txt` | Reader-facing model terminology changed from locked/selected candidate to Engineered Intelligence (EI) candidate | addressed | Abstract, Section IV, Figures 3-5, Tables IV-IX |
| `tip1.txt` / `tip3.txt` / `tip7.txt` | Author metadata changed to Jeppson Taylor and NeverHumqn Research Group | addressed | `paper/echoforge_ieee.tex` author block |
| `tip1.txt` / `tip3.txt` / `tip7.txt` | Abstract bounded to low-FPR synthetic benchmark behavior and lower ROC AUC caveat | addressed | Abstract |
| `tip1.txt` / `tip3.txt` / `tip6.txt` | Fig. 4 red recall tick overlay removed and replaced with grouped AP / Recall@<=1%FPR bars | addressed | `paper/generate_figures_major_upgrade_v2.py`, `phase_kpi.pdf` |
| `tip1.txt` / `tip3.txt` / `tip6.txt` | Fig. 5 EI component title, diverging delta colors, and numeric delta labels | addressed | `detector_ml_pipeline.pdf` |
| `tip3.txt` / `tip5.txt` | KTH anchor renamed from misleading algorithm figure to anchor overlay | addressed | `anchor_overlay.pdf`, TeX include, validators |
| `tip3.txt` / `tip5.txt` | Separate EI workflow figure added | addressed | `ei_workflow.pdf`, Section IV |
| `tip5.txt` / `tip7.txt` | Source, noise, detector, and assumption appendix added | addressed | Source ledger and assumption-family tables |
| `tip1.txt` / `tip4.txt` / `tip6.txt` | Validators ban reader-facing locked/selected candidate terminology and legacy figure names | addressed | `paper/validate_paper.py`, `paper/validate_visuals.py` |
| `tip1.txt` | Fig. 2, Fig. 3, Fig. 5, Fig. 6, Fig. 7, and Fig. 8 readability | addressed | Wide figures, physical axes, normalized anchor intervals |
| `tip1.txt` | Table IV / Fig. 3 / Fig. 5 metric consistency | addressed | `evaluation_summary.json`, `comparable_ablation_summary.csv`, Table IV |
| `tip1.txt` | Selected-threshold counts vs swept fixed-FPR metrics | addressed | `selected_threshold_confusion_matrix.csv`, `primary_kpi_table.csv` |
| `tip1.txt` | Group-level uncertainty and small positive holdout caution | addressed | `group_level_operating_metrics.csv`, Section IV, Limitations |
| `tip1.txt` | Radar model specificity and physical units | addressed | `radar_model_card.json`, `radar_model_detail_rows.csv`, Section II |
| `tip2.txt` | Strict-open definition and headline abstract result | addressed | Abstract, Claim Boundary |
| `tip2.txt` | FMCW/chirp versus pulse/CPI terminology | addressed | Section II, `radar_model_detail_rows.csv` |
| `tip2.txt` | Larger positive holdout / multi-seed evaluation | deferred | Limitations |
| `tip2.txt` | Track-level false-alarm rate | deferred | Limitations |
| `tip2.txt` | Canary wording as raw audit detection plus detector schema pass | addressed | Table VII, `detector_view_schema_evidence.csv` |
| `tip3.txt` | Radar-first positioning and related work | addressed | Claim Boundary, Section II |
| `tip3.txt` | ROC AUC vs AP tradeoff explanation | addressed | Abstract, main result paragraph, Table IV |
| `tip3.txt` | Human-readable sorted component table | addressed | Table V, `selected_component_human_weights.csv` |
| `tip3.txt` | Threshold source and metric basis | addressed | Table IV, `selected_threshold_confusion_matrix.csv` |
| `tip4.txt` | Scenario and positive-only balance outputs | addressed | `scenario_balance_by_dimension.csv`, `positive_balance_by_dimension.csv` |
| `tip4.txt` | Measured-anchor distance diagnostics | addressed | `anchor_distance_diagnostics.csv`, Fig. 8 |
| `tip4.txt` | Expanded limitations and prohibited inferences | addressed | Limitations and Prohibited Inferences |
| `tip1.txt` / `tip3.txt` | Fig. 6 chirp/CRF key compatibility | addressed | `paper/generate_figures_major_upgrade_v2.py`, `iq_drone_samples` |
| `tip1.txt` / `tip3.txt` | Fig. 7 shared range-Doppler color scale | addressed | `paper/generate_figures_major_upgrade_v2.py`, `iq_negative_samples` |
| `tip2.txt` / `tip4.txt` | Vector-backed non-heatmap figures and visual QA | addressed | `paper/figures/*.pdf`, `paper/validate_visuals.py` |
| `tip1.txt` / `tip2.txt` / `tip3.txt` / `tip4.txt` | Final-size IEEE figure system instead of 10-inch dashboards shrunk by LaTeX | addressed | `paper/generate_figures_major_upgrade_v2.py` shared IEEE visual layer |
| `tip1.txt` / `tip2.txt` | Fig. 6 radar model-card table and cue text overlap | addressed | `iq_drone_samples.pdf`, compact 2 x 3 layout with separate assumption, impairment, and cue cards |
| `tip1.txt` / `tip2.txt` | Fig. 7 row-label/y-axis collision and heatmap raster boundary | addressed | `iq_negative_samples.png`, raster-only 2 x 5 layout with dedicated label and colorbar columns |
| `tip2.txt` / `tip3.txt` | Fig. 8 KTH anchor readability in single-column placement | addressed | Full-width `figure*` and `anchor_overlay.pdf` vector include |
| `tip2.txt` | Passive-RF-only view exceeds full EI candidate on swept AP/Recall controls | addressed | Added Table IX caveat framing the EI candidate as pre-registered calibrated selected-threshold artifact |
| `tip1.txt` / `tip4.txt` | Regression guard for extensionless raster fallbacks | addressed | `paper/validate_paper.py` requires explicit `.pdf` / `.png` includes and forbids `iq_negative_samples.pdf` |
| `tip1.txt` / `tip3.txt` | Raster preview readability checks | addressed | `paper/validate_visuals.py` Pillow width, height, and nonblank checks; `paper/build.sh` installs Pillow if missing |
