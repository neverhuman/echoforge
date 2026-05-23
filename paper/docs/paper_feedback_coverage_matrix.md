# Paper Feedback V3 Coverage Matrix

This matrix summarizes how the paper repair addresses actionable feedback in
`tips/paper_feedback/v3/`. The generated evidence lane also emits the full
machine-readable matrix under
`outputs/paper-evidence/major-upgrade-v1/feedback_coverage_matrix.*`.

| Tip | Action area | Status | Paper / evidence location |
| --- | --- | --- | --- |
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
