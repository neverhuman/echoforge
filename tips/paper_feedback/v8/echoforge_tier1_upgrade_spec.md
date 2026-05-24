# EchoForge Tier-1 Paper Upgrade Engineering Spec

## Purpose

This spec defines the concrete repository changes needed to turn the current EchoForge IEEE draft into a much more reviewer-proof, tier-1 radar/sensing paper. The current draft already has the critical ingredients: a strict-open claim boundary, a group-locked synthetic corpus, a low-FPR primary KPI, an accepted prior-fusion comparator, a locked Engineered Intelligence (EI) candidate, and appendix model cards. The weak point is presentation: the story is still too scattered, the EI evolution evidence is not surfaced, and a radar expert reading quickly can still ask, “What exactly was simulated, how was it processed, what was fused, and how do I know the holdout was not tuned?”

The patch in `echoforge_tier1_upgrade.diff` is a targeted upgrade that adds:

1. a clean reader roadmap and flow;
2. simulation best-practice and Monte Carlo design sections;
3. detector-processing best-practice section with inline radar references;
4. sensor-fusion section with a familiar “branches → branches+fusion → branches+fusion+EI” ladder plot;
5. EI evolution money plot using `evolution_trace.jsonl` for train/CV incremental progress and a single locked holdout marker;
6. generated source-code appendix support with syntax highlighting, provenance coloring, and high-IP redaction/encryption envelope hooks;
7. CLI appendix with exact reproducibility commands;
8. validation gates so the new figures and sections cannot silently disappear.

## Current paper diagnosis

### Strong

- The paper has a strict claim boundary and repeatedly states that the benchmark is synthetic public-proxy evidence, not measured truth or operational performance.
- The core KPI is appropriate for this rare-positive, false-alarm-constrained benchmark: LCB95 Recall@≤1%FPR.
- The paper correctly separates selected-threshold counts from swept fixed-FPR recall and group-block LCB95.
- The comparison target is fair: accepted human-engineered prior fusion is the main comparator, not a weak single branch.
- The current EI section already admits the passive-RF-only diagnostic red flag rather than hiding it.
- The appendices already include the fixed-wing pusher-prop public-proxy card, noise/clutter/RFI model details, detector-view contract, and source ledger.

### Weak / confusing for a radar expert

- The reader does not get an “ELI5 but still technical” path before the paper starts presenting many tables.
- Radar simulation details are present, but not framed as best practices: waveform assumptions, phase windows, clutter/RFI/receiver stress, and detector-view contracts should be introduced in a more procedural order.
- Monte Carlo design is not explained in enough detail: scenario group sampling, phase expansion, group-locked split, stratification/balance, and why positive holdout is small but honestly reported.
- Detector processing methods are not introduced as a pipeline: range compression/range-Doppler, windowing, CFAR/MTD, micro-Doppler summaries, cue extraction, calibration, and leakage guards.
- Fusion is currently implied more than narrated. A reviewer needs to see what each branch contributes, why late fusion is the accepted-practice comparator, and how EI sits on top of the same detector-view schema.
- The “money plot” is missing: EI evolution should show internal-CV candidate improvement over search order and clearly annotate that holdout was scored once after `selection_lock.json`.
- The paper does not yet include a source-code appendix pipeline. Manually pasting code into TeX would be brittle; it should be generated from the repo.
- The paper validates several figures, but it does not require the new EI evolution plot, KPI ladder, CLI appendix, or generated source appendix.

## Design rules for the patch

### Claim-boundary rules

- Do not claim measured Iranian-drone radar signatures.
- Do not claim platform truth, route behavior, evasion modeling, payload inference, sensor parity, classified fidelity, proprietary equivalence, or operational FAR.
- Use “fixed-wing pusher-prop public proxy” in reader-facing prose.
- Treat KTH/Open Radar/RAD-DAR/RDRD as compare-only anchors.
- Treat phase and false-alarm-family slices as diagnostics because holdout has only eight positive scenario groups.

### EI evolution rules

- Plot train/CV evolution from `evolution_trace.jsonl`.
- Plot best-so-far internal objective/AP envelope.
- Show the final locked holdout result once, after the selection-lock marker.
- Do **not** plot a repeated holdout-improvement curve unless a future artifact explicitly logs repeated, predeclared holdout-free external validation. The current code and paper are stronger because holdout is not repeatedly touched.
- Include `holdout_rows_used_for_selection = 0` directly in figure annotation and evidence CSV.

### Source-code appendix rules

- Generate source appendix from a manifest, do not hand paste code.
- Keep syntax highlighting stable with `listings`/`tcolorbox` and avoid minted/Pygments dependency unless intentionally added later.
- Color-code code blocks by provenance category:
  - human-standard source: neutral/white;
  - generated/EI-influenced source: light blue background;
  - generated with high-IP redaction: blue background plus encrypted/redacted envelope.
- Do not falsely claim cryptographic security unless the pipeline uses a real encryption tool. The patch uses a “redaction envelope” and optional external ciphertext fields. The default paper build prints hashes, line ranges, and withheld block labels.
- Keep high-IP withheld snippets around 10–15% of the EI source appendix by configured line ranges, not by ad hoc text deletion.

## Files changed by the patch

### `paper/echoforge_ieee.tex`

Adds or modifies:

- `\usepackage{listings}`, `\usepackage{tcolorbox}`, `\usepackage{multicol}`, `\usepackage{xurl}` and source-appendix colors.
- New section `Reader Roadmap and Review Contract`.
- New section `Simulation Best Practices and Monte Carlo Design`.
- Expanded `Radar and Multimodal Generative Model` with explicit target-return/clutter/receiver/cue steps.
- New section `Detector Processing Best Practices`.
- New section `Sensor Fusion and Score Evidence` with KPI ladder figure.
- Expanded `Engineered Intelligence` with `ei_evolution_curve.pdf` and `ei_evolution_summary.csv` evidence.
- New appendix sections:
  - `How to Run EchoForge and Rebuild the Evidence`;
  - `Source-Code Appendix and Provenance Coloring`;
  - generated `\input{paper/source_appendix_generated}`.
- Stronger conclusion centered on what the result proves and what it does not prove.

### `paper/generate_figures_major_upgrade_v2.py`

Adds:

- `ei_evolution_curve.png/pdf` to vector outputs.
- `kpi_ladder.png/pdf` to vector outputs.
- `FigureContext.evolution_trace`.
- `_load_evolution_trace()` parser for JSONL.
- `_metrics_by_method()` helper.
- `figure_ei_evolution_curve()` for the money plot.
- `figure_kpi_ladder()` for the progressive branch/fusion/EI ranking.
- `generate_all()` calls for both new figures.

### `paper/summarize_ei_evolution.py` (new)

Adds a standalone paper-facing evidence summarizer for the EI trace. It reads `evolution_trace.jsonl`, `selection_lock.json`, `fusion_quality_report.json`, and `performance_summary.json` from the advanced detector root, then emits:

- `ei_evolution_summary.csv`;
- `ei_evolution_summary.json`.

This avoids changing the detector scoring path and keeps the money-plot evidence strictly post-run.

### `paper/source_appendix_manifest.json` (new)

Defines appendix source blocks:

- human baseline detectors from `detection/main_run_detectors.py`;
- accepted prior fusion source block;
- advanced EI source from `detection/advanced_main_run_detectors.py`;
- figure/evidence generator snippets.

The manifest supports line ranges, language, provenance category, and high-IP withheld ranges.

### `paper/build_source_appendix.py` (new)

Generates `paper/source_appendix_generated.tex` from the source manifest.

Key behavior:

- Reads source files by configured line ranges.
- Emits `lstlisting` blocks with background colors by provenance category.
- Replaces configured high-IP line ranges with a redaction/encryption envelope that includes source path, line range, SHA-256 digest, and optional ciphertext path if available.
- Writes a summary table at the top of the appendix.
- Requires no network and no generated arrays.

### `paper/validate_paper.py`

Adds required figures:

- `ei_evolution_curve.pdf/png`;
- `kpi_ladder.pdf/png`.

Adds required text gates:

- `Reader Roadmap and Review Contract`;
- `Simulation Best Practices and Monte Carlo Design`;
- `Detector Processing Best Practices`;
- `Sensor Fusion and Score Evidence`;
- `EI Evolution Trace`;
- `How to Run EchoForge and Rebuild the Evidence`;
- `Source-Code Appendix and Provenance Coloring`;
- `ei_evolution_summary.csv`;
- `source_appendix_generated`.

### `paper/build.sh`

Adds pre-build steps:

```bash
rtk python3 paper/summarize_ei_evolution.py --force
rtk python3 paper/build_source_appendix.py --manifest paper/source_appendix_manifest.json --out paper/source_appendix_generated.tex --force
```

This keeps the EI evolution summary and source appendix synchronized with the repository.

### `paper/references.bib`

Adds references for:

- data fusion and tracking: Hall/McMullen, Blackman/Popoli, Bar-Shalom et al.;
- sensor fusion / information fusion: Waltz/Llinas;
- CFAR variants: Gandhi/Kassam;
- evolutionary algorithms: Storn/Price, Deb et al., Koza;
- reproducible code appendices / literate programming: Knuth.

## Expected output figures after patch

1. `architecture_stack.pdf/png` — claim boundary stack.
2. `data_processing_flow.pdf/png` — scenario-to-holdout pipeline.
3. `monte_carlo_split_flow.pdf/png` — balance/leakage checks.
4. `kpi_ladder.pdf/png` — progressive branches → fusion → EI ranking.
5. `kpi_ranking.pdf/png` — primary KPI + ROC/PR/calibration.
6. `phase_kpi.pdf/png` — per-phase fixed-FPR recall and false-alarm burden.
7. `ei_workflow.pdf/png` — EI workflow boundaries.
8. `ei_evolution_curve.pdf/png` — **money plot**: train/CV incremental EI evolution + one final holdout marker.
9. `detector_ml_pipeline.pdf/png` — EI components and comparable controls.
10. `iq_drone_samples.pdf/png` — radar model card.
11. `iq_negative_samples.png` — range-Doppler sanity panels.
12. `anchor_overlay.pdf/png` — compare-only KTH overlay.
13. `appendix_modeling_map.pdf/png` — appendix modeling map.

## Rebuild sequence

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
  --search-profile v2_aggressive \
  --candidate-limit 128 \
  --evolution-rounds 5 \
  --evolution-sample-rows 4500 \
  --write-component-scores \
  --force

rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/summarize_ei_evolution.py --force
rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
rtk python3 paper/build_source_appendix.py \
  --manifest paper/source_appendix_manifest.json \
  --out paper/source_appendix_generated.tex
rtk python3 paper/validate_visuals.py
rtk bash paper/build.sh --copy-tracked
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Reviewer-risk checklist

- **Holdout leakage risk:** mitigated by explicitly plotting train/CV evolution and final holdout only once.
- **Synthetic overclaim risk:** mitigated by repeated public-proxy boundaries and appendix red lines.
- **Iranian-drone claim risk:** mitigated by fixed-wing pusher-prop proxy language and prohibited-inference list.
- **Small positive holdout risk:** mitigated by group-block bootstrap and limitations language.
- **Passive-RF-only red flag:** retained and explained as a diagnostic view, not a selected calibrated artifact.
- **Figure clutter risk:** mitigated by progressive ladder plot plus dedicated KPI/evolution figures instead of one overloaded plot.
- **Code appendix staleness risk:** mitigated by generated appendix manifest and build gate.

## Acceptance criteria

- `paper/validate_paper.py` passes.
- `paper/validate_visuals.py` passes.
- `ei_evolution_curve.pdf` exists and has no fallback metadata in strict mode.
- `kpi_ladder.pdf` exists and has no fallback metadata in strict mode.
- `outputs/paper-evidence/major-upgrade-v1/ei_evolution_summary.csv` exists and contains `holdout_rows_used_for_selection` with zero for trace rows.
- `paper/source_appendix_generated.tex` is generated before TeX compilation.
- No Rust changes are required for this paper-focused patch; Rust/Studio surfaces remain unchanged except for optional future UI exposure of paper artifacts.
- The final PDF contains the EI evolution plot, KPI ladder, CLI appendix, and source appendix.
- No phrase in the paper implies measured platform truth or operational performance.
