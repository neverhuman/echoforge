# EchoForge V8 Narrative, Radar Simulation, and Source-Appendix Engineering Spec

## Delivered files

- `echoforge_v8_narrative_code_appendix.diff`
- `echoforge_v8_engineering_spec.md`

## Objective

Move EchoForge from a dense evidence report into a tier-1 radar benchmark paper that is easy to read on first pass. The new paper flow is:

1. State the experiment.
2. State the claim boundary.
3. Explain the end-to-end data path.
4. Explain synthetic radar simulation best practices.
5. Explain the take-up, climb, and cruise scenarios.
6. Explain the fixed-wing pusher-prop public proxy and hard negatives.
7. Explain detector views and accepted processing.
8. Explain sensor fusion.
9. Introduce the KPI.
10. Introduce Engineered Intelligence.
11. Show branch → fusion → EI plots.
12. Add a one-column appendix with modeling details, CLI reproduction, and source code.

## What is strong in the current PDF

The latest paper now fixes the `NeverHuman` typo, gives the main KPI in the abstract, includes the `+742%` LCB95 gain and `+185%` point-recall gain, and clearly states that the result is synthetic public-proxy evidence rather than measured platform truth.

The strongest technical parts are:

- group-locked splitting;
- explicit holdout scoring after selection lock;
- conservative low-FPR KPI;
- false-alarm family breakdown;
- strong claim boundary;
- compare-only measured anchor wording;
- current radar model card and appendix tables.

## What remains weak

### 1. Flow

The paper still reads like a bundle of evidence artifacts rather than a guided radar paper. The v8 TeX file fixes this by beginning with “What This Paper Tests,” then introducing data processing, simulation, detector views, fusion, KPI, and EI in that order.

### 2. Data processing clarity

A reviewer needs to see exactly how scenario groups become detector-view matrices and how labels/split keys are blocked. The patch adds `data_processing_flow.pdf` and `data_processing_trace_rows.csv/json`.

### 3. Radar simulation detail

The old paper had equations, but it needed an explanatory best-practices section. The patch adds a radar simulation section explaining target return, micro-Doppler, RCS/aspect, clutter/noise, receiver impairments, multipath/RFI, cue streams, and detector-view materialization.

### 4. Fusion narrative

The user asked for a familiar build-up plot: detector branches, then fusion, then EI across take-up, climb, and cruise. The patch adds `phase_method_progression.pdf`.

### 5. EI narrative

The paper should not merely say “EI did better.” It should explain decomposition, autonomous search, sparse nonnegative fusion, odds calibration, selection lock, and one holdout pass. The patch adds `EI Candidate Discovery and Locked Scoring Algorithm` and `ei_search_progress.pdf`.

### 6. Passive-RF-only red flag

The passive-RF-only view outperforming EI on AP and swept recall is not a contradiction, but it is a reviewer red flag. The patch makes this explicit and frames it as a future stress target for missingness-heavy and multi-seed corpora.

### 7. Source appendix

The user asked for full code in the appendix, origin coloring, and encrypted high-IP excerpts. The patch adds `paper/generate_source_appendix.py`, which generates a color-coded source appendix and a manifest.

## Files changed

### `paper/build.sh`

- Runs `paper/generate_source_appendix.py`.
- Builds `paper/echoforge_ieee_v8.tex` by default.
- Supports `PAPER_TEX=...` override.

### `paper/echoforge_ieee_v8.tex`

New entry point with the revised flow. It includes:

- colored KPI values;
- clear data-processing section;
- radar simulation best practices;
- detector processing references;
- sensor fusion section;
- phase progression plot;
- EI section;
- CLI reproduction appendix;
- generated source-code appendix.

### `paper/generate_source_appendix.py`

Generates `paper/generated/source_code_appendix.tex` and `code_appendix_manifest.json`.

Default appendix targets:

1. classical radar branch processing;
2. tabular ML baseline;
3. sequence ML baseline;
4. prior fusion;
5. EI sparse calibrated fusion.

Line backgrounds:

- human-reviewed source: light blue;
- generative-origin assisted source: light yellow;
- high-IP protected excerpt: light red with digest/ciphertext.

The protected excerpt is chosen by a deterministic dense-code heuristic. If `ECHOFORGE_IP_APPENDIX_KEY` is set, the script emits a reversible XOR-base64 ciphertext with SHA-256. Without the key, it emits SHA-256-only placeholders.

### `detection/paper_evidence_major_upgrade_v1.py`

Adds evidence outputs:

- `main_kpi_gain_table.csv/json`;
- `data_processing_trace_rows.csv/json`;
- `processing_best_practice_rows.csv/json`;
- `ei_objective_catalog_rows.csv/json`.

### `paper/generate_figures_major_upgrade_v2.py`

Adds figures:

- `data_processing_flow.pdf/png`;
- `phase_method_progression.pdf/png`;
- `ei_search_progress.pdf/png`.

`ei_search_progress` is intentionally honest: if no leaderboard exists, it emits a missing-artifact panel rather than inventing progress.

### `paper/validate_paper.py`

Requires the new figures, evidence files, v8 narrative sections, and source-code appendix manifest.

## No Rust changes required

No `.rs` files are required for this revision. The current paper, evidence, detector, and figure build path is Python and TeX. If a future Rust simulator is added, it should be added as a source appendix target.

## Apply

```bash
git apply /mnt/data/echoforge_v8_narrative_code_appendix.diff
```

## Build

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
rtk python3 paper/generate_source_appendix.py --strict
rtk bash paper/build.sh --copy-tracked
```

## Validate

```bash
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee_v8.tex \
  --bib paper/references.bib \
  --pdf target/paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Acceptance criteria

The revision is acceptable when:

- the paper builds from `paper/echoforge_ieee_v8.tex`;
- all new figures render;
- the source-code appendix is generated;
- `validate_paper.py` passes;
- the main text clearly separates selected-threshold counts, swept Recall@≤1%FPR, and LCB95;
- the passive-RF-only red flag is explicitly acknowledged;
- the appendix includes public-proxy modeling cards, environment modeling detail, detector-view contracts, CLI commands, and generated source code.
