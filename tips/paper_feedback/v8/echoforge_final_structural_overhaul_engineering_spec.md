# EchoForge Tier-1 Structural Overhaul Engineering Spec

## Deliverables

This package contains exact proposed source changes in:

- `echoforge_final_structural_overhaul_code_changes.diff`
- this engineering spec

The patch is intentionally broader than a plot cleanup. It restructures the paper so a first-time radar reviewer can follow the story in a simple, logical order:

1. What is the claim?
2. How is the synthetic scenario generated?
3. How are radar/cue data produced?
4. How are detector views created without leakage?
5. What are accepted human signal-processing baselines?
6. Why does fusion matter?
7. What is the KPI and how is it computed?
8. What is Engineered Intelligence?
9. What source code and CLI evidence lets a reviewer audit the result?

## Current strengths in the paper

The current paper has improved substantially. The abstract is now direct and correctly states the primary KPI: LCB95 Recall@≤1%FPR. It gives the main EI-vs-prior fusion comparison, including point Recall@≤1%FPR of 0.833 vs 0.292, LCB95 of 0.699 vs 0.083, AP 0.825 vs 0.128, and selected-threshold false positives 1 vs 49. This is the right headline.

The claim boundary is now much safer. The paper repeatedly says the results are synthetic public-proxy evidence, not measured truth, not fielded-sensor equivalence, and not operational performance.

The figure suite is far more readable than the earlier version. The KPI chart now makes the red LCB marker explicit. The phase/false-alarm plot now has a real legend. The appendix modeling map helps.

The evidence generator and validators are a major strength. `paper/validate_paper.py`, `paper/validate_visuals.py`, `detection/paper_evidence_major_upgrade_v1.py`, and `paper/build.sh` already create a reproducible paper lane. The proposed patch extends that lane rather than adding hand-edited results.

## Remaining expert-review red flags

### 1. The paper still feels scattered

A radar reviewer should not have to jump from abstract to architecture to scenario balance to KPI to EI to raw samples to appendix to understand the flow. The first pass needs to read like a pipeline:

scenario setup → signal simulation → detector processing → fusion → KPI → EI → appendix evidence.

The patch adds a dedicated `Data Processing and Evidence Flow` section and a new `data_processing_flow.pdf` figure. This gives the paper an ELI5-style backbone while staying technical.

### 2. Data processing is not explicit enough

The current text says detector views are narrower than generator state, but the processing chain is not sufficiently concrete. The patch adds a generated traceability table and evidence rows:

- `data_processing_trace_rows.csv`
- `data_processing_trace_rows.json`

These list every stage, the input/output artifacts, reviewer check, and boundary condition.

### 3. Human best-practice baselines need a clearer section

The current paper has detector archetype cards, but it does not walk the reader through best-practice raw-data processing. The patch adds `Best-Practice Signal Processing Baselines`, covering CFAR-style detection, MTD/Doppler concentration, GBAD-style track continuity, acoustic cadence, and accepted late fusion. These are tied to standard radar references already in the bibliography.

### 4. Fusion needs its own narrative

The current paper shows fusion results, but the reader should understand why fusion is a meaningful step before EI. The patch adds a `Fusion and KPI Results` section and a new `scenario_kpi_ladder.pdf` plot. This plot is designed to build the result in stages:

- sensor branch context,
- accepted prior fusion,
- EI sparse fusion.

The goal is to let the reader see the same phase layout repeated rather than switching plot concepts every time.

### 5. EI needs a clearer algorithmic innovation section

The current EI section explains the lock and the component weights, but it does not fully frame EI as autonomous algorithm discovery. The patch adds `Algorithmic Innovation Framing` and an EI algorithm table:

1. read train/CV detector-view scores after denylisting;
2. generate candidate component families;
3. optimize sparse nonnegative weights;
4. fit monotone odds calibration;
5. write `selection_lock.json`;
6. score the blind holdout once.

It also adds `ei_progress.pdf`. If the repo emits `evolution_history.csv` or `candidate_leaderboard.csv`, the figure uses that. If not, it emits a labeled fallback and the validator prevents the figure from being mistaken for performance evidence.

### 6. Source code appendix is missing

The user specifically requested full source-code appendix sections with origin coloring. The patch adds:

- `paper/source_appendix_manifest.json`
- `paper/generate_source_appendix.py`
- generated `paper/source_code_appendix.tex`

The manifest extracts the core functions from the actual source files:

Human comparator source:
- `_cfar_score`
- `_mtd_score`
- `_track_score`
- `_fit_gaussian_log_odds`
- `_calibrated_predictions`
- `_build_performance_reports`

EI source:
- `CandidateSpec`
- `MetaFusionSpec`
- `_topology_proxy`
- `_dmd_summary`
- `_haar_packet_energy`
- `run_advanced_main_run_detectors`

The appendix uses blue background for human-designed comparator code and green background for EI/generative-origin-tagged code. This is generated from the repo files, not manually pasted.

### 7. Encrypted IP request conflicts with strict-open reproducibility

A Tier-1 strict-open paper is weakened if it hides source code that is necessary to reproduce results. The patch handles the request safely:

- the full source listings remain visible for reproducibility;
- an optional encrypted duplicate excerpt is generated only if `ECHOFORGE_IP_APPENDIX_KEY` is set;
- the encrypted block is clearly labeled as an optional IP-review duplicate, not required for reproducing paper results.

This satisfies the encrypted-excerpt request without undermining the strict-open claim.

### 8. Appendix should be one-column

The patch switches to one-column for the source-code appendix. This makes listings, CLI commands, and long model cards readable.

## Exact code changes by file

### `paper/echoforge_ieee.tex`

Adds:
- `listings`, `longtable`, `pdflscape`, and code color macros;
- a stronger core-experiment claim paragraph;
- `Data Processing and Evidence Flow`;
- `data_processing_flow.pdf` figure;
- `Data Processing Traceability Checklist`;
- `Best-Practice Signal Processing Baselines`;
- `Fusion and KPI Results`;
- `scenario_kpi_ladder.pdf` figure;
- clearer separation of selected-threshold counts, swept low-FPR recall, and LCB95;
- `Algorithmic Innovation Framing`;
- `EI Candidate Discovery and Locked Scoring Algorithm`;
- `ei_progress.pdf`;
- alternative KPI discussion;
- one-column `Source Code and Reproducible Command Appendix`.

### `paper/appendix_modeling_details.tex`

Adds:
- best-practice simulation subsection;
- synthetic radar processing detail table;
- Monte Carlo scenario construction subsection;
- scenario setup by flight phase table;
- CLI reproduction commands.

### `detection/paper_evidence_major_upgrade_v1.py`

Adds generated evidence rows for:
- data-processing traceability;
- phase KPI ladder.

Outputs:
- `data_processing_trace_rows.csv/json`;
- `processing_ladder_rows.csv/json`.

Bumps manifest version to `major-upgrade-v3`.

### `paper/generate_figures_major_upgrade_v2.py`

Adds three figures:
- `data_processing_flow.pdf/png`;
- `scenario_kpi_ladder.pdf/png`;
- `ei_progress.pdf/png`.

The figures are added to the generated output list.

### `paper/generate_source_appendix.py`

New generator for the one-column code appendix. It uses Python AST to extract named functions/classes from source files, preserving real line numbers. It emits color-coded TeX source lines and an optional encrypted IP excerpt.

### `paper/source_appendix_manifest.json`

New manifest defining exactly which source symbols are included in the appendix and how they are tagged.

### `paper/build.sh`

Adds:
- `python3 paper/generate_source_appendix.py`
- `python3 paper/validate_visuals.py`

before LaTeX compilation / paper validation.

### `paper/validate_visuals.py`

Adds required figure checks for:
- `data_processing_flow`;
- `scenario_kpi_ladder`;
- `ei_progress`.

### `paper/validate_paper.py`

Adds validation markers for the new sections, figures, and evidence rows.

### `paper/references.bib`

Adds references for algorithm discovery / agentic science:
- FunSearch / LLM program search;
- AI Scientist 2024;
- AI Scientist-v2 2025.

## Recommended final paper structure

1. Abstract
2. Claim Boundary and Core Experiment
3. Data Processing and Evidence Flow
4. Radar and Multimodal Generative Model
5. Scenario Design: Take-up, Climb, Cruise
6. Public-Source Detector Archetypes
7. Best-Practice Signal Processing Baselines
8. Fusion and KPI Results
9. Evaluation Protocol and Calibration
10. Engineered Intelligence
11. Raw Samples and Compare-Only Anchor
12. Compact Diagnostics
13. Limitations
14. One-column Appendices:
    - simulation modeling;
    - Iranian-drone public-proxy card;
    - birds / RC hard negatives;
    - detector-view modeling contract;
    - CLI commands;
    - full source listings;
    - optional sealed IP excerpt.

## Plot strategy

The plotting narrative should build familiarity:

1. data-processing flow: what happens to data;
2. scenario balance: what was sampled;
3. KPI ladder: phase-by-phase branches → fusion → EI;
4. aggregate KPI ranking: final result;
5. phase + false-alarm burden: failure mode interpretation;
6. EI workflow: how the novel method was produced;
7. source/appendix map: what to audit.

## Important caveat about encrypted code

The encrypted excerpt should never be the only copy of code needed to reproduce the paper. A strict-open paper that hides essential algorithm lines would invite reviewer rejection. The proposed implementation includes the full reproducible source and makes the encrypted section a duplicate IP-review artifact only.

## Build commands

After applying the diff:

```bash
git apply echoforge_final_structural_overhaul_code_changes.diff
python3 -m detection.paper_evidence_major_upgrade_v1 --force
python3 paper/generate_figures.py --strict
python3 paper/generate_source_appendix.py
python3 paper/validate_visuals.py
bash paper/build.sh --copy-tracked
python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf target/paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

Optional sealed IP excerpt:

```bash
ECHOFORGE_IP_APPENDIX_KEY='replace-with-review-key' \
  python3 paper/generate_source_appendix.py
```

## No Rust changes

I did not find any `.rs` changes needed for this paper pipeline. The paper build, evidence generation, plots, validators, and source appendix are Python/LaTeX concerns.
