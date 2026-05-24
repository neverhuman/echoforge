# EchoForge Tier-1 Paper Upgrade Engineering Spec

## Scope and deliverables

This package contains the implementation plan and patch for the next EchoForge IEEE paper revision. The requested deliverables are:

1. `echoforge_tier1_upgrade.diff` — a unified diff covering the paper TeX, appendix TeX, plotting code, evidence-generation code, validation code, build wrapper, source-code appendix renderer, sealed-IP tool, and appendix listing files.
2. `echoforge_tier1_engineering_spec.md` — this engineering specification.

The patch is designed to turn the paper from a strong but dense benchmark report into a tier-1 reviewer-facing manuscript with a clean narrative:

**claim boundary → scenario/data generation → radar/cue simulation → detector-view schema → human detectors → accepted fusion → EI sparse fusion → low-FPR KPI → appendices/code/evidence.**

## Critical review of the current paper

### What is already strong

The current manuscript has a valuable core. It clearly says EchoForge is a strict-open synthetic benchmark and not a measured-platform or fielded-sensor surrogate. It states the benchmark scale, the rare-positive setup, and the blind group-locked holdout. The headline result is compelling: EI reaches point Recall@≤1%FPR of 0.833 versus 0.292 and LCB95 Recall@≤1%FPR of 0.699 versus 0.083, while also improving AP and reducing selected-threshold false positives from 49 to 1. The abstract also includes the necessary ROC AUC caveat: EI has lower ROC AUC than the prior fusion baseline, so the claim is low-FPR operating behavior rather than universal ranking dominance.

The paper also has good evidence discipline: group locking, detector-view denylisting, leakage checks, label-shuffle sanity tests, metadata-only baselines, group-block bootstrap intervals, compare-only measured anchors, source cards, model cards, and explicit prohibited inferences.

### What a radar reviewer will still flag

A leading radar reviewer reading the current paper cold may still struggle with the flow. The paper contains many correct parts, but they arrive in a compressed order and sometimes feel like a ledger rather than a story. The first major risk is **cognitive load**: the reader needs a simple “what happens first, second, third” path before they see a large KPI claim.

The second risk is **simulation clarity**. The current paper names the radar equations, waveform assumptions, model card, and noise/RFI families, but it needs a more tutorial review surface: what is randomized, what is fixed, how Monte Carlo grouping works, what a phase record is, what the detector can and cannot see, and how raw IQ becomes detector features.

The third risk is **the positive-class boundary**. The paper correctly prohibits measured Iranian-drone radar truth, but the reviewer needs an appendix card that is impossible to misread: fixed-wing pusher-prop public proxy only, broad geometry/kinematics/cadence assumptions only, no payload/route/evasion/deployment claim.

The fourth risk is **fusion versus EI**. The current paper says prior fusion is the comparator, but the narrative should make fusion a first-class section before EI. The reader should see why single detector branches fail, why human-engineered fusion is the fair accepted-practice comparator, and only then why EI improves the low-FPR operating point.

The fifth risk is **the passive-RF-only diagnostic**. The current manuscript correctly admits that passive-RF-only has higher AP and swept Recall@≤1%FPR than the full EI candidate, while EI has better selected-threshold F1/ECE and a locked calibrated fusion contract. This should remain visible, because hiding it would be a red flag.

The sixth risk is **statistical fragility**. Eight positive holdout groups is not enough for broad operational claims. The patch keeps this limitation in the main text and marks phase/family slices as diagnostic.

The seventh risk is **appendix/source disclosure tension**. The user requested full source code plus encryption of the highest-IP 10–15%. Those goals conflict if “full source” means public Git plaintext. The patch resolves this by printing full paper-facing human reference implementations, printing the public EI wrapper, and sealing the high-IP EI transform core as ciphertext plus SHA-256 digest for authorized review.

## Files changed by the patch

### `paper/echoforge_ieee.tex`

Purpose: restructure the paper narrative and add tier-1 review surfaces.

Changes:

- Adds `listings`, `fancyvrb`, and `tcolorbox` for one-column code appendix and sealed-IP boxes.
- Adds a “Reader Roadmap” near the front so the paper reads as a linear chain rather than a metric ledger.
- Adds “What Exists at Each Stage” and a data-product contract table that separates scenario manifest, synthetic products, detector views, and paper evidence.
- Adds a simulation best-practice subsection and checklist table with references to radar fundamentals, CFAR, micro-Doppler, Monte Carlo, reproducibility, and measured anchors.
- Adds an explicit scenario setup table for initial take-up, climb transition, and cruise altitude.
- Adds a raw-data processing best-practices section covering range-Doppler FFT summaries, CFAR/OS-CFAR, MTD/M/N confirmation, micro-Doppler cadence, calibration, PR/ROC/ECE, and low-FPR KPI separation.
- Adds cumulative phase-KPI and tiered ranking figures so the visual grammar is familiar: branches first, fusion second, EI third.
- Adds a full sensor-fusion section before EI, including a fusion pipeline table.
- Expands EI into auditable algorithm evolution, not an opaque “AI model” claim.
- Adds an EI evolution trace figure and alternative KPI table.
- Switches appendices to one-column with `
\onecolumn` and adds `paper/appendix_source_code_listings`.

### `paper/appendix_modeling_details.tex`

Purpose: make the appendix answer radar-reviewer questions directly.

Changes:

- Adds Monte Carlo sampling contract.
- Adds positive public-proxy scenario explanation.
- Adds bird/RC/non-airborne hard-negative details.
- Adds scenario family setup table.
- Adds detector implementation detail table.
- Adds noise/impairment hierarchy and noise budget table.

### `paper/appendix_source_code_listings.tex`

Purpose: provide the one-column code appendix requested by the user.

Includes:

- Lineage color key.
- Full paper-facing human approach 1: classical CFAR/MTD radar baseline.
- Full paper-facing human approach 2: tabular/sequence ML control.
- Full paper-facing human approach 3: accepted layered fusion comparator.
- EI public wrapper and colored generated-origin lines when the generated appendix renderer is run.
- Sealed high-IP EI core digest/ciphertext block.
- CLI reproducibility appendix with concrete commands.
- Appendix acceptance criteria.

### `paper/listings/*.py`

Purpose: stable code listings for the appendix.

Files:

- `human_cfar_mtd_reference.py`
- `human_ml_reference.py`
- `human_fusion_reference.py`
- `ei_sparse_fusion_public.py`

These are paper-facing reference implementations. The production repository remains the implementation source of truth, but these listings make the paper self-contained and auditable.

### `paper/source_appendix_manifest.json`

Purpose: source-code appendix manifest for line-origin rendering.

Defines:

- Listing IDs and paths.
- Human/generative/mixed origin tags.
- `ORIGIN:GENERATIVE` marker.
- `IP_CORE_START`/`IP_CORE_END` sealed-block markers.

### `paper/render_code_appendix.py`

Purpose: generate `paper/generated/code_appendix_rendered.tex` with line-level coloring.

Behavior:

- Reads `paper/source_appendix_manifest.json`.
- Escapes TeX safely.
- Renders each source file into breakable `tcolorbox`/`Verbatim` blocks.
- Colors generated-origin lines light blue.
- Replaces sealed EI block with a red sealed notice.

### `paper/seal_ei_ip_core.py`

Purpose: encrypt the highest-IP EI block.

Behavior:

- Extracts source lines between `IP_CORE_START` and `IP_CORE_END`.
- Writes `paper/generated/ei_ip_core.sha256.txt`.
- Encrypts plaintext using OpenSSL AES-256-CBC/PBKDF2 when available.
- Requires passphrase via `ECHOFORGE_EI_IP_PASSPHRASE`.
- Has an explicit fallback mode for draft environments, but production review builds should use OpenSSL.

### `detection/paper_evidence_major_upgrade_v1.py`

Purpose: expand generated evidence so every new table/figure has an artifact source.

New evidence rows:

- `simulation_best_practice_rows.csv/json`
- `scenario_setup_rows.csv/json`
- `raw_processing_best_practice_rows.csv/json`
- `sensor_fusion_pipeline_rows.csv/json`
- `ei_alternative_kpi_rows.csv/json`

These rows are added to the paper evidence manifest so validation can fail if the paper has unsupported text.

### `paper/generate_figures_major_upgrade_v2.py`

Purpose: add familiar, cumulative plots that improve paper flow.

New figures:

- `all_runs_phase_kpi.pdf/png` — phase KPI progression from branches to fusion to EI.
- `tiered_kpi_ranking.pdf/png` — ranking plots for AP, Recall@≤1%FPR, and selected false positives.
- `detector_family_panels.pdf/png` — detector-family cards and claim boundaries.
- `ei_evolution_trace.pdf/png` — EI internal-CV discovery trace with one locked holdout marker.

### `paper/validate_paper.py`

Purpose: enforce the new tier-1 paper contract.

Changes:

- Requires the four new figures and previews.
- Requires the new evidence rows in the manifest.
- Requires the new CSVs.
- Requires key text markers such as simulation best practices, sensor fusion, source-code appendix, CLI appendix, and sealed EI core.
- Updates page-count validation from `8–18` to `12–28`, matching the user-requested longer paper with one-column appendix.

### `paper/build_tier1.sh`

Purpose: one-command tier-1 paper build wrapper.

Runs:

1. Paper evidence generation.
2. Figure generation.
3. Source-code appendix rendering.
4. Optional EI-core sealing when passphrase is set.
5. Visual validation.
6. PDF build.
7. Paper validation.

## Engineering acceptance criteria

The revision is accepted only if all of the following pass:

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
rtk python3 paper/render_code_appendix.py \
  --manifest paper/source_appendix_manifest.json \
  --out paper/generated/code_appendix_rendered.tex
ECHOFORGE_EI_IP_PASSPHRASE='<review-passphrase>' \
  rtk python3 paper/seal_ei_ip_core.py \
  --source paper/listings/ei_sparse_fusion_public.py \
  --out-dir paper/generated
rtk bash paper/build.sh --copy-tracked
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/build/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

For convenience:

```bash
ECHOFORGE_EI_IP_PASSPHRASE='<review-passphrase>' rtk bash paper/build_tier1.sh
```

## Reviewer-facing red flags and mitigations

| Red flag | Mitigation in patch |
|---|---|
| Paper is hard to follow | Adds reader roadmap and stage-by-stage data product contract |
| Simulation lacks detail | Adds simulation best-practice checklist, Monte Carlo contract, scenario setup, noise hierarchy |
| “Iranian drone” could be overread | Keeps “fixed-wing pusher-prop public proxy” as main text name and puts Iranian-drone wording in appendix-only boundary card |
| Detector views might leak labels/splits | Expands detector-view schema discussion and data product contract |
| Fusion comparator feels underspecified | Adds full sensor-fusion section and accepted comparator listing |
| EI feels opaque | Adds EI workflow expansion, component code appendix, evolution trace, origin classification, sealed core |
| Passive-RF-only control outperforms EI on some metrics | Keeps this diagnostic visible and explains why EI claim is locked calibrated selected-threshold artifact, not universal view dominance |
| Eight positive holdout groups are too few for broad claims | Preserves limitation and labels phase/family slices diagnostic |
| Plots are messy/inconsistent | Adds cumulative phase plot and tiered ranking plot with stable visual grammar |
| Appendix source/IP request is contradictory | Provides full paper-facing source and seals high-IP core as ciphertext plus digest |

## No Rust changes required

The requested upgrade is in the paper/evidence/plotting lane. I did not add Rust changes because the repo already exposes the paper build and simulation evidence through Python/TeX tooling, and there is no Rust-side API change required for the manuscript structure. If a future release wants a Rust CLI wrapper for the paper lane, it should delegate to the same evidence and validation scripts rather than duplicating logic.

## Notes on verification

I could not run the full patch against the live repository inside this sandbox because direct `git clone` failed DNS resolution. The diff is based on the uploaded PDF/prompt and fetched repository source files for the paper TeX, appendix TeX, paper evidence generator, plotting script, validator, and bibliography. The implementation intentionally uses generated evidence artifacts rather than hand-edited metrics so the paper can be rebuilt and audited.
