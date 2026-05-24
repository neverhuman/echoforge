# EchoForge v8 World-Class Narrative and Radar Review Spec

## Executive goal

This revision restructures the paper so a tier-1 radar reviewer can follow the argument without reverse-engineering the repo. The new narrative is:

1. **Claim boundary** — synthetic public-proxy evidence only.
2. **Data-processing path** — scenario groups become synthetic IQ/cues, detector views, train/CV locks, and blind-holdout evidence.
3. **Simulation best practices** — Monte Carlo scenario construction, take-up/climb/cruise phases, fixed-wing pusher-prop proxy, hard negatives, clutter/noise/RFI, and receiver impairments.
4. **Detector processing** — standard radar/cue processing and references.
5. **Sensor fusion** — individual branches, prior ML, accepted fusion, then EI.
6. **KPI result** — LCB95 Recall@<=1%FPR is primary; selected-threshold counts and swept low-FPR recall are kept separate.
7. **Engineered Intelligence** — agentic decomposition, evolutionary train/CV discovery, sparse nonnegative fusion, geodesic-odds calibration, and one blind holdout pass.
8. **One-column appendix** — modeling details, CLI reproduction commands, code appendix, origin coloring, and encrypted high-IP excerpt.

The current PDF already states the headline result clearly: EI reaches LCB95 Recall@<=1%FPR of 0.699 versus 0.083 for accepted prior fusion, a +742% relative gain, and selected false positives drop from 49 to 1. The v8 patch preserves that but makes the route to the result easier to audit.

## Main expert-review gaps addressed

### 1. Flow and readability

**Problem:** The paper still reads like evidence fragments stitched together. A reviewer sees claims, figures, tables, and appendices, but not a simple mental model of the paper.

**Change:** Add simulation-first flow:

- `Simulation Design Best Practices`
- `Monte Carlo Scenario Construction`
- `Raw Data Processing Best Practices`
- `Detector Processing and Sensor Fusion`
- `Autonomous Algorithm Decomposition`
- `Reproducing the Simulation and Paper Evidence`
- `Source-Code Appendix`

These sections are explicit and sequential. They use the same terms as the figures and CSV artifacts.

### 2. Data-processing clarity

**Problem:** A radar reviewer needs to know exactly what exists at each stage: scenario groups, raw IQ references, detector summaries, train/CV rows, selection locks, holdout scores, and evidence outputs.

**Change:** Add `method_ladder_phase_kpi.pdf` and strengthen data-processing explanation. Add evidence CSV `main_kpi_gain_table.csv` and `cli_reproduction_commands.csv` so the PDF language is backed by generated artifacts.

### 3. Radar simulation depth

**Problem:** The core signal equations are not enough. The paper needs a plain-language best-practice section explaining target return, micro-Doppler, clutter/noise, receiver effects, cue streams, and feature contracts.

**Change:** Add a `Best-Practice Synthetic Radar Processing Checklist` and richer prose. This makes the simulation model legible without claiming operational fidelity.

### 4. Sensor fusion narrative

**Problem:** Jumping from baselines to EI makes the result look like a leaderboard jump. Fusion needs its own narrative bridge.

**Change:** Add a method ladder figure by phase: best sensor branch -> best prior ML -> accepted fusion -> EI. This makes fusion familiar before EI is introduced.

### 5. Engineered Intelligence explanation

**Problem:** EI was still described too much as a final model name and not enough as a process.

**Change:** Add an EI section explaining agentic algorithm decomposition, train/CV-only evolutionary search, sparse nonnegative fusion, calibration, selection lock, reviewability, and alternative KPIs.

### 6. Passive-RF-only red flag

**Problem:** Passive-RF-only has higher AP and swept low-FPR recall than the final EI candidate. A strong reviewer will notice.

**Change:** Keep the red-flag language. The paper now says this is an important diagnostic, not a contradiction to hide, and explains why the locked EI result is still the reported artifact: calibration, selected-threshold behavior, and selection-lock basis.

### 7. Appendix and source code

**Problem:** The requested appendix needs to be rich and reviewable, not just a collection of tables.

**Change:** Add `paper/generate_code_appendix.py`. It generates a one-column source-code appendix with:

- conventional human baseline source files;
- accepted fusion source files;
- EI/evolution source files;
- line-origin prefixes `HUM`, `GEN`, `ENC`;
- SHA256 digests;
- encrypted high-IP excerpt for roughly 10-15% of highest-value EI lines.

Because “full source code” and “encrypted high-IP code” are in tension, the spec resolves it this way: the repository remains authoritative and fully reviewable, while the printed paper appendix can mark and encrypt the highest-IP excerpt in the PDF artifact.

## Files changed

### `paper/echoforge_ieee.tex`

Adds code appendix packages and macros, restructures the main narrative, inserts simulation best practices, adds the method ladder figure, extends the EI explanation, adds CLI reproduction commands, and adds source-code appendix input.

### `paper/generate_figures_major_upgrade_v2.py`

Adds:

- `method_ladder_phase_kpi.pdf/png`
- `ei_discovery_progress.pdf/png`

The EI discovery figure uses `candidate_leaderboard.csv` if present; otherwise it shows an audit trace distinguishing train/CV discovery from blind-holdout scoring.

### `detection/paper_evidence_major_upgrade_v1.py`

Adds generated evidence artifacts:

- `main_kpi_gain_table.csv`
- `cli_reproduction_commands.csv`
- `cli_reproduction_commands.json`

### `paper/generate_code_appendix.py`

New script to generate `paper/generated_code_appendix.tex` and `code_appendix_manifest.json`.

### `paper/validate_paper.py`

Adds required figures, text-pattern checks for the tier-1 structure, and evidence-artifact checks.

## Acceptance commands

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
rtk python3 paper/generate_figures.py --strict
rtk python3 paper/generate_code_appendix.py \
  --out paper/generated_code_appendix.tex \
  --manifest outputs/paper-evidence/major-upgrade-v1/code_appendix_manifest.json
rtk bash paper/build.sh --copy-tracked
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Reviewer-facing success criteria

The revised paper should make the following impossible to miss:

- what was simulated;
- how take-up, climb, and cruise differ;
- how the Iranian-drone-like fixed-wing pusher public proxy is modeled without measured-truth claims;
- how birds, RC aircraft, clutter, weather, RFI, and receiver impairments are modeled;
- what detector features are model-visible;
- what audit and generator fields are blocked;
- why accepted prior fusion is the human comparator;
- how EI is discovered, locked, and scored;
- why the primary KPI is LCB95 Recall@<=1%FPR;
- why the passive-RF-only control is a diagnostic red flag;
- how to reproduce the evidence and figures;
- where to inspect source code and origin coloring.

## Risks and notes

- The code appendix can make the PDF long. Use the full version for the appendix build and a page-limited version for conference-length submission if needed.
- The encrypted high-IP excerpt should be used carefully: reviewers may dislike any encrypted source in an academic paper. The spec keeps repository source as authoritative and treats encrypted print blocks as IP-marked excerpts, not missing evidence.
- The EI progress plot is strongest if `candidate_leaderboard.csv` exists. If not, the fallback audit trace is honest but less persuasive.
- Multi-seed evaluation and larger positive holdouts remain the most important future-work items.
