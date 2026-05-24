# EchoForge Tier-1 Radar-Paper Rescue Spec v4

## Executive summary

The current paper is materially improved over the earlier versions: the author line is corrected, the abstract states the accepted human-engineered prior-fusion comparator, the primary KPI is explicit, Fig. 3 makes the +742% LCB95 and +185% point-recall result visible, and the appendix now contains public-proxy, environment, and detector-view model cards. The newest PDF also adds a data-processing path figure and a processing traceability checklist, which are exactly the right direction.

A tier-1 radar reviewer will still ask four hard questions:

1. **Can I reconstruct the entire data path without reading generator internals?** The current paper is closer, but the flow still jumps between roadmap, model, detector views, evaluation, EI, diagnostics, ledger, and appendix. The next version should present the story as: scenario -> synthetic radar/cue generation -> detector-view features -> baseline processing -> accepted fusion -> EI -> holdout KPI -> appendix model cards.
2. **Are the radar simulation assumptions best-practice, or just words?** The paper needs a dedicated, citable subsection on radar-simulation best practices: radar equation, FMCW/range-Doppler abstraction, CPI/PRF/range resolution, clutter models, receiver effects, calibration limits, and how EchoForge turns those assumptions into evidence rows.
3. **What are the detector-processing baselines and fusion baselines, exactly?** Radar reviewers will expect CFAR/MTD/track logic, acoustic/RF cue role, and late-fusion details to be described as a processing pipeline, not just as table labels.
4. **What is Engineered Intelligence, algorithmically?** EI must read as an auditable train/CV-only sparse calibrated fusion search, not as branding. The paper should include objective functions, search stages, selection lock, code-origin summary, internal-CV progress, and final holdout isolation.

This spec and the accompanying diff implement those changes.

## What is strong now

- **Main KPI is now visible.** The abstract clearly states LCB95 Recall@<=1%FPR, the accepted human-engineered prior-fusion comparator, +742% LCB95 gain, +185% point-recall gain, and the 49->1 selected false-positive reduction.
- **Claim boundary is much safer.** The paper repeatedly says the positive class is a fixed-wing pusher-prop public proxy, not measured truth or platform equivalence.
- **Fig. 3 is now close to a tier-1 result figure.** It distinguishes LCB95 lower-bound ticks from the 1% FPR operating cap and keeps AP/ROC/PR/calibration as diagnostics.
- **Appendix model cards now exist.** The Iranian-drone/public-proxy card, noise/RFI/clutter detail table, detector-view contract, and artifact map are the right reviewer-facing artifacts.
- **Validator gates are maturing.** The validator already catches stale terminology, missing KPI-gain text, missing evidence CSVs, and missing figure assets.

## Remaining weak spots and fixes

### 1. Flow is still too ledger-like

The paper still feels like a set of evidence artifacts stitched together. A tier-1 version should walk the reader through one clean story:

1. What question is being tested?
2. How is the synthetic radar/cue corpus generated?
3. How are scenarios sampled and phase-expanded?
4. What do detector-processing best practices do to the generated data?
5. What is the accepted human fusion comparator?
6. What does EI change?
7. What does the blind holdout say?
8. What are the boundaries and appendices?

**Patch actions**

- Add `Signal Simulation Best Practices and EchoForge Choices` in the main text.
- Add `Monte Carlo Scenario Construction` before detector views.
- Add `Detector-Processing Baselines` with CFAR/MTD/acoustic/RF/fusion references.
- Add `Sensor Fusion Baseline and Why It Matters` before EI.
- Move the most detailed ledger language to the appendix.

### 2. Radar simulation needs deeper best-practice framing

The current equations are good but insufficient. The paper should say how each radar-simulation assumption maps into a detector-view feature and what it cannot claim.

**Patch actions**

- Emit `simulation_best_practice_rows.csv/json` from `detection/paper_evidence_major_upgrade_v1.py`.
- Add a table describing radar equation, RCS/aspect, CPI/PRF, range-Doppler map, clutter, micro-Doppler, receiver impairment, acoustic cues, passive-RF cues, and detector-view contract.
- Add a main-text paragraph that the generated IQ is a synthetic intermediate and the main models consume detector-view summaries, not pixels from Fig. 9.

### 3. Monte Carlo setup needs a clearer section

The paper states 10,000 groups / 50 positives / 30,000 phase records but should spell out what is randomized and why.

**Patch actions**

- Emit `monte_carlo_setup_rows.csv/json`.
- Add a section explaining strata, split, phase expansion, positive rarity, hard-negative balancing, leakage guards, and repeated-phase group locking.
- Add reviewer note: row-level bootstrap is invalid; group-block bootstrap is required.

### 4. Detector-processing baselines need to be understandable

Radar reviewers know CFAR, MTD, tracking, acoustic cues, RF cues, and late fusion. They should see those methods in one plain section before EI appears.

**Patch actions**

- Emit `detector_processing_baseline_rows.csv/json`.
- Add a table that maps: X/Ku CFAR/range-Doppler concentration, S-band MTD, GBAD track/cue score, acoustic spectral cadence, passive-RF provenance/missingness, tabular ML, sequence ML, and accepted prior fusion.
- Add inline citations to Skolnik, Richards, Rohling, Kay, Van Trees, Fawcett, Davis/Goadrich, and Saito/Rehmsmeier.

### 5. Plot progression should build the narrative

The user wants familiar plots that build up: branches, fusion, EI. The current Fig. 5 shows phase behavior but only prior fusion and EI on the left. It does not show the detector branch ladder.

**Patch actions**

- Add `phase_method_ladder.pdf/png`.
- New figure shows the same three phases with method families layered: radar branches, prior ML, accepted fusion, EI. The KPI is Recall@<=1%FPR when available; AP is available as a secondary annotation.
- Validator requires the figure.

### 6. EI needs an innovation narrative without hype

The requested “agents for autonomous innovation” is exciting, but a tier-1 paper needs careful language. The safe framing is: decomposed search agents propose candidate feature families and fusion forms; internal train/CV objectives select sparse candidates; final holdout is untouched until the lock is written.

**Patch actions**

- Add `Engineered Intelligence: Decomposition, Evolution, and Locking` subsection.
- Emit `ei_objective_rows.csv/json` and `ei_search_progress_rows.csv/json`.
- Add `ei_search_progress.pdf/png` if search-trace artifacts exist; otherwise the figure still plots candidate-rank rows derived from available leaderboard/lock artifacts and marks it as train/CV-only.
- Add alternative KPI table: lower-bound low-FPR recall, calibration-aware F1, false-alarm family burden, group-level recall, cost-weighted detection, phase-minimum recall, and robustness under hard-negative families.

### 7. Code appendix and IP encryption need to be reconciled with strict-open

The user requested full source code in the appendix, origin-coded EI source, and an encrypted highest-IP section. This conflicts with the paper’s strict-open posture unless the paper is explicit: encrypted material is an **IP escrow appendix**, not required for reproducing the published KPI. Any code used to reproduce the paper’s claims must remain reviewable.

**Patch actions**

- Add `paper/code_appendix_manifest.json`.
- Add `paper/build_code_appendix.py`.
- Build script now generates `paper/generated_code_appendix.tex` before LaTeX.
- Appendix is one-column and uses `listings`/colored code styles.
- Human baselines and accepted fusion code are included with normal syntax highlighting.
- EI code is origin-coded line by line: human-standard lines default white; generative-origin/evolved-search lines light blue; IP-escrow lines light red and replaced by an encrypted block plus SHA-256 digest.
- Add validator gates for generated code appendix, encrypted excerpt metadata, and CLI appendix markers.

### 8. CLI reproduction needs an appendix

A reviewer should be able to run the full lane from commands.

**Patch actions**

- Add one-column `Reproducibility CLI Appendix` with commands for generating training data, running baseline detectors, running advanced EI lane, building evidence, figures, paper, and validator.
- Emit `cli_reproduction_rows.csv/json` from evidence generator.

## Expected paper flow after patch

1. Abstract: result and caveat.
2. Claim boundary and one-sentence experiment.
3. Core experiment roadmap.
4. Data-processing and evidence flow.
5. Radar simulation best practices and EchoForge choices.
6. Monte Carlo scenario construction.
7. Detector-processing baselines.
8. Sensor fusion baseline.
9. Evaluation protocol and KPI.
10. Main result figure/table.
11. EI method, search, locking, progress.
12. Diagnostics and limitations.
13. Rich appendix: modeling details, detector contracts, CLI, source-code appendix.

## Build commands

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1 \
  --training-root outputs/training-data/runit-fixed-wing-pusher-proxy-v2-main-run \
  --baseline-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run \
  --advanced-root outputs/detection/runit-fixed-wing-pusher-proxy-v2-main-run-advanced-evolution \
  --anchor-root outputs/real-data/kth-drone-bird-human-77ghz/kth-measured-v1 \
  --out-root outputs/paper-evidence/major-upgrade-v1 \
  --force

rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
rtk python3 paper/build_code_appendix.py --write-tex paper/generated_code_appendix.tex
rtk bash paper/build.sh --copy-tracked
```

## Validation additions

The validator should require:

- `phase_method_ladder.pdf/png`
- `ei_search_progress.pdf/png`
- `generated_code_appendix.tex`
- `code_appendix_manifest.json`
- `encrypted_ip_excerpt.json`
- `simulation_best_practice_rows.csv`
- `monte_carlo_setup_rows.csv`
- `detector_processing_baseline_rows.csv`
- `fusion_baseline_rows.csv`
- `ei_objective_rows.csv`
- `ei_search_progress_rows.csv`
- `cli_reproduction_rows.csv`
- page count 12--25

## Notes on encrypted IP appendix

Encrypting claim-critical source code is not compatible with strict-open reproducibility. The patch therefore treats the encrypted excerpt as an IP-escrow demonstration only. It must not contain the only copy of code required to reproduce the published KPI. The appendix explicitly states this boundary.

