# EchoForge Tier-1 Paper Upgrade — Engineering Spec

## Deliverables

This bundle contains two files:

1. `echoforge_tier1_upgrade.diff` — a repo patch that adds paper-evidence rows, three new figures, a clearer paper structure, a one-column source appendix, code-provenance keying, encrypted high-IP redaction hooks, and paper validation gates.
2. `echoforge_tier1_engineering_spec.md` — this engineering spec and review memo.

I could not run the patch against the repository in the sandbox because direct `git clone` failed from the container due to DNS/network resolution. The diff is based on the uploaded `echoforge_ieee.pdf`, the current public GitHub README/paths, and fetched GitHub snapshots of `paper/echoforge_ieee.tex`, `paper/generate_figures_major_upgrade_v2.py`, and `detection/paper_evidence_major_upgrade_v1.py`.

---

## Executive summary

The current EchoForge paper is already credible in one important respect: it does not over-claim measured truth. It clearly defines EchoForge as a strict-open synthetic public-proxy benchmark, reports a group-locked holdout, centers the KPI on LCB95 Recall@≤1%FPR, and explicitly says the EI result is a low-FPR operating-point gain rather than universal rank dominance. It also has real evidence artifacts, leakage checks, calibration, phase diagnostics, and a fixed-wing pusher-prop public-proxy appendix.

The paper still reads too much like an evidence ledger assembled after the fact. A tier-1 radar reviewer will want the story in this order:

1. What exactly is being simulated?
2. What is the waveform/resolution budget?
3. What is a scenario group, and how do take-up, climb, and cruise differ?
4. What are the positive proxy, birds, RC aircraft, clutter, RFI, noise, and receiver impairments?
5. What raw data is generated?
6. What detector-view features are allowed?
7. What do the human detectors do?
8. What does accepted fusion add?
9. What exactly did EI search, lock, and score?
10. Why should I trust the numbers given the tiny positive holdout?

The diff makes that path explicit without changing any result number or expanding the claim boundary.

---

## What is strong in the current paper

### 1. Claim boundary is unusually careful

The abstract and first section already state that strict-open means assumptions, code, configuration, detector-view schemas, and evidence are reviewable, while generated arrays and measured raw files remain outside Git. The paper also says the positive class is a fixed-wing pusher-prop public proxy, not a measured-platform claim or fielded-sensor surrogate. This is exactly the kind of boundary a radar reviewer needs before reading synthetic performance results.

### 2. The headline KPI is correct for the problem

Rare-positive low-altitude UAS detection is false-alarm constrained. A point AP or AUC headline would be weaker. LCB95 Recall@≤1%FPR is a stronger headline because it includes the low-FPR operating regime and group-block uncertainty.

### 3. The paper already separates selected-threshold counts from swept low-FPR recall

This is a major strength. The current Table VIII style is the right idea: selected-threshold TP/FP/FN, selected FPR, swept Recall@≤1%FPR, and LCB95 must remain different columns.

### 4. The paper acknowledges an EI red flag instead of hiding it

The passive-RF-only view has higher AP and swept Recall@≤1%FPR than the full EI candidate, while the full EI has better selected-threshold F1/ECE and carries the pre-specified calibrated selection lock. A tier-1 reviewer will notice this. The current paper deserves credit for saying it plainly.

### 5. The appendix already has the right guardrails

The fixed-wing pusher-prop / Iranian-drone public-proxy appendix says no measured Iranian-platform signature, no route/evasion model, no payload inference, and no deployment-performance claim. That needs to stay prominent.

---

## What is weak or confusing

### 1. Flow is still too ledger-like

The paper has good pieces, but a first-time radar expert has to mentally reconstruct the pipeline. The diff adds a “Reader Roadmap” and moves the narrative toward:

**scenario → synthetic sensing → detector view → human detectors → accepted fusion → EI → holdout KPI → appendix proof.**

### 2. Radar simulation details are compact enough to raise questions

A reviewer may ask:

- Is this FMCW, pulse-Doppler, or a hybrid proxy?
- What does 20 range bins mean? Is that total surveillance range or a local crop?
- Does a 6 ms CPI actually support the stated micro-Doppler cadence interpretation?
- Do target speeds exceed unambiguous velocity?
- Are Doppler folding, PRF ambiguity, and receiver impairments truly modeled or just named?
- What are the distributions for clutter, weather, terrain glint, RFI, and multipath?

The diff adds `signal_simulation_best_practice_rows.csv`, `monte_carlo_scenario_trace_rows.csv`, and expanded TeX text that calls out local range crop, Doppler resolution, micro-Doppler sampling limits, and Doppler folding.

### 3. Monte Carlo setup needs more “how it works” clarity

The current figures show group counts and splits, but the paper should explain in simple terms:

- A scenario group is the event-level unit.
- Each group expands into take-up, climb, cruise phase records.
- Split happens before phase expansion.
- The group lock prevents phase leakage.
- Phase slices have only eight positive holdout groups and are diagnostic.

The diff adds a Monte Carlo trace table with this language.

### 4. Detector modeling needs a crosswalk from “detector name” to “code and assumptions”

The current detector archetype section is good, but the paper should make the mapping explicit:

- CA-CFAR / OS-CFAR / MTD / micro-Doppler summary / calibration.
- What features each sees.
- What it cannot support.
- Which source listing implements or wraps it.
- Which references justify the method family.

The diff adds `detector_processing_best_practice_rows.csv` and `detector_source_lookup.pdf`.

### 5. Sensor fusion should be its own section

Right now, accepted fusion appears as a comparator, but the paper should explain why fusion is powerful in this setting:

- Different sensors fail differently.
- Acoustic is affected by weather/traffic.
- Passive RF can be absent or contaminated.
- GBAD can be stale/horizon-limited.
- Radar can face clutter, multipath, folding, and glint.
- Late fusion gives a reviewable way to combine partial evidence.

The diff adds `sensor_fusion_pipeline_rows.csv` and a new “Sensor Fusion as a First-Class Baseline” section.

### 6. EI language needs to be powerful but less hype-prone

“Engineered Intelligence” can be compelling if framed as observable algorithm evolution, not magic. The paper should say:

- The problem is decomposed into view families and score transformations.
- Search is train/CV-only.
- The final lock is sparse and calibrated.
- The holdout is scored once.
- Source is reviewable and not a black box.
- Generative-origin LOC is self-classification, not authorship proof.
- Alternative objectives can evolve later.

The diff adds an “Observable Engineered Intelligence, Not a Black Box” subsection and `ei_iteration_trace_rows.csv`.

### 7. The passive-RF-only control requires explicit stress tests

A reviewer may suspect leakage or proxy shortcut learning. The paper already notes the red flag, but tier-1 readiness needs more:

- Passive-RF feature shuffle.
- Passive-RF missingness stress.
- Provenance-only ablation.
- Site/range/aspect metadata-only comparisons.
- OOD holdout where passive-RF provenance is perturbed.
- Calibration/ECE comparison.

The diff requires the text to discuss this red flag. The engineering backlog should add the above stress tests before any broader claim.

### 8. Source code appendix needs a safe mechanism

The user asked for full source code and high-IP encryption. The safe approach is:

- Full listings via `\lstinputlisting` in a one-column appendix.
- Provenance key with human-standard, generative-inspired, and encrypted/redacted classes.
- An encryption hook that never commits the key.
- A digest-only fallback when the key is missing.
- Generated artifact `encrypted_ip_redaction_rows.csv` with method ID, source path, encrypted fraction, SHA-256, status, and note.

The diff implements this pattern.

---

## Files changed by the diff

### `detection/paper_evidence_major_upgrade_v1.py`

Adds paper-facing evidence rows:

- `signal_simulation_best_practice_rows.csv`
- `monte_carlo_scenario_trace_rows.csv`
- `detector_processing_best_practice_rows.csv`
- `sensor_fusion_pipeline_rows.csv`
- `ei_iteration_trace_rows.csv`
- `source_code_appendix_inventory.csv`
- `encrypted_ip_redaction_rows.csv`

Also adds an appendix code target inventory for:

- high-resolution X/Ku branch
- tactical S-band branch
- GBAD branch
- accepted layered fusion
- EI sparse calibrated late fusion

The encryption hook uses `ECHOFORGE_IP_REDACTION_KEY`. The key must never be committed.

### `paper/generate_figures_major_upgrade_v2.py`

Adds three figures:

1. `method_progression_ladder.pdf/.png`
   - Shows three human radar branches, accepted fusion, and EI by take-up/climb/cruise phase.
   - Keeps the visual progression familiar and readable.

2. `detector_source_lookup.pdf/.png`
   - Maps detector family to paper role, references, and appendix code.

3. `ei_iteration_progress.pdf/.png`
   - Shows progress only if evidence rows exist.
   - If iteration data is absent, displays an audit card rather than inventing holdout progress.

### `paper/echoforge_ieee.tex`

Adds:

- Code-listing packages and styles.
- Reader roadmap.
- Simulation best-practice text.
- Detector-view processing section.
- Sensor fusion section.
- EI observability section.
- Progression, detector lookup, and EI progress figures.
- One-column appendix.
- CLI reproduction commands.
- Source-code appendix key.
- Human detector source listing.
- EI source listing.
- Encrypted high-IP redaction notice.

### `paper/validate_paper.py`

Adds required figures and required terms:

- `method_progression_ladder`
- `detector_source_lookup`
- `ei_iteration_progress`
- “Reader Roadmap”
- “Simulation Best Practices”
- “Sensor Fusion as a First-Class Baseline”
- “Observable Engineered Intelligence”
- “Source-Code Appendix Key”
- “encrypted high-IP”
- “passive-RF-only control”

Also adds additional prohibited-overclaim patterns.

### `paper/docs/tier1_upgrade_spec.md`

Adds a repo-local copy of the implementation spec.

---

## No Rust changes required

No `.rs` changes are required to deliver this paper upgrade. The existing Rust Studio lane already exposes run/artifact review workflows through the Studio API; the requested deliverable is a paper/evidence/figure/validation lane. Add Rust only if the Studio UI must browse the new paper-evidence CSVs directly.

If that becomes necessary, the next patch should add a read-only artifact category under the existing `/api/runs/{id}/artifacts` surface rather than creating a new simulation path.

---

## Acceptance criteria

### Evidence generation

Command:

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1 --force
```

Must create:

```text
outputs/paper-evidence/major-upgrade-v1/signal_simulation_best_practice_rows.csv
outputs/paper-evidence/major-upgrade-v1/monte_carlo_scenario_trace_rows.csv
outputs/paper-evidence/major-upgrade-v1/detector_processing_best_practice_rows.csv
outputs/paper-evidence/major-upgrade-v1/sensor_fusion_pipeline_rows.csv
outputs/paper-evidence/major-upgrade-v1/ei_iteration_trace_rows.csv
outputs/paper-evidence/major-upgrade-v1/source_code_appendix_inventory.csv
outputs/paper-evidence/major-upgrade-v1/encrypted_ip_redaction_rows.csv
```

### Figure generation

Command:

```bash
rtk python3 paper/generate_figures_major_upgrade_v2.py --strict
```

Must create:

```text
paper/figures/method_progression_ladder.pdf
paper/figures/method_progression_ladder.png
paper/figures/detector_source_lookup.pdf
paper/figures/detector_source_lookup.png
paper/figures/ei_iteration_progress.pdf
paper/figures/ei_iteration_progress.png
```

### Paper validation

Command:

```bash
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

Must fail if:

- New figures are missing.
- New required sections are missing.
- Forbidden overclaim phrases appear.
- Fallback metadata is present in strict figures.

### Paper narrative

The paper must be understandable in this order:

1. Claim boundary.
2. Reader roadmap.
3. Scenario group and Monte Carlo setup.
4. Take-up/climb/cruise definitions.
5. Positive public-proxy card.
6. Bird/RC hard-negative modeling.
7. Noise, clutter, RFI, and receiver impairments.
8. Raw synthetic IQ/cue products.
9. Detector-view schema.
10. Classical radar and ML baselines.
11. Accepted sensor fusion baseline.
12. KPI definition and thresholds.
13. EI process and lock.
14. Results and red flags.
15. Source-code appendix.
16. Limitations and conclusion.

---

## Highest-priority next engineering tasks beyond this diff

These require new benchmark outputs and should not be claimed until generated:

1. **Multi-seed evaluation**  
   Run at least 5–10 seeds and report distribution of LCB95, AP, FP burden, and phase recall.

2. **Passive-RF red-flag stress tests**  
   Add passive-RF shuffle, provenance-only, RF-missing, RF-noisy, and site-shift ablations.

3. **Track-level false-alarm rate**  
   Add track lifecycle generation and report false tracks per hour/km² or a clearly synthetic proxy.

4. **Micro-Doppler sampling audit**  
   Add a short subsection or appendix table explicitly connecting CPI length, Doppler bin width, cadence proxies, and summary features.

5. **OOD targeted holdout**  
   Use targeted holdouts for site/range/weather/family shifts and keep group-lock.

6. **Hardware-in-loop hook**  
   Add a compare-only measured positive-class hook only when actual measured data exists; do not infer named-platform truth from KTH.

7. **Appendix source extraction**
   If the full detector runner is too long, generate exact symbol-range listings with line numbers and keep full files linked by repository path.

---

## Reviewer-facing risk register

| Risk | Severity | Fix in diff | Remaining work |
|---|---:|---|---|
| Synthetic benchmark mistaken for measured truth | High | Repeated claim-boundary text and validation forbidden patterns | Keep boundary in abstract/conclusion |
| FMCW/pulse-Doppler confusion | High | Simulation best-practice rows and TeX explanation | Add chirp slope/ADC details if modeled |
| Micro-Doppler overclaim | High | CPI/cadence sampling-limit text | Add explicit feature extraction table |
| Doppler folding hidden | High | Text calls out speed vs unambiguous velocity | Add branch-level ambiguity table |
| Passive-RF shortcut suspicion | High | Required red-flag discussion | Add shuffle/missingness stress tests |
| Positive holdout too small | High | Group-block LCB95 and diagnostic caveat | Multi-seed larger positive holdout |
| EI perceived as black box | Medium | Source appendix, component weights, lock evidence | Add full candidate leaderboard |
| Appendix code too long | Medium | One-column appendix and inventory | Consider symbol-range extraction |
| Encrypted code mishandled | Medium | Env-key-only encryption; digest fallback | Document key custody outside Git |

---

## Conclusion

The patch turns the paper from “good evidence ledger with strong results” into “radar-first benchmark paper with a clear story.” It does not change the claim; it makes the claim easier to trust. The most important additions are the simulation best-practice rows, method progression figure, sensor-fusion section, EI observability section, and source-code appendix with safe encrypted high-IP handling.
