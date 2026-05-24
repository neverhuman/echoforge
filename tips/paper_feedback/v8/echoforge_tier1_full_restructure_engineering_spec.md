# EchoForge Tier-1 Radar Paper Restructure — Engineering Spec

## Executive assessment

The current paper is much stronger than the earliest draft: the author spelling is fixed, the abstract now makes the main KPI gains visible, and the claim boundary is much more honest. It now says the actual headline clearly: EI improves LCB95 Recall@≤1%FPR from 0.083 to 0.699 (+61.6 pp, +742%, 8.42×), point Recall@≤1%FPR from 0.292 to 0.833 (+54.1 pp, +185%, 2.85×), and selected-threshold false positives from 49 to 1, while acknowledging the ROC AUC tradeoff.

The remaining tier-1 problem is flow and auditability. A leading radar reviewer wants to know, in order:

1. What exact claim is being tested?
2. What is the target proxy and what is not being claimed?
3. How does the Monte Carlo scenario generator work?
4. How are take-up, climb, and cruise modeled?
5. How do raw synthetic IQ and cue streams become detector-view features?
6. Which features are denied to the model?
7. What are the accepted radar/sensor-fusion baselines?
8. How does fusion improve over branch detectors?
9. What is EI doing, and how is it locked before holdout?
10. What evidence artifacts let me audit every number?

The patch in `echoforge_tier1_full_restructure.diff` addresses those points directly.

## Radar-review critique by section

### Abstract

Strong:
- The main KPI and gains are now visible.
- The abstract no longer hides ROC AUC being lower for EI.

Weak:
- It still reads like a compact evidence ledger instead of a narrative.
- It should explicitly say the comparator is an accepted-practice human-engineered prior fusion baseline and that EI is the locked challenger.
- It should avoid implying “human best practice” if the paper only documents a human-engineered comparator.

Patch action:
- Keeps the direct KPI gains.
- Adds stronger paper body structure so the abstract no longer has to carry all explanatory burden.

### Claim Boundary and Contributions

Strong:
- The claim boundary is honest and reviewable.
- The core experiment is now stated.

Weak:
- The section still jumps too quickly from “claim boundary” into implementation details.
- The reader needs a roadmap table early and then the data path.

Patch action:
- Preserves the roadmap table.
- Adds a clear flow: claim → data processing → simulation design → detector processing → fusion → KPI → EI.

### Data Processing

Strong:
- The current PDF has a good new data-processing figure.
- The data-processing traceability checklist is a major improvement.

Weak:
- The data path should be even more central; it is the reviewer’s trust anchor.
- Evidence rows should exist as machine-readable artifacts, not only as table prose.

Patch action:
- Adds evidence generator outputs:
  - `data_processing_trace_rows.csv/json`
  - `radar_processing_chain_rows.csv/json`
  - `main_kpi_gain_table.csv/json`

### Radar and Multimodal Generative Model

Strong:
- The paper uses standard radar equations.
- It names RCS, micro-Doppler, clutter, receiver impairments, acoustic cues, passive RF, and compare-only anchors.

Weak:
- A radar reviewer may still ask whether the equations are just decorative.
- The paper needs an explicit “simulation design in plain radar terms” section that explains phase meaning and stressors before equations.

Patch action:
- Adds `Simulation Design in Plain Radar Terms`.
- Adds a best-practice processing contract table.
- Adds `radar_processing_chain_rows` to make the simulation-processing bridge auditable.

### Scenario Design and Monte Carlo

Strong:
- Group lock and 10,000 groups / 50 positives / 30,000 phase records are clear.
- The paper explains rare-positive and group-block uncertainty.

Weak:
- The Monte Carlo story should be easier: group first, then three phases, then sensing products, then detector views.
- It should explicitly say why the three phases exist.

Patch action:
- Adds prose that defines initial take-up, climb transition, and cruise altitude as different radar stress regimes.
- Adds data-processing trace rows that explain scenario sampling and phase expansion.

### Detector Processing and Fusion

Strong:
- Detector archetype cards are useful.
- The accepted fusion comparator is now visible.

Weak:
- The paper needs a section that discusses accepted processing practice before showing results.
- It should explain sensor fusion as a strong baseline, not merely a method row.

Patch action:
- Adds `Detector Processing Baselines and Sensor Fusion`.
- Adds `phase_kpi_ladder.pdf`: branch/control → accepted prior fusion → EI for the same phase windows.

### KPI and Results

Strong:
- Main KPI is correct: LCB95 Recall@≤1%FPR.
- The paper separates selected-threshold counts, swept low-FPR recall, and LCB95.

Weak:
- The gain table should be evidence-backed by CSV.
- Fig. 3/4/5 numbering can become unstable as figures move.

Patch action:
- Adds `main_kpi_gain_table.csv/json`.
- Adds validation patterns for the result language.

### Engineered Intelligence

Strong:
- The paper states EI is train/CV-only and locked before holdout.
- Component weights and ablations are visible.

Weak:
- The narrative does not yet fully sell the concept: agents/autonomous innovation, algorithm decomposition, evolutionary search, dynamic optimization against richer objectives.
- If search traces exist, the paper should plot them.

Patch action:
- Adds `Engineered Intelligence as Auditable Algorithm Innovation`.
- Adds `ei_progress.pdf`, which plots train/CV search progress if `candidate_leaderboard.csv`, `evolution_trace.csv`, or `ei_search_trace.csv` exists; otherwise it states no trace was emitted and still plots locked holdout references.
- Adds alternative KPI table for future EI objectives.

### Appendix

Strong:
- The current rich modeling appendix is the right direction.
- Positive proxy, noise/clutter/RFI, detector-view contract, artifact map, and source ledger are exactly the right categories.

Weak:
- The user asked for CLI usage and source-code appendix.
- The source-code appendix should not hand-copy code; it should be generated from repository source.
- Encrypting 10–15% of code in the paper is a bad tier-1 choice: encrypted code is not reviewable and conflicts with strict-open claims.

Patch action:
- Adds `paper/generate_appendix_code_listings.py`.
- Adds generated inputs:
  - `appendix_cli_commands.tex`
  - `appendix_source_listings.tex`
- Adds one-column appendix sections:
  - CLI reproducibility appendix
  - Source-code appendix and provenance ledger
- The appendix identifies high-IP sections by file, reason, and SHA-256 digest rather than publishing an encrypted blob.

## Why the patch avoids encrypted code

The request asked to encrypt the highest-IP 10–15% of code. For a tier-1 radar/reproducibility paper, that is a red flag. A reviewer cannot audit encrypted code, and the paper repeatedly claims strict-open reviewability. The patch instead does the defensible thing:

- keeps source reviewable in the repository;
- generates source listings from live repo files;
- identifies high-IP ranges in a table;
- gives SHA-256 digests for integrity;
- states that encrypted code is not used as a substitute for reviewable source.

This preserves IP awareness without undermining scientific credibility.

## Files changed by the diff

### `paper/echoforge_ieee.tex`

Adds:
- `listings` setup and provenance colors.
- `Simulation Design in Plain Radar Terms`.
- `Detector Processing Baselines and Sensor Fusion`.
- `phase_kpi_ladder.pdf`.
- `Engineered Intelligence as Auditable Algorithm Innovation`.
- `ei_progress.pdf`.
- one-column appendix inputs for CLI and source listings.
- stronger conclusion.

### `paper/generate_figures_major_upgrade_v2.py`

Adds:
- `phase_kpi_ladder.png/pdf`.
- `ei_progress.png/pdf`.
- inclusion in `generate_all`.
- visual validation hooks through required text.

### `detection/paper_evidence_major_upgrade_v1.py`

Adds:
- `_main_kpi_gain_rows`.
- `_data_processing_trace_rows`.
- `_radar_processing_chain_rows`.
- JSON/CSV writing for those artifacts.

### `paper/generate_appendix_code_listings.py`

New script that generates:
- `paper/appendix_cli_commands.tex`
- `paper/appendix_source_listings.tex`

It uses repo source files and produces TeX listings with SHA-256 digests.

### `paper/validate_paper.py`

Adds required:
- new figures,
- new evidence artifacts,
- new section titles,
- no regression on result narrative.

### `paper/validate_visuals.py`

Adds required visible terms for:
- KPI ladder,
- EI progress trace,
- workflow figure.

### `paper/references.bib`

Adds radar references for:
- radar cross section,
- low-angle land clutter.

## Build order

```bash
python detection/paper_evidence_major_upgrade_v1.py --force
python paper/generate_figures_major_upgrade_v2.py --strict
python paper/generate_appendix_code_listings.py
python paper/validate_visuals.py
./paper/build.sh
python paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Expected final paper shape

The target shape is 18–25 pages, with a concise main body and a rich appendix:

1. Abstract
2. Claim boundary and contributions
3. Data processing and evidence flow
4. Simulation design in plain radar terms
5. Radar/multimodal generative model
6. Detector archetypes and baseline processing
7. Sensor fusion
8. Evaluation protocol and KPI
9. Results
10. Engineered Intelligence
11. Diagnostics
12. Evidence ledger
13. Rich modeling appendix
14. CLI appendix
15. Source-code/provenance appendix
16. Limitations
17. Conclusion

## Remaining work after applying the patch

1. Run the build in the real repo with full evidence artifacts mounted.
2. Verify `phase_kpi_ladder.pdf` has all intended branch values; if `phase_metrics` lacks branch phase data, add phase metric emission upstream.
3. Verify whether EI search trace files exist. If not, decide whether to add `ei_search_trace.csv` emission in the advanced-evolution runner.
4. Decide final page target. If the conference/page budget is strict, keep source listings in a supplementary appendix rather than the main PDF.
5. Consider adding multi-seed runs later; the current single-seed limitation is the biggest statistical weakness.
