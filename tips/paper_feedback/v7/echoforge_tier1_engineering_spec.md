# EchoForge Tier-1 Radar Paper Engineering Spec

## Executive judgment

The current manuscript is much stronger than the earlier draft: the title page now says **NeverHuman**, the abstract states the KPI gains directly, Fig. 3 and Fig. 4 are substantially clearer, and the 15-page render includes a richer appendix map and public-proxy/noise tables. The remaining tier-1 gaps are not cosmetic. A senior radar reviewer will still ask: *what exactly flows from the simulator into the model? which fields are denied? how does raw complex-IQ become detector-view evidence? what assumptions create the Iranian-style fixed-wing public proxy? how are clutter, RFI, multipath, and receiver impairments represented?* The patch adds direct answers to those questions in code, generated figures, evidence CSV/JSON artifacts, and validation gates.

## Current strengths

1. **Clear primary result**: the abstract foregrounds LCB95 Recall@≤1%FPR and correctly states the gain over prior fusion: 0.083 to 0.699, +61.6 pp, +742%, 8.42x; point Recall@≤1%FPR rises 0.292 to 0.833.
2. **Honest guardrail**: the paper explicitly concedes that EI ROC AUC is lower than prior fusion, which prevents overclaiming.
3. **Claim boundary**: the positive class is a fixed-wing pusher-prop public proxy, not a measured platform or sensor-surrogate claim.
4. **Strong leakage posture**: group-locked holdout, denylist framing, label-shuffle, metadata-only, stratum-only, and nearest-neighbor checks are all the right kinds of audit evidence.
5. **Better figures**: Fig. 3 and Fig. 4 now tell the KPI and false-alarm story much more clearly than the early draft.
6. **Appendix has the right direction**: the modeling map and proxy/noise tables show that the paper is no longer just a leaderboard with caveats.

## Remaining tier-1 risks

### 1. Data processing is still under-specified
A radar/ML reviewer will want a concrete chain: scenario generation → raw IQ references → range-Doppler products → detector views → denylist gate → train/CV selection → blind holdout scoring → paper evidence. Right now the manuscript says this in prose, but it does not yet have a visual/data contract that makes the data path auditable.

**Fix in patch:** Add `data_processing_flow.pdf/png`, `data_processing_contract_rows.csv/json`, `radar_processing_chain_rows.csv/json`, and a `Data-Processing Contract` subsection.

### 2. Radar simulation details need to be evidence-backed, not only described
The paper gives equations and a model card, but the appendix should explicitly tell the reader what is generated, why it matters, and what it cannot prove.

**Fix in patch:** Add `public_proxy_model_detail_rows.csv/json`, `environment_impairment_model_rows.csv/json`, and explicit appendix tables for the positive proxy, noise/clutter/RFI/receiver details, and detector-view modeling contract.

### 3. Fig. 3 should compute displayed gains from evidence, not hard-coded text
The current figure generator includes headline gain text. A tier-1 reproducibility stance requires that the displayed percentages derive from the evidence summary, not manual constants.

**Fix in patch:** Add `_gain_payload(context)` and use it inside `figure_kpi_ranking()`.

### 4. Fig. 4 still encodes too much at once if rendered small
The full AP + recall bar set is dense. A reviewer cares most about fixed-FPR recall by phase; AP can be diagnostic annotation.

**Fix in patch:** Simplify the left panel to fixed-FPR recall bars only, with AP printed as small diagnostic text.

### 5. Validation should block regression
The current validators already check many figure and paper properties, but they should also require the data-processing figure, proxy/noise model card, and evidence CSVs.

**Fix in patch:** Extend `validate_paper.py` and `validate_visuals.py` with new figure/evidence requirements and required visible terms.

## File-level changes

### `paper/echoforge_ieee.tex`

- Adds reusable macros:
  - `\mainkpi`
  - `\pointkpi`
  - `\proxypositive`
  - `\fusionbaseline`
- Rewrites abstract to begin with the main research question and colored gain values.
- Adds a `Data-Processing Contract` subsection and table.
- Adds `data_processing_flow.pdf` as a new figure.
- Adds `proxy_noise_model_card.pdf` to the rich modeling appendix.
- Adds appendix tables for positive proxy modeling, environment/receiver modeling, and detector-view contracts.

### `paper/generate_figures_major_upgrade_v2.py`

- Adds two new vector-backed figures:
  - `data_processing_flow.png/.pdf`
  - `proxy_noise_model_card.png/.pdf`
- Adds `_gain_payload(context)` so KPI-gain text in figures is computed from evidence rows.
- Improves Fig. 2 explanatory subtitle to emphasize denylist/data-contract reading.
- Simplifies Fig. 4 phase panel to fixed-FPR recall bars with AP as diagnostic text.
- Includes the new figures in `generate_all()`.

### `detection/paper_evidence_major_upgrade_v1.py`

- Adds a tier-1 evidence schema version.
- Extends the radar model card with aspect lobes, fluctuation model, and RCS basis.
- Adds new evidence emitters:
  - `_main_kpi_gain_rows()`
  - `_positive_proxy_model_detail_rows()`
  - `_environment_impairment_model_rows()`
  - `_data_processing_contract_rows()`
  - `_radar_processing_chain_rows()`
- Writes new evidence artifacts:
  - `main_kpi_gain_table.csv/json`
  - `public_proxy_model_detail_rows.csv/json`
  - `environment_impairment_model_rows.csv/json`
  - `data_processing_contract_rows.csv/json`
  - `radar_processing_chain_rows.csv/json`

### `paper/validate_paper.py`

- Requires the two new figures and previews.
- Requires new paper phrases for the data-processing and rich modeling appendix.
- Requires new evidence rows and CSVs in the manifest/output directory.

### `paper/validate_visuals.py`

- Requires the new PNG/PDF outputs.
- Requires visible PDF terms in the new figures:
  - `generator state never becomes model input`
  - `Denylist gate`
  - `Blind holdout`
  - `Fixed-wing pusher-prop`
  - `Environment and receiver`
  - `no measured Iranian-drone radar truth`

## Recommended paper structure after patch

Main body should be ruthlessly focused:

1. Claim boundary + one experiment.
2. Radar/model assumptions, compact enough for credibility.
3. Scenario split, detector views, and data-processing contract.
4. Primary KPI result, Fig. 3, and main gain ledger.
5. EI high-level mechanism.
6. Raw/anchor diagnostics only as support.
7. Limitations.
8. Rich appendix with public-proxy/noise/clutter/RFI/data-processing contracts.

## Expert red flags and how the patch addresses them

| Expert concern | Current risk | Patch response |
|---|---|---|
| “Is this a measured Iranian drone claim?” | The appendix mentions Iranian-style sources, which can be misunderstood. | Adds explicit positive-proxy card with boundaries and figure text saying no measured Iranian-drone radar truth. |
| “Do model features leak split or group identity?” | Current text says no, but the processing flow is not visual. | Adds data-processing flow and evidence rows showing denylist gate. |
| “How exactly do radar products become scores?” | Equations exist, but pipeline is not concrete enough. | Adds radar-processing chain rows. |
| “Why should I trust the +742% gain?” | Numbers appear in prose and figures; reviewers may worry about manual drift. | Adds `main_kpi_gain_table` and `_gain_payload(context)`. |
| “Fig. 4 is too much.” | AP and recall bars plus stacked false alarms can be visually overloaded. | Simplifies left panel to operating metric; AP becomes diagnostic text. |
| “Appendix is just prose.” | Prose is less auditable than artifacts. | Adds CSV/JSON-backed modeling rows. |

## Commands to run

```bash
git checkout -b paper/tier1-radar-review-pass
git apply /path/to/echoforge_tier1_revision.patch

python detection/paper_evidence_major_upgrade_v1.py --force
python paper/generate_figures_major_upgrade_v2.py --strict
python paper/validate_visuals.py
./paper/build.sh
python paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Tier-1 acceptance criteria

- The abstract has colored KPI gains and the ROC AUC caveat.
- Fig. 2 has no title/axis overlap and explicitly supports the leakage/data-contract story.
- Fig. 3 displays the 1% FPR operating cap and KPI gains from evidence, not hand-written constants.
- Fig. 4 can be interpreted without reading the caption first.
- Fig. 5 makes EI understandable to a first-time reader.
- Appendix includes auditable model cards for positive proxy, radar/noise/RFI/receiver assumptions, detector views, and data-processing flow.
- Validators fail on stale spelling, stale percentage values, missing figures, missing appendix artifacts, or missing data-processing evidence.

## Things I would not claim

- Do not claim measured Iranian-drone radar signatures.
- Do not claim fielded sensor parity or operational performance.
- Do not claim that EI dominates all metrics: ROC AUC and passive-RF-only AP/recall controls complicate that.
- Do not claim phase or family results are stable field rates; the holdout has only eight positive groups.

