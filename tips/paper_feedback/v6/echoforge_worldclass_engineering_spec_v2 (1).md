# EchoForge world-class paper repair engineering spec v2

## Objective
Make the paper direct, radar-literate, and reviewer-friendly. The main body should answer four questions quickly:

1. **What is the experiment?** A group-locked synthetic public-proxy benchmark compares accepted human-engineered detector/fusion baselines against a locked EI sparse calibrated fusion candidate.
2. **What are the accepted best practices?** CFAR/MTD/radar branch processing, acoustic/passive-RF cue summaries, and human-designed layered late fusion are the baseline family.
3. **What does EI do differently?** It performs train/CV-only candidate discovery, sparse nonnegative fusion, monotone geodesic-odds calibration, and one blind holdout score after a written selection lock.
4. **What changed on the main KPI?** LCB95 Recall@≤1%FPR improves from 0.083 to 0.699: +0.616 absolute, +742% relative, 8.42x. Point Recall@≤1%FPR improves +185%, AP improves about +545%, and selected-threshold false positives drop from 49 to 1.

The paper must stop making the reader mine a pile of diagnostics. Diagnostics stay available, but the main story must lead with the experiment, comparator, EI mechanism, KPI result, failure modes, and claim boundary.

## Current paper status after review

The latest uploaded PDF has fixed the author block and already reports the headline KPI values and gains in the abstract. It says the paper uses a 4,500-record blind holdout with 24 positive records from eight positive groups and reports EI point Recall@≤1%FPR of 0.833 vs 0.292 and LCB95 Recall@≤1%FPR of 0.699 vs 0.083. That is the right headline, but the document still feels overstuffed and the main body still lets tables and appendix material interrupt the result flow.

The older/current code path still needs hard validation gates so the `NeverHumqn` typo, unlabeled red-line figures, missing stack legends, and low-value tables cannot return.

## Section-by-section content plan

### 1. Abstract
Keep the abstract short and result-forward. Use color only for numeric gains, not every metric. The abstract must contain:

- Primary KPI label: `LCB95 Recall@≤1%FPR`.
- Comparator: `accepted human-engineered prior fusion baseline`.
- EI point result and baseline result.
- LCB95 gain: `+742%`, `8.42x`, `+0.616 absolute`.
- Point Recall@≤1%FPR gain: `+185%`, `2.85x`.
- AP and FP reduction: `+545% AP`, `98% fewer selected-threshold false positives`.
- Trade-off: ROC AUC is lower, so the claim is low-FPR operating gain, not universal rank dominance.

### 2. Claim boundary and contribution
Cut the long contribution bullet list. Replace it with a compact experiment map:

- Corpus: 10,000 scenario groups, 30,000 phase records, 50 positive groups.
- Split: group-locked train/CV and blind holdout.
- Human best-practice comparator: prior layered fusion baseline built from detector branches.
- EI: train/CV-only sparse calibrated late fusion.
- Result: Table/Fig references.
- Boundary: public-proxy synthetic evidence, no measured Iranian-platform signature and no operational claim.

### 3. Core experiment first, modeling second
The current paper introduces many modeling details before the reader understands the experiment. Move a short “Core Experiment” subsection into the first page and postpone detailed radar model cards to appendix. The main body should define only the formulas required to understand the detector products.

### 4. Radar model
Keep the radar equations, but add missing support:

- Cite Swerling/target fluctuation and Ward K-distribution/sea clutter where the current prose mentions these assumptions.
- Explain the mismatch between simplified FMCW range product and pulse/CPI Doppler indexing.
- State that SNR/noise/clutter sweeps are stressors, not measured-site distributions.
- Move exhaustive impairment lists to the appendix.

### 5. Evaluation protocol
Good, but add a short “Why low-FPR recall is primary” note in plain English: in rare-positive UAS surveillance, false alarms are workload/cost, so recall at a false-alarm cap is the operating metric. Keep AP/ROC/calibration as guardrails.

### 6. Main result
Move the main KPI table directly under Fig. 3. The main body should have one table with values and gains. Move KPI-selection rationale to appendix.

### 7. EI explanation
Add a high-level explanation before the components figure:

- **What EI is doing**: selecting complementary score families on train/CV, weighting them sparsely, calibrating scores, locking, then scoring holdout.
- **What EI is not doing**: no holdout tuning, no raw generator-state access, no measured-platform truth.
- Explain why passive-RF-only can have better AP/Recall but worse F1/ECE: full EI is selected for calibrated selected-threshold behavior and lock discipline, not post-hoc universal dominance.

### 8. Diagnostics and appendix
Compact Diagnostics Snapshot should become an appendix block. In the main paper, include only:

- leakage sanity result summary;
- selected-threshold confusion matrix;
- top false alarm families.

Everything else should be appendix/evidence ledger.

### 9. Rich modeling appendix
Create a rich appendix file that documents:

- Iranian fixed-wing pusher-prop public-source family / positive proxy card;
- geometry and kinematics;
- launch/take-up/climb/cruise phase model;
- RCS/aspect envelope;
- pusher-prop micro-Doppler proxy;
- radar branch model;
- clutter/noise/RFI model;
- receiver impairments;
- acoustic cue model;
- passive-RF cue model;
- hard-negative bird/RC families;
- detector-view denylist and allowed features;
- prohibited inferences.

Keep the phrase “Iranian drone” only in the appendix boundary/prohibition text, not as a measured-signature claim.

## Figure-by-figure plan

### Fig. 1: Radar evidence stack
Useful but too governance-heavy. Add one line connecting the stack to the experiment: “detector branches feed human prior fusion and EI fusion.” Reduce box text.

### Fig. 2: Scenario balance / leakage
Current issue: older PDF has a title colliding with the Site panel and footer/x-axis overlap. Latest rendered figure is improved but still too dense. Replace with a simpler two-row figure:

- left: split counts;
- middle: top strata balance counts;
- right: leakage checks and imbalance metrics.

Move detailed site/range/hard-negative count bars to appendix/evidence artifact. This figure’s job is not to show every count; it is to prove group locking and leakage sanity.

### Fig. 3: KPI figure
Keep it as the main result figure. Improvements:

- Title: `Holdout KPI: EI vs accepted prior fusion baseline`.
- Subtitle: `LCB95 +742%; point low-FPR recall +185%; AP +545%; FP 49→1`.
- Red ticks: label as LCB95 ticks directly on left panel.
- Red dashed ROC line: label as 1% FPR operating cap.
- Caption: explicitly distinguish red LCB95 ticks from ROC operating cap.

### Fig. 4: Phase and false-alarm burden
Split into two figures or make the right side full-width. The current combined chart has too many encodings. Recommended:

- Fig. 4A: phase AP and low-FPR recall, one clear legend above.
- Fig. 4B: false-alarm burden, stacked horizontal bars with a readable legend, “near threshold” as a hatched background, and method rows sorted by FP count.

### Fig. 5: EI workflow
Make it a reader story, not a data pipeline poster. Six boxes only:

`Detector scores → Train/CV search → Sparse fusion → Odds calibration → Lock → Blind holdout`

Below it, four short rails:

`Group split`, `Denylist`, `Public-proxy boundary`, `No generated arrays in Git`.

### Fig. 6: Components and controls
Split or de-emphasize. The current plot is more appendix-grade. For main paper, use it only after a paragraph explaining passive-RF-only vs full EI. Add clear legends for component families and AP-vs-recall dot encodings.

### Fig. 7: Radar model card
This is appendix-grade. Keep a compact version in main if space allows, but move richer content to the appendix. Add explicit `public-source proxy`, `synthetic assumption`, `not measured signature` labels.

### Fig. 8: Range-Doppler samples
This should be labeled as qualitative diagnostics only. Improve by using shared normalization, stronger axis labels, and a caption that says “not a detector input and not measured imagery.” If the actual heatmaps are too noisy/random-looking, keep them in appendix.

### Fig. 9: KTH anchor
Good concept. Add a sentence that z-score intervals are normalized to measured-anchor distributions and do not validate the positive proxy.

## Radar-expert gaps to close

1. **Positive proxy is under-specified.** The current appendix has three bullets. Add a table specifying geometry, speed, phase windows, RCS/aspect envelope, prop cadence, launch proxy, and prohibited inferences.
2. **Noise/clutter is listed but not parameterized.** Add `noise_rfi_clutter_model_table.csv` and table rows for AWGN/SNR, Weibull/K clutter, sea/terrain/weather, RFI burst, multipath/ghosting, AGC/clock/quantization/dropout.
3. **RCS model needs boundary language.** State it is an aspect/frequency proxy, not measured monostatic/bistatic RCS.
4. **Passive-RF dominance is a reviewer trap.** The passive-RF-only view has higher AP and fixed-FPR recall than EI but worse F1/ECE. Explain why EI is the locked calibrated selected-threshold artifact and include this as a limitation, not a footnote.
5. **Single seed / eight positive holdout groups is a key weakness.** Keep this visible, not hidden only in limitations.
6. **Track-level FAR is not present.** Say explicitly that the paper reports record-level selected-threshold FP/FPR and fixed-FPR recall, not track-level operational FAR.
7. **Iranian-drone modeling must be rich but bounded.** Show how the public-source family is modeled, but avoid platform-truth language.

## Code changes included in patch

### `paper/echoforge_ieee.tex`

- Adds xcolor KPI macros.
- Fixes `NeverHumqn` and adds validator gates.
- Rewrites abstract for colored values/gains.
- Adds a direct core experiment block.
- Adds a plain-English EI explanation.
- Replaces low-value Table VII with a main KPI gain ledger.
- Moves primary KPI selection rationale and bulky diagnostics toward appendix.
- Inputs a new rich modeling appendix file.
- Updates captions for Fig. 2–Fig. 9.

### `paper/appendix_modeling_details.tex`

New appendix file with rich modeling cards for:

- Iranian fixed-wing pusher-prop public-source proxy;
- radar and branch assumptions;
- clutter/noise/RFI/receiver impairments;
- acoustic/passive-RF cue modeling;
- hard negatives;
- detector-view denylist;
- prohibited inferences.

### `paper/generate_figures_major_upgrade_v2.py`

- Adds `experiment_ladder.pdf` / `.png`.
- Adds `positive_proxy_model_card.pdf` / `.png`.
- Simplifies Fig. 2.
- Strengthens Fig. 3 labels and captions.
- Splits false-alarm burden from phase chart into `false_alarm_burden.pdf` / `.png`.
- Adds actual legends for family colors, near-threshold bars, component colors, and AP-vs-recall encodings.
- Improves box text wrapping.

### `detection/paper_evidence_major_upgrade_v1.py`

- Adds main KPI gain table evidence.
- Adds modeling appendix CSVs for positive proxy, noise/clutter/RFI, and detector-view model cards.
- Extends public proxy positive class card with richer fields.

### `paper/validate_paper.py`

- Requires NeverHuman and bans NeverHumqn.
- Requires abstract gain markers and KPI macros.
- Requires improved figure caption language.
- Requires rich appendix/modeling artifacts and new figures.
- Prevents reintroduction of the generic pass/pass/pass main table.

## Build path

```bash
rtk python3 -m detection.paper_evidence_major_upgrade_v1
rtk python3 paper/generate_figures.py --strict
rtk bash paper/build.sh --copy-tracked
rtk python3 paper/validate_paper.py \
  --tex paper/echoforge_ieee.tex \
  --bib paper/references.bib \
  --pdf paper/echoforge_ieee.pdf \
  --figures-dir paper/figures \
  --paper-evidence-root outputs/paper-evidence/major-upgrade-v1
```

## Acceptance criteria

- Page 1 abstract visibly contains colored +742%, +185%, +545%, and 98% FP reduction values.
- No `NeverHumqn` anywhere in TeX, PDF text, or generated artifacts.
- Main paper reaches the core experiment and KPI result on page 1–2.
- Fig. 2 has no title or x-axis overlap.
- Fig. 3 has two distinct red-line explanations: LCB95 ticks and 1% FPR operating cap.
- Fig. 4/false-alarm chart has readable family and near-threshold legends.
- Fig. 5 boxes do not overflow.
- Table VII is no longer a vague pass/pass/pass table; main table is a KPI gain ledger or the table is moved to appendix.
- Appendix documents positive proxy, radar model, noise/clutter/RFI, receiver impairments, hard negatives, and prohibited inferences in inspectable tables.
