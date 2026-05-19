# Passive RF, EO/IR, and Passive Radar Backlog current

Generated: `2026-05-18T20:16:38Z`

This report defines backlog packets for the passive and confirmatory sensing
layers behind EchoForge Detector Realism current. These packets are implementation
planning artifacts for public-proxy data products, false-alarm controls, and
claim boundaries. They do not replace the active radar, acoustic cueing, or C2
fusion work.

## Claim Boundary

Strict-open public-proxy backlog only. Packets describe modeling work, data products, and validation gates for passive/supporting sensor layers. They are not measured Shahed/Geran performance, receiver sensitivity, classified fidelity, deployment geometry, or proprietary-equivalent sensor behavior.

## Source Basis

- `detection/RADAR_REALISM.md`
- `detection/reports/detector_osint_family_matrix.md`
- `tips/detectors/tip1.txt`
- `tips/detectors/tip2.txt`
- `tips/detectors/tip3.txt`
- `tips/detectors/tip4.txt`
- `tips/detectors/tip5.txt`
- `tips/detectors/tip6.txt`
- `tips/detectors/tip7.txt`
- `tips/detectors/tip8.txt`

## Critical Constraints

- Shahed-class one-way UAVs may be RF-silent; passive RF must support explicit missing-signal behavior.
- EO/IR needs a separate visual asset/model policy before imagery or thermal crops become detector-facing products.
- Passive radar depends on illuminator geometry and must not claim own-transmitter radar performance.

## Backlog Packets

### `passive_rf_esm_optional_evidence`: Passive RF / ESM Optional Evidence

| Field | Value |
|---|---|
| Priority | `P2` |
| Modality | Passive RF, ESM, direction finding, COMINT-style emitter observation |
| Role | Detect, classify, or direction-find UAV-related emissions when they are present, then pass optional evidence into fusion. |

#### Rationale

- The current OSINT family matrix ranks passive RF/ESM as a supporting layer, not a primary detector.
- RF evidence can be strong for controller, telemetry, video, or payload emitters, but it is absent when the target does not emit useful signals.
- Shahed-class one-way UAVs may fly pre-programmed routes and may be RF-silent from a control-link perspective.
- The correct simulator behavior is explicit missing-signal evidence, not an automatic RF miss or a hidden positive-class shortcut.

#### Implementation Blockers

- No current emitter-state schema exists for optional, intermittent, or absent emissions.
- Emitter libraries need strict-open labels and broad band/protocol categories rather than vendor-specific signatures.
- Direction-of-arrival uncertainty and channel occupancy models are not yet attached to current sensor geometry.
- Fusion has not yet defined how absent RF evidence should affect posterior confidence.

#### Expected Data Products

- rf_esm_detections.csv with time, sensor_id, bearing_deg, bearing_sigma_deg, frequency_band, emitter_class_proxy, signal_strength_proxy, and emission_present flag
- rf_esm_tracks.csv with emitter-track hypotheses, source provenance, confidence, and stale-track state
- rf_esm_quality.json with missing-signal rate, crowded-band stress cases, and false-alarm source counts
- rf_esm_schema.json documenting detector-facing columns and audit-only restricted fields

#### False-Alarm Controls

- Require emitter-class and frequency-band consistency across dwell windows before track promotion.
- Track direction-of-arrival intersections with uncertainty rather than treating a single bearing as a location.
- Model crowded RF environments with friendly emitters, cell/Wi-Fi/background signals, RFI bursts, and stale library matches.
- Keep passive RF as optional evidence that must be correlated with radar, acoustic, EO/IR, or C2 tracks for classification confidence.

#### Claim Boundaries

- Do not claim passive RF can detect RF-silent or pre-programmed one-way UAVs.
- Do not publish exact receiver sensitivity, demodulation capability, exploit logic, or emitter fingerprints.
- Do not convert a missing RF signal into proof that no target exists.
- Use broad public-proxy band and emitter categories only.

#### Dependencies

- fusion_c2_layered_architecture track provenance and optional-evidence semantics
- acoustic-cueing-network-products for passive cue sidecar patterns
- generator-truth-denylist so emitter truth does not leak into detector features
- hard-negative/RFI confuser object packs and false-alarm accounting

#### Acceptance Evidence

- RF-silent positive cases produce no RF detection but remain valid positives for other sensors.
- Crowded-band negative cases create false-cue candidates that are counted and controlled.
- Detector-facing columns exclude target family, true route, and ground-truth emitter state.
- Report states passive RF is supporting evidence only.

### `eo_ir_confirmation_asset_policy`: EO/IR Confirmation and Visual Asset Policy

| Field | Value |
|---|---|
| Priority | `P1` |
| Modality | Visible EO, thermal IR, PTZ slew-to-cue, image classifier support |
| Role | Confirm and refine tracks after radar, acoustic, passive RF, or C2 cueing; do not act as the wide-area primary detector in current. |

#### Rationale

- The OSINT family matrix ranks EO/IR as a confirmatory layer with weather, visibility, and cue-latency limits.
- EO/IR products are valuable for operator confirmation and class-confidence updates after a cue exists.
- A visual detector needs a separate asset/model policy because image crops, silhouettes, thermal textures, labels, and augmentation can leak target identity.
- EO/IR must represent denied or degraded visual confirmation under clouds, fog, smoke, glare, night conditions, target aspect, and gimbal saturation.

#### Implementation Blockers

- No current visual asset provenance policy exists for silhouettes, thermal appearances, or generated imagery.
- No EO/IR detector-facing schema separates image/crop products from restricted generator truth.
- Weather, background, contrast, atmospheric attenuation, and slew-to-cue latency are not yet first-class current EO/IR fields.
- Operator confirmation and automated image-classifier confidence need separate states.

#### Expected Data Products

- eoir_cue_requests.csv with source_track_id, cue_source, request_time_s, slew_latency_s, and gimbal_available flag
- eoir_observations.csv with line_of_sight angles, crop_id, visual_band, contrast_proxy, weather_visibility_gate, and confirmation_state
- eoir_asset_manifest.json with asset provenance, license boundary, generation method, class-label policy, and thermal/visual representation notes
- eoir_quality.json with confirmation latency, denied-confirmation cases, false-confirmation counts, and no-raw-sensitive-imagery status

#### False-Alarm Controls

- Gate visual confidence by cue quality, line of sight, contrast, background class, and weather/visibility state.
- Represent birds, balloons, aircraft lights, debris, terrain features, and hot machinery as visual/thermal hard negatives.
- Separate operator-confirmed, model-suggested, and unconfirmed states.
- Require source-track provenance so EO/IR confirmation does not silently invent a new primary track.

#### Claim Boundaries

- Do not claim EO/IR is all-weather or wide-area primary detection for current.
- Do not claim exact identification range for a Shahed/Geran-class target.
- Do not use generated or public imagery as proprietary-equivalent object signatures.
- Do not ship raw sensitive imagery or visual assets without an explicit strict-open asset policy.

#### Dependencies

- visual asset/model policy for strict-open EO/IR crops, labels, and thermal proxies
- fusion_c2_layered_architecture for cue requests and confirmation state
- RADAR_REALISM phase model for latency and phase-specific confirmation metrics
- generator-truth-denylist for visual label and object-family leakage controls

#### Acceptance Evidence

- EO/IR rows are linked to cue requests or external tracks by provenance.
- Denied-confirmation cases are explicit and counted rather than dropped.
- Asset manifest states license, source, and generated-image boundaries.
- Report states EO/IR is confirmatory and policy-gated.

### `passive_radar_illuminator_geometry`: Passive Radar Illuminator-of-Opportunity Geometry

| Field | Value |
|---|---|
| Priority | `P2` |
| Modality | Passive bistatic/multistatic radar using FM, DAB, DVB-T, cellular, or other public-proxy illuminators |
| Role | Provide silent cueing or gap-filler tracks when illuminator geometry, signal quality, and receiver placement support useful passive returns. |

#### Rationale

- The OSINT family matrix ranks passive radar after active radar and acoustic layers because performance is geometry- and illuminator-dependent.
- Passive radar does not emit its own waveform; it depends on transmitters of opportunity and bistatic/multistatic geometry.
- The artifact must not present passive radar products as own-transmitter radar performance.
- Urban multipath can be both a source of observability and a source of false tracks, so illuminator provenance is a first-class product.

#### Implementation Blockers

- No current illuminator catalog schema exists for transmitter class, coverage, signal quality, and lawful public-proxy assumptions.
- Bistatic range, bistatic Doppler, baseline geometry, and receiver synchronization models are not yet in detector products.
- Multipath, shadowing, and illuminator outage states need hard-negative and quality-gate coverage.
- Fusion has not defined how passive-radar cue confidence compares with active radar tracks.

#### Expected Data Products

- passive_radar_illuminators.json with illuminator_id, class, band_proxy, availability, geometry notes, and claim boundary
- passive_radar_detections.csv with receiver_id, illuminator_id, bistatic_range_proxy, bistatic_doppler_proxy, bearing_proxy, snr_proxy, and quality gates
- passive_radar_tracks.csv with multistatic association, illuminator provenance, covariance/uncertainty, confidence, and stale-track state
- passive_radar_quality.json with geometry-eligible fraction, illuminator outage cases, multipath false tracks, and active-radar correlation metrics

#### False-Alarm Controls

- Require illuminator quality gating before detection promotion.
- Use bistatic/multistatic consistency checks across illuminators or receivers.
- Model urban multipath ghosts, traffic, towers, terrain masking, and broadcast outages as false-cue sources.
- Correlate passive-radar cues with active radar, acoustic, EO/IR, or C2 tracks before high-confidence classification.

#### Claim Boundaries

- Do not claim passive radar has own-transmitter radar range, update rate, or waveform control.
- Do not claim universal detection envelopes independent of illuminator geometry.
- Do not publish real transmitter exploitation details or deployment geometry.
- Keep products as public-proxy cue/track artifacts, not measured passive-radar validation.

#### Dependencies

- illuminator catalog and geometry eligibility model
- fusion_c2_layered_architecture for passive cue provenance and correlation
- RADAR_REALISM phase metrics for passive cue latency and missed/false-track accounting
- hard-negative multipath and RFI stress cases

#### Acceptance Evidence

- Passive radar reports geometry-ineligible scenes explicitly.
- Every detection references illuminator provenance.
- False tracks from multipath and outage states are counted in quality evidence.
- Report states passive radar is illuminator-dependent and not own-transmitter radar performance.

## Integration Notes

- Keep passive RF, EO/IR, and passive radar as provenance-rich sidecars until
  fusion semantics are defined.
- Publish generated JSON/Markdown copies only under `outputs/`; this committed
  report is the reviewable planning artifact.
- Use the current three-tier phase IDs when later packets add metrics:
  `initial_take_up`, `climb_transition`, and `cruise_altitude`.
- Treat hard negatives as robustness work: RFI, crowded emitters, birds,
  balloons, lights, weather, towers, multipath, and illuminator outages should
  increase false-alarm accounting rather than encode evasion logic.
