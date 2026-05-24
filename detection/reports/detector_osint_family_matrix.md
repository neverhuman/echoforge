# Detector OSINT Family Matrix current

Generated: `2026-05-18T19:59:01Z`

This report ranks public-source detector families for EchoForge Detector Realism
current. It treats detector realism as a layered sensing and fusion problem: wide-area
radars provide warning, high-resolution C-UAS radars form and classify difficult
low-altitude tracks, acoustic/RF/EO layers add supporting evidence, and C2
fusion controls false tracks and handoff.

## Claim Boundary

Strict-open public-proxy summary only. Values are vendor/government/media range or capacity proxies, not guaranteed Shahed/Geran detection ranges, receiver sensitivity, Pd/Pfa curves, classified modes, deployment geometry, or proprietary sensor behavior.

## Source Basis

- `tips/detectors/tip1.txt`
- `tips/detectors/tip2.txt`
- `tips/detectors/tip3.txt`
- `tips/detectors/tip4.txt`
- `tips/detectors/tip5.txt`
- `tips/detectors/tip6.txt`
- `tips/detectors/tip7.txt`
- `tips/detectors/tip8.txt`

## Priority Meaning

| Priority | Meaning |
|---|---|
| P0 | Implement first for Detector Realism current benchmarks. |
| P1 | Implement in current if scope allows; needed for realistic layered operation. |
| P2 | Supporting or specialized layer; model after primary radar/fusion surfaces. |
| P3 | Context layer only for current unless explicitly needed. |

## Ranked Family Matrix

| Rank | Family | Role | Band / modality | Public range proxy | Update | Tracks | False-alarm controls | Products | EchoForge priority |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | `fusion_c2_layered_architecture`<br>Layered C2 and sensor-fusion architectures | Correlate heterogeneous radar, acoustic, EO/IR, RF, and external tracks for alerting, classification, and effector handoff. | multi-sensor<br>C2<br>track fusion | No single range; inherits sensor envelopes. Public examples combine Ku/L-band radars, EO/IR, passive RF, acoustic, and external feeds. | System-level real-time correlation; public details usually describe track correlation/cueing rather than fixed Hz. | Architecture dependent; relevant proxy is multi-sensor correlation and weapon-target pairing under saturation. | cross-sensor confirmation<br>track correlation and deconfliction<br>IFF/external feed correlation where available<br>false-track mitigation and operator confirmation | fused track id<br>source provenance<br>classification confidence<br>cue messages<br>engagement handoff state | **P0**<br>Make fusion the top-level benchmark artifact: track provenance, source confidence, stale-track handling, and public-proxy false-track accounting. |
| 2 | `high_resolution_cuas_x_ku`<br>High-resolution X/Ku-band C-UAS radars | Terminal-area detection, track formation, drone/bird discrimination, and fire-control-quality cueing for low/slow/small targets. | X-band<br>Ku-band<br>pulse-Doppler<br>FMCW<br>AESA/e-scan | Public proxies include nano/small UAV kilometer-scale examples, Blighter A400 RCS-class ranges, SPEXER small-UAV classification claims, and KuRFS small-object discrimination claims. | Fast revisit or persistent sector/hemisphere coverage; some public systems advertise sub-second to 10 Hz-class track products. | Representative public claims include hundreds of tracks or more than 300 tracks per sector for some systems. | Doppler filtering<br>micro-Doppler or spectral classification<br>CFAR/adaptive thresholds<br>biological/non-biological discrimination<br>subclutter visibility claims | range<br>azimuth<br>elevation<br>radial velocity<br>RCS/amplitude proxy<br>micro-Doppler features<br>track quality | **P0**<br>Model as the primary low-altitude discriminator: range-Doppler cube features, micro-Doppler confidence, clutter class, revisit cadence, and false-alarm budget. |
| 3 | `tactical_multimission_s_band_aesa`<br>Tactical S-band multi-mission AESA radars | Mobile C-UAS, VSHORAD, C-RAM, and defended-site hemispheric surveillance. | S-band<br>4D radar<br>AESA<br>GaN<br>pulse-Doppler | RADA MHR public examples include nano UAV 5 km and medium UAV 25 km; Green Rock publishes ultralight-UAV range classes by configuration. | Green Rock public refresh proxy: 1 Hz search with priority target updates in the 4-10 Hz class; MHR uses track-while-scan/revisit modes. | MHR-family public material describes hundreds of tracks; configuration dependent. | advanced clutter and multipath mitigation<br>Doppler processing<br>ECCM/IFF support where integrated<br>multi-panel 360 degree correlation | 4D tracks<br>target class proxy<br>track-while-scan state<br>C-RAM/C-UAS mission tags<br>sensor health | **P0**<br>Use as the reference medium tactical radar family with target-class range bins, configurable update priority, and clutter/multipath stress cases. |
| 4 | `medium_range_gbad_3d_4d`<br>Medium-range 3D/4D GBAD and SHORAD surveillance radars | Wide defended-area air picture, low-altitude gap coverage, and cueing for NASAMS/IRIS-T/SHORAD-type networks. | L/S/C/X/G-band public labels<br>3D/4D<br>rotating or AESA | Public examples include Sentinel 75-120 km instrumented range, TRML-4D 250 km instrumented range with fighter and missile proxies, GM200 250-400 km class surveillance proxies, and Giraffe 4A 400 km instrumented range. | Public proxies include 1 s 360 degree revisit for Giraffe-family radars and 1.5-3 s mode-dependent GM200 updates. | TRML-4D public proxy: more than 1,500 tracks; Sentinel F1 public proxy: more than 60 tracks. | track-while-scan<br>low-altitude target processing<br>Mode 5/IFF where integrated<br>sector prioritization<br>C2 correlation | regional 3D/4D track<br>air-defense cue<br>classification proxy<br>track confidence<br>handoff to fire-control layer | **P1**<br>Represent as cueing and context sensors: radar-horizon masks, revisit delay, lower-resolution classification, and high-capacity saturation queues. |
| 5 | `l_band_counterfire_gap_filler`<br>L-band counterfire and gap-filler multi-mission radars | Compact 360 degree volume surveillance for C-RAM, C-UAS, counterfire, and low-altitude warning. | L-band<br>electronically steered<br>coherent pulse-Doppler | AN/TPQ-50 public proxy: more than 35 km air-surveillance instrumented range and up to 15 km counterfire/RAM range. | Continuous 360 degree surveillance is the public proxy; exact dwell/update behavior is not publicly specified. | Multiple simultaneous incoming weapons and air targets; exact public capacity varies by source and mode. | coherent pulse-Doppler processing<br>automatic discrimination/geolocation<br>mission-mode filtering<br>C2 correlation in LIDS-like stacks | 3D target location<br>air-surveillance track<br>point-of-origin/point-of-impact for RAM<br>mission mode | **P1**<br>Add as a broad 360 degree gap-filler with coarser classification than X/Ku C-UAS but strong cueing value. |
| 6 | `distributed_acoustic_networks`<br>Distributed acoustic detection networks | Low-cost passive cueing for piston/propeller one-way attack UAVs in radar-horizon or clutter gaps. | acoustic<br>microphone arrays<br>edge ML<br>bearing/time correlation | No universal range. Public Ukraine examples emphasize thousands of low-cost sensors and regional cueing, not guaranteed single-node range. | Event-driven detections with bearing/time records; fusion cadence depends on network and communications. | Network-level capacity, not radar track count; saturation behavior depends on sensor density and fusion backend. | engine/propeller acoustic classifiers<br>bearing/time triangulation<br>human or C2 confirmation<br>radar/visual correlation<br>local-noise rejection | bearing estimate<br>time of arrival<br>acoustic class confidence<br>triangulated cue<br>sensor node provenance | **P1**<br>Implement as passive cue generation with false positives from traffic, weather, industrial noise, and friendly aircraft. |
| 7 | `eo_ir_confirmation_layer`<br>EO/IR confirmation and precision cueing sensors | Visual/thermal confirmation, classification, and fire-control support after radar/acoustic/RF cueing. | visible EO<br>thermal IR<br>PTZ<br>image-based classifiers | Kilometer-scale public ID/recognition/detection claims vary heavily by optic, weather, aspect, and cue quality. | Video frame-rate products; slew-to-cue latency is more important than raw frame rate. | Usually one or few high-quality confirmed tracks per gimbal; wide-area cameras vary by design. | human-in-the-loop confirmation<br>thermal/visual cross-check<br>radar slew-to-cue<br>background/weather quality gating | image crop<br>line-of-sight angle<br>visual class confidence<br>operator confirmation state<br>track refinement | **P1**<br>Treat as a confirmatory sensor with weather/visibility masks and cue latency rather than wide-area primary detection. |
| 8 | `passive_rf_esm_direction_finding`<br>Passive RF / ESM / direction-finding sensors | Detect, classify, or locate drone RF control/video/telemetry emissions when present. | passive RF<br>ESM<br>direction finding<br>COMINT | Emission- and environment-dependent; not reliable as a primary range proxy for pre-programmed RF-silent one-way UAVs. | Scan/dwell dependent; public sources usually describe continuous monitoring and direction finding. | Depends on emitter density and channelization; public capacity is rarely comparable across vendors. | protocol/frequency classification<br>direction-of-arrival correlation<br>emitter library matching<br>radar/EO/acoustic confirmation | bearing<br>emitter class<br>frequency band<br>signal strength proxy<br>operator/controller cue where applicable | **P2**<br>Model as optional supporting evidence with explicit missing-signal behavior for autonomous public-proxy one-way attack UAVs. |
| 9 | `passive_radar_multistatic`<br>Passive radar and multistatic illuminator-of-opportunity sensors | Silent gap-filler surveillance using broadcast/cellular/other illuminators rather than own transmitter. | VHF/UHF broadcast<br>FM/DAB/DVB-T<br>cellular<br>passive bistatic/multistatic radar | Geometry- and illuminator-dependent; public examples include passive 3D tracking systems and micro-UAV claims, but not universal target-class envelopes. | Some public systems advertise sub-second or near-real-time tracking; actual cadence depends on illuminators and processing. | Cluster/fusion dependent; public capacity is less standardized than active radar. | bistatic/multistatic consistency<br>illuminator quality gating<br>Doppler processing<br>active-radar cue correlation | passive track<br>bistatic range/Doppler proxy<br>illuminator provenance<br>track confidence | **P2**<br>Implement after active/acoustic layers as a silent cue source with illuminator geometry and urban multipath assumptions. |
| 10 | `organic_shorad_fire_control_radars`<br>Organic SHORAD search/tracking and fire-control radars | Vehicle or battery organic search, tracking, and engagement support for guns/missiles in point defense. | S/X/Ku/K public labels<br>search radar<br>tracking radar<br>EO/thermal backup | Public examples include Gepard 15 km airspace monitoring and 5 km gun engagement; Pantsir/Tor public sources publish tens-of-kilometers class aircraft/UAV proxies by variant. | Engagement-system dependent; public data usually focuses on search/track/engage ranges rather than radar update rates. | Public examples include Tor scanning/tracking subsets and Pantsir tactical-aircraft-sized track counts; exact small-UAV capacity is context dependent. | separate search and tracking sensors<br>EO/thermal backup<br>operator/C2 confirmation<br>fire-control track gates | local search track<br>fire-control track<br>engagement state<br>sensor handoff status | **P2**<br>Represent as downstream engagement-quality consumers of fused tracks, with local radar fallback and saturation limits. |
| 11 | `strategic_long_range_missile_defense_radars`<br>Strategic and high-end missile-defense radars | High-value air and missile defense surveillance/discrimination, not cost-matched primary Shahed-class point defense. | X-band<br>C/G-band public labels<br>large phased array<br>BMD/IAMD | Public sources publish high-end air/missile-defense ranges and target capacities, but these are not Shahed-specific public-proxy ranges. | System-specific and often not public; integration through IAMD/C2 is the relevant proxy. | High-capacity IAMD tracking by design; exact low-altitude small-UAV behavior is not publicly specified. | large-aperture discrimination<br>IAMD track correlation<br>threat classification<br>high-value asset doctrine | IAMD track<br>threat class<br>engagement-quality data<br>discrimination output | **P3**<br>Keep as context/outer-layer cueing only unless a scenario explicitly tests high-value IAMD integration. |

## Representative Public Systems

- `fusion_c2_layered_architecture`: U.S. LIDS / FS-LIDS with FAAD C2; Ukraine Sky Map / Sky Fortress-style fusion; EDGE/SIGN4L SKYSHIELD C2
- `high_resolution_cuas_x_ku`: RTX KuRFS / KuMRFS; HENSOLDT SPEXER 2000 3D MkIII; Blighter A400/A800; Echodyne EchoShield; Robin Radar IRIS; Saab Giraffe 1X
- `tactical_multimission_s_band_aesa`: Leonardo DRS / RADA MHR, nMHR, exMHR; ELTA Green Rock; ELTA MMR / MS-MMR family
- `medium_range_gbad_3d_4d`: AN/MPQ-64 Sentinel; HENSOLDT TRML-4D; Thales GM200; Saab Giraffe AMB / 4A
- `l_band_counterfire_gap_filler`: SRC AN/TPQ-50; LCMR-derived multi-mission radar family
- `distributed_acoustic_networks`: Ukraine Sky Fortress / Zvook-style networks; Sky Map acoustic fusion
- `eo_ir_confirmation_layer`: LIDS EO/IR cameras; Drone Dome EO/IR; SKYSHIELD EO/IR; AUDS EO/IR
- `passive_rf_esm_direction_finding`: SRC EW in LIDS; Drone Dome RF sensors; SKYSHIELD direction finders
- `passive_radar_multistatic`: HENSOLDT Twinvis / TwinSens; Leonardo AULOS; Chinese JY/YLC passive families
- `organic_shorad_fire_control_radars`: Gepard; Pantsir family; Tor family; SkyKnight / Skynex inner-layer systems
- `strategic_long_range_missile_defense_radars`: Patriot AN/MPQ-53/65; LTAMDS / GhostEye; AN/TPY-2

## EchoForge Implementation Notes

- Start with fused track provenance, because false-track handling and cue
  confidence are the realism surface shared by every sensor family.
- Keep public range values as broad proxies by target class or system role; do
  not encode them as guaranteed Shahed/Geran detection truth.
- Separate `detection`, `tracking`, and `classification` products. Public
  sources often advertise one while omitting the others.
- Model clutter and hard negatives as first-class conditions: birds, weather,
  terrain masking, wind turbines, vehicles, multipath, RFI, and acoustic noise.
- Treat passive RF as optional evidence. Pre-programmed one-way attack UAVs may
  have no useful control-link emission for a passive RF sensor to detect.
- Keep generated JSON/Markdown copies under `outputs/`; only this committed
  report belongs in Git.
