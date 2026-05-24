Radar noise / clutter / interference backlog

This directory tracks public-proxy robustness topics that should feed future
scene generation and audit work. The entries below are intentionally broad and
non-measured.

Categories:
- Receiver: AWGN floor, phase noise, amplitude scintillation, clipping, IQ
  imbalance, dropped pulses, AGC saturation.
- Propagation: horizon masking, two-ray multipath, rain attenuation, gaseous
  attenuation, path-dependent delay migration.
- Clutter: terrain glint, ground vehicles, vegetation, weather returns, sea
  clutter, heavy-tailed clutter regimes.
- Interference: co-channel emitters, CW bursts, RFI bursts, sidelobe spillover,
  impulsive interference.
- Scene dynamics: multi-entity crossings, ghost returns, target masking, flock
  motion, static-to-moving transitions, zero-target scenes.
- Tracker: initiation/confirmation latency, missed detections, false tracks,
  fragmentation, track coalescing.
- Validation: seed-only leakage, metadata leakage, target-masked AUC, no-target
  false-alarm rate, group holdout by scene/sensor/target/noise seed.

Backlog principle:
- Treat hard negatives as robustness work, not as evasion optimization.
- Keep all claims at the public-proxy / uncertainty-scored level.
