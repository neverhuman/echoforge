# Public-Proxy Physics Dossier — Shahed-136 / Geran-2 Class

Single source of truth for the physical parameters used by EchoForge's
downstream radar simulation lanes (catapult-launch-trajectory,
rust-link-budget-snr-removal, multi-axis RCS lookup) when modelling a
delta-wing pusher-prop one-way uncrewed aerial system in the
Shahed-136 / Geran-2 / Shahed-136-series public-proxy class.

- Schema: `object-packs/public-proxy-v1/`
- Companion files: `source_dossier.yaml`, `object_card.yaml`,
  `material_card.yaml`, `pack.manifest.json`
- Owner: EchoForge dataset curators
- Last revised: 2026-05-18

---

## 1. Strict-open posture statement

Every numeric value, range, table entry, and qualitative description in
this dossier is drawn from public open-source intelligence and from
published peer-reviewed or pre-print radar literature for similar-class
small fixed-wing UAS. No measured platform signatures are included; no
restricted, classified, or vendor-proprietary data is used. Aliases
(Shahed-136, Geran-2, Shahed-136-series) are used in accordance with the
20260518T160000Z naming-policy reversal, which permits these platform
names in default code paths while the Wave-1 banlist guards
(see `agent/banned-terms.toml`) continue to forbid overclaim phrases
such as "validated against real platform", "guaranteed detection range",
and "ground truth signature". Every parameter here is a public-proxy
range with explicit citation; downstream consumers MUST treat it as a
bounded statistical prior, not as a calibrated measurement of any
specific airframe.

---

## 2. Airframe and mass

The Shahed-136-series airframe is a fixed delta-wing pusher-prop
loitering one-way aerial system. Open-source reporting converges on the
following bulk geometry and mass envelope.

| Parameter | Public-proxy range | Citation |
|---|---|---|
| Length (m) | 3.3 – 3.7 | [OSMP 2023], [ArmyRecognition 2022] |
| Wingspan (m) | 2.3 – 2.7 | [OSMP 2023], [ArmyRecognition 2022] |
| Height (m) | 0.35 – 0.75 | [CSIS 2022], visual-imagery estimate |
| Takeoff mass (kg) | ~200 (range 175 – 220) | [ArmyRecognition 2022], [CSIS 2022] |
| Wing planform | Tail-less swept delta with vertical fin tips | [OSMP visual guide 2023] |
| Propulsion configuration | Rear pusher, 2-blade propeller, piston engine | [ArmyRecognition 2022], [OSMP 2023] |
| Tail surfaces | None (tail-less delta); vertical winglets at outer wing | [OSMP visual guide 2023] |
| Canard surfaces | None reported in any public visual reference | [OSMP visual guide 2023] |

OSMP lists the series as approximately 3.5 m long with approximately
2.5 m wingspan. EchoForge widens those public figures into ranges and
excludes payload effects, terminal behavior, and operational route
modelling per the source-dossier-only alias policy in
`source_dossier.yaml`.

---

## 3. Propulsion

The propulsion system is publicly reported as a small piston engine
driving a 2-blade rear pusher propeller. Open reporting most often
attributes the engine to a copy of a Limbach L550E or similar small
two-stroke / four-stroke piston engine in the 35 – 50 hp class.

| Parameter | Public-proxy value | Citation |
|---|---|---|
| Engine type | Small piston, ~35 – 50 hp class | [Janes UAS handbook 2023, public summaries], [CSIS 2022] |
| Propeller blade count B | 2 | [OSMP visual guide 2023], [ArmyRecognition 2022] |
| Cruise RPM (estimated) | 4500 – 6500 rpm | Derived from Limbach L550E public specifications |
| Idle / start RPM | ~1500 – 2500 rpm | Inferred from engine class |
| Cruise blade-pass frequency f_bp = B × (RPM/60) | ~150 – 220 Hz | Derived from B = 2 and cruise RPM range above |

Numeric derivation:

```
B  = 2 blades
RPM_cruise ~ 4500 ... 6500 rpm
f_bp = B * RPM / 60
     = 2 * 4500 / 60  ... 2 * 6500 / 60
     = 150 Hz ... 217 Hz
```

The blade-pass frequency drives the micro-Doppler signature discussed in
Section 7. The estimated RPM range is consistent with the class of
piston engine publicly reported for this airframe family; it is not a
measurement of any specific tested propulsor and should be carried as a
broad envelope in downstream simulation.

---

## 4. Kinematics

| Parameter | Public-proxy range | Citation |
|---|---|---|
| Cruise speed (m/s) | 50 – 55 | [ArmyRecognition 2022] (~185 km/h top speed) |
| Cruise speed (km/h) | 180 – 200 | [ArmyRecognition 2022] |
| Cruise speed (kt) | 97 – 108 | Conversion of the above |
| Max speed (m/s) | ~55 – 60 | [ArmyRecognition 2022], [OSMP 2023] |
| Min sustainable airspeed (m/s) | ~35 – 40 | Inferred from delta-wing planform stall margin |
| Range (km) | 1500 – 2500 | [ArmyRecognition 2022], [CSIS 2022] |
| Endurance (hr) | ~7 – 12 | Range / cruise speed |
| Climb rate (m/s) | 2 – 4 (typical) | Inferred from low installed power and ~200 kg mass |
| Operational altitude band, low-ingress (m AGL) | 60 – 200 | [Defense Express 2023], [CSIS 2022] |
| Operational altitude band, mid-cruise (m AGL) | 500 – 1500 | [Defense Express 2023] |
| Maximum reported altitude (m AGL) | up to ~3000 | [ArmyRecognition 2022] |

The cruise speed range (50 – 55 m/s) is the published top-speed envelope
divided into a working operational band; it is used by EchoForge as a
radar kinematic stressor, not as an operational flight model.

The low-ingress altitude band is what makes detection difficult against
ground clutter; the mid-cruise altitude band is more typical of
transit phases. Both should appear in the Monte-Carlo airspace
configuration so the detector training set captures the operationally
relevant flight regimes.

---

## 5. Launch mode

The Shahed-136-series is launched from a rail or truck-mounted catapult
using a solid-propellant booster (often described in public sources as a
RATO unit). After booster burnout the piston engine takes over.

| Parameter | Public-proxy value | Citation |
|---|---|---|
| Launch method | Rail / catapult with solid-rocket booster (RATO) | [CSIS 2022], [OSMP 2023], [Janes UAS 2023 public summaries] |
| Rail length (m) | ~6 – 8 (truck-bed cluster, 5-tube launcher) | [CSIS 2022] (visual analysis of 5-tube launcher) |
| Booster burn time (s) | ~1 – 2 (typical small RATO) | Inferred from RATO class |
| Exit velocity (m/s) | ~25 – 35 (catapult + booster boost-out) | Derived from kinematics: rail length and booster impulse |
| Initial climb angle from launcher (deg) | ~30 – 45 above horizontal | [CSIS 2022] (visual analysis) |
| Time from launch to powered cruise (s) | ~10 – 30 | Inferred from climb-out kinematics |

Numeric sanity check for exit velocity, assuming a 7 m rail and uniform
acceleration over a 1.5 s boost:

```
v_exit = 2 * L_rail / t_boost
       = 2 * 7 m / 1.5 s
       ~= 9.3 m/s   (rail alone, no booster impulse contribution)

With booster impulse providing the remainder of the boost-out to
operational airspeed (~45 m/s), the airframe reaches powered cruise
within ~10 – 30 seconds of launch as the piston engine spools up and
the airframe trims to climb attitude.
```

The exit velocity stated in the table (~25 – 35 m/s) is the
booster-augmented value at the moment the booster separates; the rail
contribution alone is smaller.

---

## 6. RCS proxy distribution

This section gives a public-proxy RCS envelope for similar-class
fixed-wing small UAS. **These are NOT Shahed-specific measurements** —
no measured-platform RCS is included in this dossier. The numeric
ranges are reference distributions for small fixed-wing UAS drawn from
the published radar literature, and downstream consumers MUST carry
them as bounded statistical proxies, not as calibrated truth.

The values below should be read as **typical aggregate envelopes** for a
small (1 – 3 m wingspan) fixed-wing UAV with a composite skin and a
metallic engine block. The methodology mirrors MDPI Drones 2023 7(1):39
(small fixed-wing UAV RCS aggregate distribution), Ezuma et al.
arXiv:2102.11954 (UAV RF and RCS statistical recognition), and the
K-band drone radar methodology of Rahman & Robertson, Nature Sci Rep
8:17396 (2018).

### 6.1 X-band (10 GHz)

| Aspect | dBsm range (public-proxy reference) | Notes |
|---|---|---|
| Nose-on (0 deg) | -25 to -18 | Small projected area, edge diffraction-limited |
| Quartering (45 / 315 deg) | -20 to -12 | Wing-leading-edge dihedral contribution |
| Broadside (90 / 270 deg) | -12 to -5 | Maximum projected area, wing-flash possible |
| Tail-on (180 deg) | -22 to -15 | Propeller modulation contributes additional time-varying return |
| Above (elevation +30 deg) | reduce by 3 – 6 dB | Reduced projected planform |

### 6.2 S-band (3 GHz)

| Aspect | dBsm range (public-proxy reference) | Notes |
|---|---|---|
| Nose-on | -28 to -20 | RCS scales below resonance for wingspan; airframe enters Mie / Rayleigh transition |
| Broadside | -15 to -8 | Wing chord ~ wavelength; resonant peak possible |
| Tail-on | -25 to -18 | Engine and propulsor still dominate aft sector |

### 6.3 Ku-band (15 GHz)

| Aspect | dBsm range (public-proxy reference) | Notes |
|---|---|---|
| Nose-on | -24 to -16 | Edge / corner specular returns more prominent |
| Broadside | -10 to -3 | Skin-roughness scatter contributes |
| Tail-on | -20 to -12 | Propulsor blade flash strongest in this band |

These ranges are consistent with published K-band drone radar
measurements for small fixed-wing UAS (Rahman & Robertson 2018) and
with the aggregate distribution reported in MDPI Drones 2023 7(1):39.
They are public-proxy reference distributions, NOT Shahed-specific
measurements.

---

## 7. Micro-Doppler signature

The dominant micro-Doppler feature of a piston-powered pusher-prop UAV
is the blade-pass spectral line and its harmonics, modulated by the
amplitude envelope of the propulsor disc backscatter as the relative
geometry between the airframe and the radar changes.

### 7.1 Blade-pass spectrum

Using B = 2 blades and the cruise RPM envelope from Section 3:

```
f_bp_cruise = 150 ... 217 Hz   (fundamental)
f_bp_2nd    = 300 ... 434 Hz   (first harmonic)
f_bp_3rd    = 450 ... 651 Hz   (second harmonic)
```

The fundamental and the first two harmonics are typically the most
prominent lines in a coherent integration window of 50 – 200 ms; higher
harmonics fall off with the propulsor blade thickness and the
backscatter coherence across the disc.

### 7.2 Amplitude modulation depth

For a 2-blade pusher seen from the tail aspect, the propulsor disc
modulates the body return with an amplitude modulation depth in the
range of 3 – 12 dB peak-to-peak (typical for small fixed-wing UAVs per
Rahman & Robertson 2018 methodology). The modulation depth depends on
the ratio of propulsor blade RCS to body RCS at the given aspect, and on
the polarization of the radar.

### 7.3 Aspect-angle dependence

The propulsor is at the tail of the airframe and is partly occluded by
the fuselage and the wing root from forward aspects. Public-proxy
visibility envelope:

| Aspect | Propulsor visibility | Expected micro-Doppler depth |
|---|---|---|
| Tail-on (180 deg) | Full disc visible | Maximum (6 – 12 dB peak-to-peak) |
| Quartering rear (135, 225 deg) | Partial disc | Moderate (3 – 8 dB) |
| Broadside (90, 270 deg) | Edge-on, narrow visibility | Small (1 – 3 dB) |
| Quartering nose (45, 315 deg) | Mostly occluded | Negligible to small (~1 dB) |
| Nose-on (0 deg) | Fully occluded by fuselage | Absent or very weak |

Per the methodology of Rahman & Robertson, Nature Sci Rep 8:17396
(2018), K-band and W-band drone micro-Doppler signatures are dominated
by rotating-blade returns; the same methodology transfers to X / S / Ku
bands with appropriate scaling of the blade-pass and harmonic
amplitudes.

---

## 8. Behavior phases

The detection problem is partitioned into three behaviorally distinct
phases. These drive the three-phase decomposition called out in the
correctness-first plan and consumed by the catapult-launch-trajectory
packet.

| Phase | Time window | Altitude (m AGL) | Speed (m/s) | Notes |
|---|---|---|---|---|
| 1 — Catapult launch | 0 – 30 s | 0 – 80 | 25 – 45 (transient) | Booster boost-out, transition to powered flight |
| 2 — Climb | 30 – 180 s | 80 – 800 | 45 – 55 | Steady climb at low climb rate |
| 3 — Cruise (steady state) | > 180 s | 100 – 3000 | 50 – 55 | Operational range per public reports |

Phase 1 begins at booster ignition and ends when the airframe is on
powered cruise from the piston engine. During this phase the airframe
is in a non-steady kinematic regime: speed climbs rapidly from the
catapult exit value (~25 – 30 m/s) to powered cruise (~45 m/s), and the
altitude climbs from 0 to roughly 80 m AGL. The booster burn produces a
brief but bright thermal signature that is out of scope for this radar
dossier.

Phase 2 is a quasi-steady climb at the typical climb rate (2 – 4 m/s)
until the airframe reaches the lower bound of the cruise altitude band.

Phase 3 is steady-state cruise. The altitude band is wide (100 – 3000
m AGL) because operational reporting describes both low-altitude ingress
profiles and higher-altitude transit profiles. Downstream simulation
should sample across this band.

---

## 9. Detection-relevant geometry and physics constants

The constants in this section are universal physics or
internationally-standardized conventions, used by the radar-horizon and
link-budget calculations in `crates/echoforge-radar/src/` and by the
catapult-launch-trajectory packet. They are not platform-specific.

```
# Effective Earth radius for radar-horizon calculations (standard 4/3-Earth)
R_eff = 8.5e6 m

# Speed of light in vacuum (CODATA defined value)
c = 2.998e8 m/s

# Boltzmann constant (CODATA defined value)
k = 1.38e-23 J/K

# International Standard Atmosphere, sea level
T_isa_sl = 288.15 K
P_isa_sl = 101325 Pa
```

### 9.1 Radar horizon for low-altitude targets

For a radar antenna at height h_r and a target at height h_t, the
4/3-Earth radar horizon is:

```
d_horizon = sqrt(2 * R_eff * h_r) + sqrt(2 * R_eff * h_t)
```

At a target altitude of 100 m AGL (low-ingress profile) and a typical
ground radar antenna height of 10 m, the line-of-sight horizon is
approximately:

```
d_horizon = sqrt(2 * 8.5e6 * 10) + sqrt(2 * 8.5e6 * 100)
          ~= 13.0 km + 41.2 km
          ~= 54.2 km
```

At a target altitude of 1000 m AGL, the horizon extends to approximately
143 km. This is geometric line-of-sight only; atmospheric attenuation
and the radar equation set the actual detection range.

### 9.2 Doppler shift at cruise

For a target closing at v_c = 55 m/s and an X-band radar at f = 10 GHz,
the Doppler shift is:

```
f_d = 2 * v_c * f / c
    = 2 * 55 * 10e9 / 2.998e8
    ~= 3.67 kHz
```

For a tangential pass with the same speed, the Doppler shift is near
zero. The blade-pass micro-Doppler lines (Section 7.1) sit on top of
this body Doppler.

---

## 10. References

All sources cited in this dossier are public. Where a DOI or stable URL
exists it is given; where the source is a public open-source-intelligence
report or a vendor public summary, the institution and retrieval window
are given.

### Public open-source-intelligence and public technical reporting

1. **Open Source Munitions Portal (OSMP) — Shahed-136-series catalogue
   entry.** Open Source Munitions Portal, retrieved 2026-05-18.
   URL: <https://osmp.ngo/model/shahed-136-series/>
2. **OSMP — Shahed-131 / Shahed-136 UAVs: A visual guide.** Open Source
   Munitions Portal, retrieved 2026-05-18.
   URL: <https://osmp.ngo/collection/shahed-131-136-uavs-a-visual-guide/>
3. **Army Recognition — Shahed-136 loitering munition public technical
   data.** Army Recognition, retrieved 2026-05-18.
   URL: <https://www.armyrecognition.com/military-products/army/unmanned-systems/unmanned-aerial-vehicles/shahed-136-loitering-munition-kamikaze-suicide-drone-technical-data>
4. **CSIS — Iran's Drones in Ukraine: A Visual Investigation.** Center
   for Strategic and International Studies, 2022, retrieved 2026-05-18.
   (Use as public-source visual analysis of the 5-tube launcher and
   airframe geometry.)
5. **Defense Express — Public reporting on Shahed-136 / Geran-2
   operational profiles.** Defense Express, 2023, retrieved 2026-05-18.
   (Use as public-source reporting on low-altitude ingress profiles and
   mid-cruise altitude bands.)
6. **Janes Unmanned Aerial Systems handbook — public summaries.**
   Public-domain summaries of the Janes UAS handbook (full handbook is
   commercial). Used here for engine class corroboration only.

### Published radar literature for similar-class small UAS

7. **MDPI Drones 2023, 7(1):39 — Small fixed-wing UAV RCS aggregate
   distribution.** MDPI, 2023. DOI: 10.3390/drones7010039.
   URL: <https://www.mdpi.com/2504-446X/7/1/39>
8. **Ezuma et al., arXiv:2102.11954 — UAV RF and RCS statistical
   recognition.** arXiv pre-print, 2021.
   URL: <https://arxiv.org/abs/2102.11954>
9. **Rahman & Robertson, Nature Sci Rep 8:17396 (2018) — Radar
   micro-Doppler signatures of drones and birds at K-band and W-band.**
   Nature Scientific Reports, 2018. DOI: 10.1038/s41598-018-35880-9.
   URL: <https://www.nature.com/articles/s41598-018-35880-9>

### Reference textbooks and physical constants

10. **Skolnik, M. I. — Introduction to Radar Systems, 3rd ed.**
    McGraw-Hill, 2001. Used for the radar equation, the radar-horizon
    geometry, the 4/3-Earth convention, and the Swerling fluctuation
    model formulations referenced in the multi-axis RCS lookup packet.
11. **CODATA Internationally Recommended Values of the Fundamental
    Physical Constants.** National Institute of Standards and
    Technology. URL: <https://physics.nist.gov/cuu/Constants/>
    (Used for c and k.)
12. **International Standard Atmosphere — ISO 2533:1975.**
    International Organization for Standardization. Used for sea-level
    temperature and pressure.

---

## Appendix A — Cross-references to companion files

- Aliases and prior envelope: `object-packs/public-proxy-v1/source_dossier.yaml`
- Object card fixture: `object-packs/public-proxy-v1/object_card.yaml`
- Material card (composite skin baseline): `object-packs/public-proxy-v1/material_card.yaml`
- Pack manifest: `object-packs/public-proxy-v1/pack.manifest.json`

## Appendix B — Banlist compliance note

This dossier was authored against the Wave-1 banlist in
`agent/banned-terms.toml`. The 20260518T160000Z naming-policy reversal
permits Shahed-136, Geran-2, and Shahed-136-series in default code paths;
the Wave-1 overclaim guards remain active. The vendor-scrub gate
(`tools/vendor_scrub.mjs`) reports zero matches against the published
banlist for this file.
