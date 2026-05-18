pub mod antenna;
mod backend;
pub mod beamforming;
mod cfar;
mod chain;
pub mod clutter;
pub mod detectors;
pub mod empirical_pfa;
pub mod fusion;
pub mod impairments;
pub mod link_budget;
pub mod micro_doppler_gen;
pub mod mti_mtd;
pub mod scene;
mod pulse_compression;
pub mod propagation;
pub mod rcs;
pub mod rda_cube;
pub mod rfi;
mod sim;
mod tracking_fusion;
mod waveform;

pub use antenna::{
    AntennaPattern, CosinePatternAntenna, IsotropicAntenna, PhasedArrayManifold, TableLookupAntenna,
};
pub use backend::{
    cpu_worker_budget, gpu_worker_budget, ArrayBackend, BackendMode, BackendSelectionError,
    BackendSignals, CpuBackend, GpuBackendUnavailable, RuntimeBackend, RuntimePlan,
};
pub use beamforming::{
    steering_vector, Beamformer, CaponStubBeamformer, DelayAndSumBeamformer, SumBeamformer,
};
pub use cfar::{ca_cfar_1d, ca_cfar_scale, CfarDecision, CfarParams};
pub use chain::{RadarChain, RadarChainOutput};
pub use clutter::{
    apply_clutter_to_profile, generate_clutter_sequence, sample_clutter_amplitude,
    sample_clutter_frame, sample_k_distribution, sample_log_normal, sample_weibull,
    ClutterDistribution, ClutterFrameSample, ClutterProfile, ClutterRegime, TerrainClass,
};
pub use detectors::{
    BlobRdDetector, BlobRdParams, CaCfarDetector, DetectionEvent, DetectionKind, Detector,
    GoCfarDetector, MicroDopplerDetector, MicroDopplerParams, OsCfarDetector, OsCfarParams,
    RangeDoppler, SoCfarDetector,
};
pub use detectors::phase_tiered::{
    classify_boost_sub_state, BoostDecision, BoostSubState, BoostThrustProfile, BoostTierConfig,
    BoostTierDetector, ClimbDecision, ClimbOutTierDetector, ClimbTierConfig, CruiseDecision,
    CruiseTierConfig, CruiseTierDetector, KinematicGate, KinematicObservation, KinematicSample,
    PhaseTieredConfig, PhaseTieredDecision, PhaseTieredDetector, PropulsionClass, SpeedClassifier,
    Tier, TierArbiter, TierTransition, MTI_NOTCH_BODY_DOPPLER_HZ,
};
pub use detectors::tbd::{hough_tbd_detect, TbdConfig, TbdTrackCandidate};
pub use scene::{
    EnvironmentDescriptor, SceneDescriptor, SiteGeometry, TargetClass, TargetEntity,
    TargetKinematics,
};
pub use detectors::cfar_alpha::{
    alpha_library_len, alpha_library_lookup, ca_cfar_scale_gaussian, calibrate_alpha_monte_carlo,
    os_cfar_scale_gaussian, resolve_alpha, CfarVariant, NoiseDistribution,
};
pub use empirical_pfa::{
    calibrate_standard_table, measure_pfa, render_jsonl, render_markdown, wilson_ci_95,
    PfaObservation, PfaTrial,
};
pub use fusion::{DetectorGraphRuntime, FusedDetections};
pub use mti_mtd::{
    apply_mti, doppler_filter_bank, mtd_chain, mti_improvement_factor_db, MtiOrder,
};
pub use impairments::{
    apply_receiver_impairments, sample_receiver_impairments, ReceiverImpairmentProfile,
    ReceiverImpairmentSample,
};
pub use link_budget::{
    evaluate_link_budget, snr_to_target_amplitude, LinkBudget, LinkBudgetResult,
    PropagationContext, BOLTZMANN_J_PER_K, REFERENCE_NOISE_TEMPERATURE_K,
};
pub use micro_doppler_gen::{
    sample_velocity_series, BirdWingbeatGenerator, HelicopterRotorGenerator,
    JetCompressorGenerator, MicroDopplerGenerator, PropellerGenerator,
};
pub use pulse_compression::{
    coefficients, magnitude, matched_filter, pulse_compress, pulse_compress_windowed,
    CompressionWindow,
};
pub use propagation::{
    itu_r_p453_refractivity_n_units, itu_r_p676_gas_attenuation_db,
    itu_r_p838_rain_attenuation_db, min_target_altitude_for_los_m, target_above_horizon,
    two_ray_propagation_factor_magnitude, RainPolarization, EARTH_RADIUS_M,
    EFFECTIVE_EARTH_RADIUS_M, SPEED_OF_LIGHT_M_PER_S, STANDARD_K_FACTOR,
};
pub use rcs::{
    AspectGrid, Polarization, Rcs, RcsLookup, SwerlingModel, SWERLING_DEFAULT_SCAN_SIZE,
};
pub use rda_cube::{
    build_rda_cube, rda_extract_angle_slice, rda_peak, rda_to_rd_sum, AngleGrid, RangeDopplerAngle,
    RdaPeak,
};
pub use rfi::{apply_rfi_to_profile, sample_rfi_frame, RfiFrameSample, RfiProfile};
pub use sim::{
    range_bin_to_m, slow_time_complex_dft, slow_time_dft_magnitude, synthesize_scene,
    synthesize_takeoff_episode, DetectionRecord, EpisodeSeed, NoiseProfile, RadarSimConfig,
    SyntheticEpisode, TakeoffProfile, TargetState,
};
pub use tracking_fusion::{TrackingFusionAdapter, TrackingFusionReport, TrackingTrack};
pub use waveform::{lfm_chirp, LfmChirp};

pub type ComplexSample = num_complex::Complex<f32>;
