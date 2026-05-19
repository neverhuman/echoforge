use super::*;

fn sample_request() -> SolverPlanRequest {
    SolverPlanRequest {
        object_card_id: "ef:object_card:proxy-1:0123456789abcdef:1".to_string(),
        mesh_manifest_id: None,
        material_card_id: None,
        frequency_range_hz: NumericRange {
            min: 8.0e9,
            max: 12.0e9,
        },
        tx_polarization: Polarization::H,
        rx_polarization: Polarization::H,
        monostatic: true,
    }
}

fn sample_plan() -> SolverPlan {
    SolverPlan {
        plan_id: "null-plan-1".to_string(),
        estimated_frequency_points: 0,
        estimated_angle_samples: 0,
        estimated_peak_memory_mb: 0,
        notes: Vec::new(),
    }
}

#[test]
fn null_solver_reports_expected_identity_and_license() {
    let s = NullSolver::new();
    assert_eq!(s.name(), "null");
    assert_eq!(s.version(), "0.0.0");
    assert_eq!(s.license_class(), LicenseClass::CoreOpen);
    // No capability flags asserted by default.
    assert!(s.capabilities().solves_pec.is_none());
    assert!(s.capabilities().solves_monostatic.is_none());
    assert!(s.capabilities().parallelism_class.is_none());
}

#[test]
fn null_solver_returns_unsupported_on_plan_and_run() {
    let s = NullSolver::new();
    match s.plan(&sample_request()) {
        Err(SolverError::Unsupported(_)) => {}
        other => panic!("expected Unsupported from plan, got {other:?}"),
    }
    match s.run(&sample_plan()) {
        Err(SolverError::Unsupported(_)) => {}
        other => panic!("expected Unsupported from run, got {other:?}"),
    }
}

#[test]
fn null_solver_with_identity_overrides_name_and_version() {
    let s = NullSolver::with_identity("test-null", "9.9.9");
    assert_eq!(s.name(), "test-null");
    assert_eq!(s.version(), "9.9.9");
    // License class is still CoreOpen even with custom identity.
    assert_eq!(s.license_class(), LicenseClass::CoreOpen);
}

#[test]
fn license_class_round_trips_through_serde() {
    for (variant, wire) in [
        (LicenseClass::CoreOpen, "\"core_open\""),
        (LicenseClass::OptionalOpen, "\"optional_open\""),
        (LicenseClass::RestrictedPlugin, "\"restricted_plugin\""),
    ] {
        let encoded = serde_json::to_string(&variant).expect("encode");
        assert_eq!(encoded, wire, "wire form for {variant:?}");
        let decoded: LicenseClass = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded, variant);
    }
}

#[test]
fn parallelism_class_round_trips_through_serde() {
    for (variant, wire) in [
        (ParallelismClass::SingleCore, "\"single_core\""),
        (ParallelismClass::MultiCore, "\"multi_core\""),
        (ParallelismClass::Gpu, "\"gpu\""),
        (ParallelismClass::Distributed, "\"distributed\""),
    ] {
        let encoded = serde_json::to_string(&variant).expect("encode");
        assert_eq!(encoded, wire);
        let decoded: ParallelismClass = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded, variant);
    }
}

#[test]
fn capabilities_round_trip_skips_absent_fields() {
    // Fully populated: round-trip preserves every field.
    let full = SolverCapabilities {
        frequency_range_hz: Some(NumericRange {
            min: 1.0e9,
            max: 18.0e9,
        }),
        solves_pec: Some(true),
        solves_dielectric: Some(false),
        solves_layered_dielectric: Some(false),
        solves_anisotropy: Some(false),
        solves_bistatic: Some(true),
        solves_monostatic: Some(true),
        supports_far_field: Some(true),
        supports_near_field: Some(false),
        max_electrical_size_lambda: Some(64.0),
        parallelism_class: Some(ParallelismClass::Gpu),
    };
    let encoded = serde_json::to_string(&full).expect("encode full");
    let decoded: SolverCapabilities = serde_json::from_str(&encoded).expect("decode full");
    assert_eq!(decoded, full);

    // Empty capabilities serialize to an empty object (all fields
    // skipped); decode round-trips back to default.
    let empty = SolverCapabilities::default();
    let encoded_empty = serde_json::to_string(&empty).expect("encode empty");
    assert_eq!(encoded_empty, "{}");
    let decoded_empty: SolverCapabilities =
        serde_json::from_str(&encoded_empty).expect("decode empty");
    assert_eq!(decoded_empty, empty);
}

#[test]
fn inputs_outputs_convergence_round_trip() {
    let inputs = SolverInputs {
        requires_mesh: Some(true),
        requires_material_card: Some(true),
        requires_polarization: Some(true),
        frequency_sampling: Some(FrequencySampling::Adaptive),
    };
    let outputs = SolverOutputs {
        produces_rcs_cube: Some(true),
        produces_currents: Some(false),
        produces_near_field_volume: Some(false),
    };
    let convergence = SolverConvergence {
        produces_richardson_estimate: Some(true),
        produces_cross_solver_delta: Some(true),
    };

    for (name, value) in [
        (
            "inputs",
            serde_json::to_string(&inputs).expect("encode inputs"),
        ),
        (
            "outputs",
            serde_json::to_string(&outputs).expect("encode outputs"),
        ),
        (
            "convergence",
            serde_json::to_string(&convergence).expect("encode convergence"),
        ),
    ] {
        // Each encoded form should be a JSON object.
        assert!(value.starts_with('{'), "{name} did not encode as object");
    }

    let inputs_back: SolverInputs =
        serde_json::from_str(&serde_json::to_string(&inputs).unwrap()).unwrap();
    assert_eq!(inputs_back, inputs);
    let outputs_back: SolverOutputs =
        serde_json::from_str(&serde_json::to_string(&outputs).unwrap()).unwrap();
    assert_eq!(outputs_back, outputs);
    let convergence_back: SolverConvergence =
        serde_json::from_str(&serde_json::to_string(&convergence).unwrap()).unwrap();
    assert_eq!(convergence_back, convergence);
}

#[test]
fn solver_error_messages_are_distinct_per_variant() {
    let unsupported = SolverError::Unsupported("no can do".to_string()).to_string();
    let invalid_req = SolverError::InvalidRequest("missing mesh".to_string()).to_string();
    let invalid_plan = SolverError::InvalidPlan("wrong adapter".to_string()).to_string();
    let runtime = SolverError::Runtime("solver crashed".to_string()).to_string();
    let internal = SolverError::Internal("oops".to_string()).to_string();
    // All five should be distinct human-readable strings carrying
    // their inner message.
    for (name, text, needle) in [
        ("unsupported", &unsupported, "no can do"),
        ("invalid_request", &invalid_req, "missing mesh"),
        ("invalid_plan", &invalid_plan, "wrong adapter"),
        ("runtime", &runtime, "solver crashed"),
        ("internal", &internal, "oops"),
    ] {
        assert!(
            text.contains(needle),
            "{name} error did not include inner message: {text}"
        );
    }
    // Distinctness: at minimum, no two messages collide.
    let all = [
        unsupported.clone(),
        invalid_req.clone(),
        invalid_plan.clone(),
        runtime.clone(),
        internal.clone(),
    ];
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(all[i], all[j], "duplicate error text at {i},{j}");
        }
    }
}
