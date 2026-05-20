    use super::*;

    fn sample_request() -> SolverPlanRequest {
        SolverPlanRequest {
            object_card_id: "ef:object_card:proxy-1:0123456789abcdef:1".to_string(),
            mesh_manifest_id: Some("ef:mesh_manifest:proxy-1:fedcba9876543210:1".to_string()),
            material_card_id: None,
            frequency_range_hz: NumericRange {
                min: 8.0e9,
                max: 12.0e9,
            },
            tx_polarization: echoforge_solver_contract::Polarization::H,
            rx_polarization: echoforge_solver_contract::Polarization::H,
            monostatic: true,
        }
    }

    #[test]
    fn new_adapter_is_not_available() {
        let a = SagittaSbrAdapter::new();
        assert!(
            !a.is_available(),
            "freshly-constructed adapter must be unwired"
        );
        assert!(a.binary_path.is_none());
        assert!(a.container_digest.is_none());
    }

    #[test]
    fn with_binary_makes_adapter_available() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        assert!(a.is_available());
        assert_eq!(
            a.binary_path.as_deref(),
            Some(std::path::Path::new("/fake/path/to/sagittasbr"))
        );
    }

    #[test]
    fn with_container_digest_records_digest_without_changing_availability() {
        let a = SagittaSbrAdapter::new().with_container_digest(
            "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
        );
        assert!(!a.is_available(), "digest alone must not flip availability");
        assert!(a.container_digest.is_some());
    }

    #[test]
    fn identity_and_license_class_are_constant() {
        let a = SagittaSbrAdapter::new();
        assert_eq!(a.name(), "sagitta-sbr");
        assert_eq!(a.name(), ADAPTER_NAME);
        assert!(
            a.version().starts_with("unwired-"),
            "version must be unwired-tagged; got {:?}",
            a.version()
        );
        assert_eq!(a.license_class(), LicenseClass::OptionalOpen);

        // Wiring a binary must NOT change identity or license class.
        let b = SagittaSbrAdapter::with_binary("/fake");
        assert_eq!(b.name(), a.name());
        assert_eq!(b.version(), a.version());
        assert_eq!(b.license_class(), a.license_class());
    }

    #[test]
    fn capabilities_declare_honest_sbr_surface() {
        let caps = SagittaSbrAdapter::new().capabilities().clone();
        assert_eq!(caps.solves_pec, Some(true));
        assert_eq!(caps.solves_dielectric, Some(true));
        assert_eq!(caps.solves_layered_dielectric, Some(false));
        assert_eq!(caps.solves_anisotropy, Some(false));
        assert_eq!(caps.solves_monostatic, Some(true));
        assert_eq!(caps.solves_bistatic, Some(true));
        assert_eq!(caps.supports_far_field, Some(true));
        assert_eq!(caps.supports_near_field, Some(false));
        assert_eq!(caps.parallelism_class, Some(ParallelismClass::Gpu));
        let band = caps
            .frequency_range_hz
            .as_ref()
            .expect("frequency band declared");
        assert!(band.min > 0.0 && band.min < band.max);
        assert!(
            caps.max_electrical_size_lambda.unwrap_or(0.0) > 0.0,
            "max electrical size must be positive"
        );
    }

    #[test]
    fn plan_on_unavailable_adapter_returns_unsupported() {
        let a = SagittaSbrAdapter::new();
        match a.plan(&sample_request()) {
            Err(SolverError::Unsupported(msg)) => {
                let m = msg.to_lowercase();
                assert!(
                    m.contains("not configured"),
                    "unavailable plan() error should mention not configured; got {msg:?}",
                );
            }
            other => panic!("expected Unsupported from plan() on unwired adapter, got {other:?}"),
        }
    }

    #[test]
    fn plan_on_available_adapter_returns_populated_plan() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        let plan = a.plan(&sample_request()).expect("plan on unwired adapter");
        assert!(
            plan.plan_id.starts_with("sagitta-sbr-"),
            "plan_id must be namespaced by this adapter; got {:?}",
            plan.plan_id,
        );
        assert!(
            !plan.notes.is_empty(),
            "unwired plan must carry at least one note"
        );
        // Cost estimates are intentionally zero in the unwired adapter.
        // Document that expectation so a future wired adapter that starts
        // returning real estimates trips the test and forces a review.
        assert_eq!(plan.estimated_frequency_points, 0);
        assert_eq!(plan.estimated_angle_samples, 0);
        assert_eq!(plan.estimated_peak_memory_mb, 0);
        let joined = plan.notes.join(" ").to_lowercase();
        assert!(
            joined.contains("unwired") || joined.contains("not yet connected"),
            "plan notes must flag the unwired status; got {:?}",
            plan.notes,
        );
    }

    #[test]
    fn plan_rejects_out_of_band_frequency_request() {
        let mut req = sample_request();
        req.frequency_range_hz = NumericRange {
            min: 1.0e12, // 1 THz, well above declared 100 GHz ceiling
            max: 2.0e12,
        };
        let a = SagittaSbrAdapter::with_binary("/fake");
        match a.plan(&req) {
            Err(SolverError::InvalidRequest(msg)) => {
                assert!(
                    msg.to_lowercase().contains("outside declared range"),
                    "frequency rejection should name the declared range; got {msg:?}",
                );
            }
            other => panic!("expected InvalidRequest for out-of-band frequency, got {other:?}"),
        }
    }

    #[test]
    fn run_on_available_adapter_returns_unsupported_unwired_message() {
        let a = SagittaSbrAdapter::with_binary("/fake/path/to/sagittasbr");
        let plan = a
            .plan(&sample_request())
            .expect("plan succeeds on unwired adapter");
        match a.run(&plan) {
            Err(SolverError::Unsupported(msg)) => {
                let m = msg.to_lowercase();
                assert!(
                    m.contains("not yet wired") || m.contains("not wired"),
                    "run() error MUST say compute path is not wired; got {msg:?}",
                );
            }
            Ok(report) => panic!(
                "unwired run() must NEVER return Ok — got report {report:?}. \
                 Unwired adapters that return fake reports would silently corrupt \
                 downstream validation."
            ),
            other => panic!("expected Unsupported from run(), got {other:?}"),
        }
    }

    #[test]
    fn run_rejects_foreign_plan_id() {
        let a = SagittaSbrAdapter::with_binary("/fake");
        let foreign = SolverPlan {
            plan_id: "null-plan-1".to_string(),
            estimated_frequency_points: 0,
            estimated_angle_samples: 0,
            estimated_peak_memory_mb: 0,
            notes: Vec::new(),
        };
        match a.run(&foreign) {
            Err(SolverError::InvalidPlan(msg)) => {
                assert!(
                    msg.contains("not produced by this adapter"),
                    "foreign-plan rejection should name the cause; got {msg:?}",
                );
            }
            other => panic!("expected InvalidPlan for foreign plan_id, got {other:?}"),
        }
    }

    #[test]
    fn capabilities_round_trip_through_serde() {
        let caps = SagittaSbrAdapter::new().capabilities().clone();
        let encoded = serde_json::to_string(&caps).expect("encode caps");
        let decoded: SolverCapabilities = serde_json::from_str(&encoded).expect("decode caps");
        assert_eq!(decoded, caps);
        // Spot-check that the GPU enum survived as the expected wire form.
        assert!(
            encoded.contains("\"gpu\""),
            "encoded caps missing GPU class: {encoded}"
        );
    }

    #[test]
    fn adapter_object_safe_via_trait_object() {
        // Exercise SagittaSbrAdapter through a &dyn SolverAdapter so any
        // future change that breaks object-safety on the contract trait
        // trips here, not in a downstream registry crate.
        let a: Box<dyn SolverAdapter> = Box::new(SagittaSbrAdapter::with_binary("/fake"));
        assert_eq!(a.name(), ADAPTER_NAME);
        assert_eq!(a.license_class(), LicenseClass::OptionalOpen);
        let plan = a.plan(&sample_request()).expect("plan via trait object");
        assert!(plan.plan_id.starts_with("sagitta-sbr-"));
        assert!(matches!(a.run(&plan), Err(SolverError::Unsupported(_))));
    }
