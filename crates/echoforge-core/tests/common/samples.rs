// Canonical minimal-instance builders for every EchoForge document kind.
//
// Shared between the schema_round_trip integration test, the schemars_drift
// integration test, and the `emit_canonical` binary used by the Node-side
// `tools/canonical_diff.mjs` byte-identity check.
//
// Builders deliberately produce the *same* payload every call (no time, no
// randomness) so the canonical JSON SHA-256 is stable across runs. Each
// builder returns the finalized document (id + fingerprint populated by
// `impl_document!::finalize`).

#![allow(dead_code)]

use echoforge_core::{
    ComplexScalar, DatasetCard, DatasetSplits, DetectorGraph, EchosigManifest, LicenseInfo,
    MaterialCard, MeshManifest, NumericRange, ObjectCard, Provenance, RadarEpisode, RcsCampaign,
    Scenario, SensorArchetype, SolverCard, ValidationCheck, ValidationInfo, Vector3,
};
// The `models::ValidationReport` document type is shadowed by
// `artifact::ValidationReport` in the crate-root re-exports, so reach into
// the module directly.
use echoforge_core::models::ValidationReport;

pub const KINDS: &[&str] = &[
    "object_card",
    "material_card",
    "mesh_manifest",
    "solver_card",
    "rcs_campaign",
    "echosig_manifest",
    "sensor_archetype",
    "scenario",
    "radar_episode",
    "detector_graph",
    "dataset_card",
    "validation_report",
];

fn provenance() -> Provenance {
    Provenance {
        source_kind: "synthetic".to_string(),
        source_refs: vec!["tests/contract".to_string()],
        generated_by: "echoforge-core-contract".to_string(),
        generated_at: "2026-05-18T00:00:00Z".to_string(),
        fingerprint_sha256: String::new(),
    }
}

fn license() -> LicenseInfo {
    LicenseInfo {
        spdx_id: "CC0-1.0".to_string(),
        notice: String::new(),
    }
}

fn validation_info() -> ValidationInfo {
    ValidationInfo {
        tier: "basic".to_string(),
        status: "pass".to_string(),
        uncertainty_score: 0.25,
        checks: vec![ValidationCheck {
            name: "contract-check".to_string(),
            status: "pass".to_string(),
            message: "ok".to_string(),
        }],
    }
}

// A canonical pre-computed `ef:` id used wherever a schema requires a
// deterministic_id-shaped cross-document reference. Matches the regex
// `^ef:[a-z0-9_-]+:[a-z0-9_-]+:[0-9a-f]{16}:1$`.
fn ref_id(kind: &str) -> String {
    format!("ef:{kind}:proxy.contract:0123456789abcdef:1")
}

pub fn object_card() -> ObjectCard {
    ObjectCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        display_name: "Contract Proxy Object".to_string(),
        object_family: "fixed_wing_uav".to_string(),
        geometry_variant: "baseline".to_string(),
        material_variant: "default".to_string(),
        dimensions_m: Vector3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        },
        tags: vec!["proxy".to_string()],
    }
    .finalize()
    .expect("object_card finalize")
}

pub fn material_card() -> MaterialCard {
    MaterialCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        material_name: "Contract Material".to_string(),
        material_family: "dielectric".to_string(),
        frequency_range_hz: NumericRange {
            min: 1.0e9,
            max: 1.0e10,
        },
        permittivity: ComplexScalar {
            real: 2.5,
            imag: 0.01,
        },
        conductivity_s_per_m: 0.0,
        roughness_m: 0.0,
    }
    .finalize()
    .expect("material_card finalize")
}

pub fn mesh_manifest() -> MeshManifest {
    MeshManifest {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        mesh_name: "Contract Mesh".to_string(),
        mesh_format: "glb".to_string(),
        source_files: vec!["meshes/contract.glb".to_string()],
        units: "m".to_string(),
        triangle_count: 1024,
        watertight: true,
        mesh_sha256: "0".repeat(64),
    }
    .finalize()
    .expect("mesh_manifest finalize")
}

pub fn solver_card() -> SolverCard {
    SolverCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        solver_name: "Contract Solver".to_string(),
        solver_family: "po".to_string(),
        version: "1.0.0".to_string(),
        container_image: "example.invalid/solver@sha256:".to_string() + &"0".repeat(64),
        supported_polarizations: vec!["H".to_string(), "V".to_string()],
    }
    .finalize()
    .expect("solver_card finalize")
}

pub fn rcs_campaign() -> RcsCampaign {
    RcsCampaign {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        campaign_name: "Contract Campaign".to_string(),
        object_card_id: ref_id("object_card"),
        solver_card_id: ref_id("solver_card"),
        frequency_range_hz: NumericRange {
            min: 1.0e9,
            max: 1.0e10,
        },
        azimuth_deg: NumericRange {
            min: 0.0,
            max: 360.0,
        },
        tx_polarization: "H".to_string(),
        rx_polarization: "V".to_string(),
        run_count: 1,
    }
    .finalize()
    .expect("rcs_campaign finalize")
}

pub fn echosig_manifest() -> EchosigManifest {
    EchosigManifest {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        artifact_name: "Contract Echosig".to_string(),
        object_card_id: ref_id("object_card"),
        tensor_axes: vec!["frequency_hz".to_string()],
        tensor_paths: vec!["tensors/contract.zarr".to_string()],
        qa_paths: vec![],
    }
    .finalize()
    .expect("echosig_manifest finalize")
}

pub fn sensor_archetype() -> SensorArchetype {
    SensorArchetype {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        sensor_name: "Contract Sensor".to_string(),
        band_name: "X".to_string(),
        waveform_family: "fmcw".to_string(),
        center_frequency_hz: 9.4e9,
        sample_rate_hz: 1.0e8,
    }
    .finalize()
    .expect("sensor_archetype finalize")
}

pub fn scenario() -> Scenario {
    Scenario {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        scenario_name: "Contract Scenario".to_string(),
        sensor_archetype_id: ref_id("sensor_archetype"),
        object_card_ids: vec![ref_id("object_card")],
        environment_label: "open_field".to_string(),
        seed: 42,
    }
    .finalize()
    .expect("scenario finalize")
}

pub fn radar_episode() -> RadarEpisode {
    RadarEpisode {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        episode_name: "Contract Episode".to_string(),
        scenario_id: ref_id("scenario"),
        sample_rate_hz: 1.0e8,
        product_paths: vec!["products/range_doppler.zarr".to_string()],
    }
    .finalize()
    .expect("radar_episode finalize")
}

pub fn detector_graph() -> DetectorGraph {
    DetectorGraph {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        graph_name: "Contract Detector".to_string(),
        nodes: vec!["cfar".to_string(), "dbscan".to_string()],
        edges: vec!["cfar->dbscan".to_string()],
    }
    .finalize()
    .expect("detector_graph finalize")
}

pub fn dataset_card() -> DatasetCard {
    DatasetCard {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        dataset_name: "Contract Dataset".to_string(),
        source_campaign_ids: vec![ref_id("rcs_campaign")],
        splits: DatasetSplits {
            train: 100,
            validation: 25,
            test: 25,
        },
    }
    .finalize()
    .expect("dataset_card finalize")
}

pub fn validation_report() -> ValidationReport {
    ValidationReport {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: "proxy.contract".to_string(),
        provenance: provenance(),
        license: license(),
        validation: validation_info(),
        report_name: "Contract Report".to_string(),
        subject_kind: "object_card".to_string(),
        subject_id: ref_id("object_card"),
        checks: vec![ValidationCheck {
            name: "structure".to_string(),
            status: "pass".to_string(),
            message: "ok".to_string(),
        }],
        overall_status: "pass".to_string(),
    }
    .finalize()
    .expect("validation_report finalize")
}

// Serialize the canonical sample for `kind` to a `serde_json::Value`. Errors
// (unknown kind) are propagated as `Err(String)`.
pub fn canonical_value(kind: &str) -> Result<serde_json::Value, String> {
    let value = match kind {
        "object_card" => serde_json::to_value(object_card()),
        "material_card" => serde_json::to_value(material_card()),
        "mesh_manifest" => serde_json::to_value(mesh_manifest()),
        "solver_card" => serde_json::to_value(solver_card()),
        "rcs_campaign" => serde_json::to_value(rcs_campaign()),
        "echosig_manifest" => serde_json::to_value(echosig_manifest()),
        "sensor_archetype" => serde_json::to_value(sensor_archetype()),
        "scenario" => serde_json::to_value(scenario()),
        "radar_episode" => serde_json::to_value(radar_episode()),
        "detector_graph" => serde_json::to_value(detector_graph()),
        "dataset_card" => serde_json::to_value(dataset_card()),
        "validation_report" => serde_json::to_value(validation_report()),
        other => return Err(format!("unknown kind: {other}")),
    };
    value.map_err(|e| format!("serialize {kind}: {e}"))
}
