use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkReportSkeleton {
    pub schema_ref: String,
    pub dataset_card_schema_ref: String,
    pub required_sections: Vec<String>,
    pub required_artifacts: Vec<String>,
    pub known_limitations: Vec<String>,
}

pub fn benchmark_report_template() -> String {
    let skeleton = BenchmarkReportSkeleton {
        schema_ref: "schemas/benchmark_report.schema.json".to_string(),
        dataset_card_schema_ref: "schemas/dataset_card.schema.json".to_string(),
        required_sections: vec![
            "summary".to_string(),
            "dataset_scope".to_string(),
            "split_policy".to_string(),
            "leakage_checks".to_string(),
            "benchmark_metrics".to_string(),
            "known_limitations".to_string(),
            "artifact_manifest".to_string(),
        ],
        required_artifacts: vec![
            "dataset_card.md".to_string(),
            "leakage_report.json".to_string(),
            "split_manifest.json".to_string(),
        ],
        known_limitations: vec![
            "Proxy data is not measured truth.".to_string(),
            "Geometry/material/source assumptions must remain explicit.".to_string(),
            "Hard-negative coverage is incomplete until the pack grows.".to_string(),
        ],
    };

    format!(
        "# Benchmark Report Template\n\n\
schema_ref: {schema_ref}\n\
dataset_card_schema_ref: {dataset_card_schema_ref}\n\n\
## Required Sections\n{required_sections}\n\n\
## Required Artifacts\n{required_artifacts}\n\n\
## Known Limitations\n{known_limitations}\n\n\
## Suggested Closing Line\n\
This benchmark uses public-proxy artifacts with explicit uncertainty and leakage checks.\n",
        schema_ref = skeleton.schema_ref,
        dataset_card_schema_ref = skeleton.dataset_card_schema_ref,
        required_sections = bullet_list(&skeleton.required_sections),
        required_artifacts = bullet_list(&skeleton.required_artifacts),
        known_limitations = bullet_list(&skeleton.known_limitations),
    )
}

fn bullet_list(items: &[String]) -> String {
    items
        .iter()
        .map(|item| format!("- {item}"))
        .collect::<Vec<_>>()
        .join("\n")
}

