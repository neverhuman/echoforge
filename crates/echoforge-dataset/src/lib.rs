//! Dataset and benchmark scaffold for EchoForge.
//!
//! This crate owns split policy, leakage checks, and benchmark placeholders.

pub mod benchmark;
pub mod leakage;
pub mod split;

pub use benchmark::{benchmark_report_template, BenchmarkReportSkeleton};
pub use leakage::{build_leakage_report, LeakageFinding, LeakageReport};
pub use split::{assign_split, default_public_proxy_split_policy, DatasetRecord, SplitKind, SplitPolicy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_policy_fixture_parses() {
        let policy: SplitPolicy = serde_json::from_str(include_str!(
            "../../../tests/datasets/split_policy.json"
        ))
        .expect("split policy fixture should parse");
        assert_eq!(policy.policy_id, "dataset.split.public_proxy_v1");
        assert_eq!(policy.protected_keys.len(), 6);
    }

    #[test]
    fn leakage_fixture_detects_overlap() {
        let records: Vec<DatasetRecord> = serde_json::from_str(include_str!(
            "../../../tests/datasets/leakage_input.json"
        ))
        .expect("leakage fixture should parse");
        let report = build_leakage_report(&records, &default_public_proxy_split_policy());
        assert!(!report.is_clean());
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn benchmark_template_mentions_schema_refs() {
        let template = benchmark_report_template();
        assert!(template.contains("schemas/benchmark_report.schema.json"));
        assert!(template.contains("schemas/dataset_card.schema.json"));
    }
}

