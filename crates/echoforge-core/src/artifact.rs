use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::validation::{ensure_non_empty, ensure_slug};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    Draft,
    AnalyticV1,
    CrossSolverV1,
    MeasuredAnchor,
}

impl ValidationTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationTier::Draft => "draft",
            ValidationTier::AnalyticV1 => "analytic_v1",
            ValidationTier::CrossSolverV1 => "cross_solver_v1",
            ValidationTier::MeasuredAnchor => "measured_anchor",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub source: String,
    pub generated_by: String,
    pub created_at_utc: String,
    pub seed: u64,
    pub lineage: Vec<String>,
}

impl ProvenanceRecord {
    pub fn unscored() -> Self {
        Self {
            source: "synthetic".to_string(),
            generated_by: "echoforge-core".to_string(),
            created_at_utc: "1970-01-01T00:00:00Z".to_string(),
            seed: 0,
            lineage: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LicenseRecord {
    pub expression: String,
    pub spdx_id: Option<String>,
    pub notes: Option<String>,
}

impl LicenseRecord {
    pub fn unscored() -> Self {
        Self {
            expression: "Apache-2.0".to_string(),
            spdx_id: Some("Apache-2.0".to_string()),
            notes: Some("pending license record for synthetic artifacts".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactId(String);

impl ArtifactId {
    pub fn new(
        kind: &str,
        public_proxy_id: &str,
        hash: &str,
        version: &str,
    ) -> Result<Self, CoreError> {
        ensure_slug(kind, "kind")?;
        ensure_slug(public_proxy_id, "public_proxy_id")?;
        ensure_non_empty(hash, "hash")?;
        ensure_non_empty(version, "version")?;
        Ok(Self(format!(
            "ef:{kind}:{public_proxy_id}:{hash}:{version}"
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub field: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

pub fn deterministic_artifact_id(
    kind: &str,
    public_proxy_id: &str,
    hash: &str,
    version: &str,
) -> String {
    format!("ef:{kind}:{public_proxy_id}:{hash}:{version}")
}

pub fn validate_required_fields(
    value: &serde_json::Value,
    _schema_ref: &str,
    _tier: ValidationTier,
    required_fields: &[&str],
) -> ValidationReport {
    let mut issues = Vec::new();
    for field in required_fields {
        if value.get(field).is_none() {
            issues.push(ValidationIssue {
                field: (*field).to_string(),
                message: "missing required field".to_string(),
            });
        }
    }

    for field in ["provenance", "license", "seed"] {
        if value.get(field).is_none() {
            issues.push(ValidationIssue {
                field: field.to_string(),
                message: "missing required field".to_string(),
            });
        }
    }

    ValidationReport { issues }
}

pub fn validate_metadata(
    provenance: &ProvenanceRecord,
    license: &LicenseRecord,
    _tier: ValidationTier,
) -> ValidationReport {
    let mut issues = Vec::new();
    if provenance.generated_by.trim().is_empty() {
        issues.push(ValidationIssue {
            field: "provenance.generated_by".to_string(),
            message: "missing generated_by".to_string(),
        });
    }
    if license.expression.trim().is_empty() {
        issues.push(ValidationIssue {
            field: "license.expression".to_string(),
            message: "missing license expression".to_string(),
        });
    }
    ValidationReport { issues }
}
