pub mod analytic;
pub mod artifact;
pub mod bundle;
pub mod dynamic;
pub mod error;
pub mod tensor;

// New EchoSig bundle I/O surface (Packet 5).
pub use bundle::{CardKind, EchosigBundle, EchosigBundleReader, EchosigBundleWriter, TensorDecl};
pub use echoforge_core::EchosigManifest;
pub use error::{SigError, SigResult};
pub use tensor::Dtype;

// Prior surface retained so existing radar_chain tests and downstream
// callers compile. Replaced incrementally as packets land.
pub use analytic::{
    pending_analytic_report, AnalyticPrimitive, AnalyticValidationCase,
    AnalyticValidationReport, ValidationStatus,
};
pub use artifact::{
    default_axes, AxisDescriptor, BundleCard, EchoSigArtifactBundle, EchoSigManifest,
    LicenseRecord, ProvenanceRecord, ValidationTier,
};
pub use error::{EchoSigError, Result};
