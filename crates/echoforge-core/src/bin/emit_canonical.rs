// Emit a canonical JSON serialization of a finalized minimal instance for
// a given EchoForge document kind. Used by `tools/canonical_diff.mjs` to
// pin Rust <-> schema byte identity.
//
// Usage: emit_canonical <kind>
// Exit:  0 success, 1 error.

use std::env;
use std::process::ExitCode;

use echoforge_core::canonical_json;

// Share builders with the integration tests so the on-the-wire payload is
// produced by the same code path the tests assert against.
#[path = "../../tests/common/samples.rs"]
mod samples;

#[derive(Debug)]
enum EmitError {
    Usage,
    UnknownKind(String),
    Serialize(String),
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage => write!(f, "usage: emit_canonical <kind>"),
            Self::UnknownKind(e) => write!(f, "{e}"),
            Self::Serialize(e) => write!(f, "canonical_json: {e}"),
        }
    }
}

fn run() -> Result<String, EmitError> {
    let Some(kind) = env::args().nth(1) else {
        return Err(EmitError::Usage);
    };
    let value = samples::canonical_value(&kind).map_err(EmitError::UnknownKind)?;
    canonical_json(&value).map_err(|e| EmitError::Serialize(e.to_string()))
}

fn main() -> ExitCode {
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("emit_canonical: {err}");
            ExitCode::FAILURE
        }
    }
}
