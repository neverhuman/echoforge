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

fn run() -> Result<String, String> {
    let kind = env::args()
        .nth(1)
        .ok_or_else(|| "usage: emit_canonical <kind>".to_string())?;
    let value = samples::canonical_value(&kind)?;
    canonical_json(&value).map_err(|e| format!("canonical_json: {e}"))
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
