use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::CoreError;
use crate::validation::ID_VERSION;

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut ordered = BTreeMap::new();
            for (key, nested) in map {
                ordered.insert(key, canonicalize(nested));
            }
            Value::Object(ordered.into_iter().collect())
        }
        Value::Array(items) => Value::Array(items.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<String, CoreError> {
    let value = serde_json::to_value(value)?;
    Ok(serde_json::to_string(&canonicalize(value))?)
}

pub fn fingerprint_sha256<T: Serialize>(value: &T) -> Result<String, CoreError> {
    let canonical = canonical_json(value)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

pub fn deterministic_id<T: Serialize>(
    kind: &str,
    public_proxy_id: &str,
    value: &T,
) -> Result<String, CoreError> {
    let fingerprint = fingerprint_sha256(value)?;
    let mut hasher = Sha256::new();
    hasher.update(kind.as_bytes());
    hasher.update(b"|");
    hasher.update(public_proxy_id.as_bytes());
    hasher.update(b"|");
    hasher.update(ID_VERSION.to_string().as_bytes());
    hasher.update(b"|");
    hasher.update(fingerprint.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    Ok(format!(
        "ef:{kind}:{public_proxy_id}:{}:{ID_VERSION}",
        &digest[..16]
    ))
}
