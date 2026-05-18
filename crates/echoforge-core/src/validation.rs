use regex::Regex;

use crate::error::CoreError;

pub const SCHEMA_VERSION: &str = "1.0.0";
pub const ID_VERSION: u32 = 1;

pub fn ensure_non_empty(value: &str, field: &str) -> Result<(), CoreError> {
    if value.trim().is_empty() {
        return Err(CoreError::Validation(format!("{field} must be non-empty")));
    }
    Ok(())
}

pub fn ensure_non_empty_vec<T>(value: &[T], field: &str) -> Result<(), CoreError> {
    if value.is_empty() {
        return Err(CoreError::Validation(format!("{field} must not be empty")));
    }
    Ok(())
}

pub fn ensure_probability(value: f64, field: &str) -> Result<(), CoreError> {
    if !(0.0..=1.0).contains(&value) {
        return Err(CoreError::Validation(format!(
            "{field} must be between 0 and 1"
        )));
    }
    Ok(())
}

pub fn ensure_slug(value: &str, field: &str) -> Result<(), CoreError> {
    let regex = Regex::new(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$").expect("valid slug regex");
    if !regex.is_match(value) {
        return Err(CoreError::Validation(format!(
            "{field} must be a lowercase slug"
        )));
    }
    Ok(())
}
