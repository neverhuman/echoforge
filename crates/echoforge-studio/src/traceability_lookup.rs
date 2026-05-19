use super::*;

// ---------------------------------------------------------------------------
// Campaign artifacts (typed, minimal — we only deserialize what we need so
// we are tolerant to upstream schema additions).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub(super) struct CampaignManifestMin {
    pub(super) campaign_request_id: String,
    pub(super) root_seed: u64,
    pub(super) records: Vec<CampaignRecordMin>,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct CampaignRecordMin {
    pub(super) record_id: String,
    pub(super) class_id: String,
    pub(super) target_family: String,
    pub(super) truth_metadata_path: String,
    pub(super) detector_events_path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // `is_shahed_public_proxy` is round-tripped from
                    // truth_metadata.json; reserved for future positive-class
                    // enrichment in the trace response.
pub(super) struct TruthMetadataMin {
    #[serde(default)]
    pub(super) neutral_object_id: String,
    #[serde(default)]
    pub(super) target_family: String,
    #[serde(default)]
    pub(super) is_shahed_public_proxy: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct DetectorEvent {
    pub(super) model_id: String,
    pub(super) frame_index: usize,
    pub(super) time_s: f64,
    pub(super) confidence: f64,
    pub(super) threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct DatasetCardMin {
    pub(super) id: String,
    #[serde(default)]
    pub(super) splits: BTreeMap<String, u64>,
}

// ---------------------------------------------------------------------------
// Lookup core
// ---------------------------------------------------------------------------

pub fn lookup_detection(
    detection_id: &str,
    config: &TraceabilityConfig,
) -> Result<TraceabilityResponse, TraceabilityError> {
    let parsed = parse_detection_id(detection_id)?;
    if config.campaign_roots.is_empty() {
        return Err(TraceabilityError::NotFound(format!(
            "no campaign roots configured; cannot resolve {}",
            parsed.record_id
        )));
    }

    for root in &config.campaign_roots {
        match try_lookup_in_root(detection_id, &parsed, root, config) {
            Ok(response) => return Ok(response),
            Err(TraceabilityError::NotFound(_)) | Err(TraceabilityError::MissingArtifact(_)) => {
                continue;
            }
            Err(other) => return Err(other),
        }
    }

    Err(TraceabilityError::NotFound(format!(
        "no campaign root contained record {}",
        parsed.record_id
    )))
}

fn try_lookup_in_root(
    detection_id: &str,
    parsed: &ParsedDetectionId,
    root: &Path,
    config: &TraceabilityConfig,
) -> Result<TraceabilityResponse, TraceabilityError> {
    let manifest_path = root.join("campaign_manifest.json");
    if !manifest_path.is_file() {
        return Err(TraceabilityError::MissingArtifact(format!(
            "{}",
            manifest_path.display()
        )));
    }
    let manifest: CampaignManifestMin = read_json(&manifest_path)?;
    let record = match manifest
        .records
        .iter()
        .find(|r| r.record_id == parsed.record_id)
    {
        Some(r) => r,
        None => return Err(TraceabilityError::NotFound(parsed.record_id.clone())),
    };

    let events_path = root.join(&record.detector_events_path);
    let events: Vec<DetectorEvent> = if events_path.is_file() {
        read_json(&events_path)?
    } else {
        Vec::new()
    };
    let event = match events
        .into_iter()
        .find(|ev| ev.model_id == parsed.model_id && ev.frame_index == parsed.frame_index)
    {
        Some(ev) => ev,
        None => {
            return Err(TraceabilityError::DetectionAbsent(format!(
                "no detector event for ({}, frame {}) in {}",
                parsed.model_id, parsed.frame_index, record.record_id
            )))
        }
    };

    let truth_path = root.join(&record.truth_metadata_path);
    let truth: TruthMetadataMin = if truth_path.is_file() {
        read_json(&truth_path)?
    } else {
        TruthMetadataMin {
            neutral_object_id: record.class_id.clone(),
            target_family: record.target_family.clone(),
            is_shahed_public_proxy: false,
        }
    };

    let dataset_card_path = root.join("dataset_card.json");
    let dataset: DatasetCardMin = if dataset_card_path.is_file() {
        read_json(&dataset_card_path)?
    } else {
        return Err(TraceabilityError::MissingArtifact(format!(
            "{}",
            dataset_card_path.display()
        )));
    };

    let runtime_report_path = root.join("runtime_report.json");
    let produced_at = match read_produced_at(&runtime_report_path) {
        Some(v) => v,
        None => match read_manifest_generated_at(&manifest_path) {
            Some(v) => v,
            None => "unknown".to_string(),
        },
    };

    let target = resolve_target(
        &record.class_id,
        &truth,
        config.object_packs_root.as_deref(),
    )?;
    let material = resolve_material(&truth, &record.class_id);
    let sensor = resolve_sensor();

    Ok(TraceabilityResponse {
        detection_id: detection_id.to_string(),
        record_id: parsed.record_id.clone(),
        scenario: ScenarioRef {
            seed: manifest.root_seed,
            preset: manifest.campaign_request_id.clone(),
        },
        target,
        material,
        sensor,
        dataset: DatasetRef {
            dataset_card_id: dataset.id,
            splits: dataset.splits,
        },
        detection: DetectionRef {
            model_id: event.model_id,
            frame_index: event.frame_index,
            time_s: event.time_s,
            confidence: event.confidence,
            threshold: event.threshold,
        },
        produced_at,
        limitation: PUBLIC_PROXY_LIMITATION.to_string(),
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, TraceabilityError> {
    let raw = fs::read_to_string(path)
        .map_err(|e| TraceabilityError::Io(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&raw)
        .map_err(|e| TraceabilityError::Parse(format!("{}: {e}", path.display())))
}

fn read_produced_at(runtime_report: &Path) -> Option<String> {
    if !runtime_report.is_file() {
        return None;
    }
    let raw = fs::read_to_string(runtime_report).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("generated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn read_manifest_generated_at(manifest_path: &Path) -> Option<String> {
    let raw = fs::read_to_string(manifest_path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    value
        .get("generated_at")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn resolve_target(
    class_id: &str,
    truth: &TruthMetadataMin,
    object_packs_root: Option<&Path>,
) -> Result<TargetRef, TraceabilityError> {
    if let Some(root) = object_packs_root {
        let candidates = [
            root.join(format!("{class_id}.yaml")),
            root.join("object_card.yaml"),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                if let Some(card) = read_object_card_yaml(&candidate) {
                    return Ok(TargetRef {
                        object_card_id: if card.id.is_empty() {
                            class_id.to_string()
                        } else {
                            card.id
                        },
                        public_proxy_id: card.public_proxy_id,
                        display_name: card.display_name,
                        object_family: Some(card.object_family),
                        geometry_variant: Some(card.geometry_variant),
                    });
                }
            }
        }
    }
    let neutral = if truth.neutral_object_id.is_empty() {
        class_id.to_string()
    } else {
        truth.neutral_object_id.clone()
    };
    let display_name = humanize_class_id(&neutral);
    Ok(TargetRef {
        object_card_id: format!("ef:object_card:{neutral}:derived:1"),
        public_proxy_id: neutral.clone(),
        display_name,
        object_family: Some(if truth.target_family.is_empty() {
            "unknown".to_string()
        } else {
            truth.target_family.clone()
        }),
        geometry_variant: None,
    })
}

fn read_object_card_yaml(path: &Path) -> Option<ObjectCard> {
    let raw = fs::read_to_string(path).ok()?;
    serde_yaml::from_str::<ObjectCard>(&raw).ok()
}

fn resolve_material(truth: &TruthMetadataMin, class_id: &str) -> MaterialRef {
    let family = if truth.target_family.is_empty() {
        "composite_proxy".to_string()
    } else {
        format!("{}_material_proxy", truth.target_family)
    };
    MaterialRef {
        material_card_id: format!("ef:material_card:{class_id}:derived:1"),
        material_family: family,
    }
}

fn resolve_sensor() -> SensorRef {
    SensorRef {
        sensor_archetype_id: DEFAULT_SENSOR_ARCHETYPE_ID.to_string(),
        band_name: DEFAULT_SENSOR_BAND.to_string(),
    }
}

fn humanize_class_id(class_id: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for ch in class_id.chars() {
        if ch == '-' || ch == '_' {
            out.push(' ');
            capitalize = true;
        } else if capitalize {
            out.extend(ch.to_uppercase());
            capitalize = false;
        } else {
            out.push(ch);
        }
    }
    out
}
