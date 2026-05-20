//! Static card writers — extracted from sampling.rs for LOC compliance.

use echoforge_core::models::{Scenario, SensorArchetype};

use crate::export::{write_json_pretty, write_text};
use crate::monte_carlo::error::DatasetError;
use crate::monte_carlo::config::MonteCarloDemoConfig;
use crate::monte_carlo::helpers::{license, midpoint, provenance, validation_info};

use super::ResolvedPreset;

pub fn write_static_cards(
    config: &MonteCarloDemoConfig,
    resolved: &ResolvedPreset<'_>,
) -> Result<(), DatasetError> {
    let sensor = SensorArchetype {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved.object.id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.42),
        sensor_name: resolved.sensor.display_name.clone(),
        band_name: resolved.sensor.band_name.clone(),
        waveform_family: "lfm_pulse_doppler_public_proxy".to_string(),
        center_frequency_hz: midpoint(resolved.sensor.center_frequency_hz),
        sample_rate_hz: config.sample_rate_hz,
    }
    .finalize()?;
    write_json_pretty(&config.output_dir.join("sensor_archetype.json"), &sensor)?;

    let scenario = Scenario {
        id: String::new(),
        kind: String::new(),
        schema_version: String::new(),
        public_proxy_id: resolved.object.id.clone(),
        provenance: provenance(config),
        license: license(),
        validation: validation_info(0.4),
        scenario_name: resolved.preset.scenario_label.clone(),
        sensor_archetype_id: sensor.id.clone(),
        object_card_ids: vec![resolved.object.id.clone()],
        environment_label: resolved.environment.display_name.clone(),
        seed: config.seed,
    }
    .finalize()?;
    write_text(
        &config.output_dir.join("object_card.yaml"),
        &format!(
            "id: {}\nkind: object_card\ndisplay_name: {}\nobject_family: {}\nsource_status: public-proxy\nlimitation: not measured truth; not proprietary-equivalent\n",
            resolved.object.id, resolved.object.display_name, resolved.object.object_family
        ),
    )?;
    write_text(
        &config.output_dir.join("material_card.yaml"),
        "id: monte-carlo-material-public-proxy-v1\nkind: material_card\nmaterial_family: mixed public-proxy composite/polymer/metal assumptions\nsource_status: public-proxy\nlimitation: statistical uncertainty proxy, not measured truth\n",
    )?;
    write_text(
        &config.output_dir.join("scenario.yaml"),
        &format!(
            "id: {}\nkind: scenario\nscenario_name: {}\nenvironment_label: {}\nsource_status: public-proxy\ncontested_airspace: robustness stressors only\n",
            scenario.id, scenario.scenario_name, scenario.environment_label
        ),
    )?;
    Ok(())
}
