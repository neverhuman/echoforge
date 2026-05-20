import { radarStore } from './radarStore';
import RadarEmptyState from './RadarEmptyState';
import { useRadarSnapshot } from './useRadarStore';

function statRow(label: string, value: string) {
  return (
    <div className="radar-stat" key={label}>
      <span className="radar-stat__label">{label}</span>
      <span className="radar-stat__value">{value}</span>
    </div>
  );
}

/** Live link-budget and per-frame diagnostics. */
export default function TelemetryPanel() {
  const { session } = useRadarSnapshot();
  const scan = radarStore.latestScan;

  return (
    <section className="radar-panel" data-testid="telemetry-panel">
      <h3 className="radar-panel__title">Link budget &amp; telemetry</h3>
      {!scan ? (
        <RadarEmptyState message="Awaiting telemetry" />
      ) : (
        <div className="radar-stat-grid">
          {statRow('Target SNR', `${scan.meta.telemetry.snr_db.toFixed(1)} dB`)}
          {statRow(
            'Received power',
            `${scan.meta.telemetry.received_power_dbw.toFixed(1)} dBW`,
          )}
          {statRow(
            'Noise power',
            `${scan.meta.telemetry.noise_power_dbw.toFixed(1)} dBW`,
          )}
          {statRow(
            'Free-space loss',
            `${scan.meta.telemetry.free_space_path_loss_db.toFixed(1)} dB`,
          )}
          {statRow(
            'Atmospheric loss',
            `${scan.meta.telemetry.atmospheric_loss_db.toFixed(2)} dB`,
          )}
          {statRow('Rain loss', `${scan.meta.telemetry.rain_loss_db.toFixed(2)} dB`)}
          {statRow(
            'Integration gain',
            `${scan.meta.telemetry.coherent_integration_gain_db.toFixed(1)} dB`,
          )}
          {statRow(
            'Above horizon',
            scan.meta.telemetry.above_horizon ? 'yes' : 'no',
          )}
          {statRow('Detections', String(scan.meta.telemetry.detections_this_frame))}
          {statRow('Scan rate', `${scan.meta.telemetry.scan_rate_hz.toFixed(1)} Hz`)}
          {statRow(
            'Frame compute',
            `${scan.meta.telemetry.frame_compute_ms.toFixed(2)} ms`,
          )}
          {statRow('Sim time', `${scan.meta.sim_time_s.toFixed(1)} s`)}
          {statRow(
            'Range window',
            `${((session?.range_max_m ?? 0) / 1000).toFixed(1)} km`,
          )}
          {statRow(
            'Doppler window',
            `±${(session?.doppler_max_hz ?? 0).toFixed(0)} Hz`,
          )}
        </div>
      )}
    </section>
  );
}
