import { radarStore } from './radarStore';
import RadarEmptyState from './RadarEmptyState';
import { useRadarSnapshot } from './useRadarStore';

/** Operator track table — selectable rows cross-highlight the PPI. */
export default function TracksPanel() {
  const { selectedTrackId } = useRadarSnapshot();
  const tracks = radarStore.latestScan?.meta.tracks ?? [];

  return (
    <section className="radar-panel" data-testid="tracks-panel">
      <h3 className="radar-panel__title">
        Tracks <span className="radar-panel__count">{tracks.length}</span>
      </h3>
      {tracks.length === 0 ? (
        <RadarEmptyState message="No active tracks" />
      ) : (
        <table className="radar-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Range</th>
              <th>Radial vel</th>
              <th>SNR</th>
              <th>Class</th>
              <th>Conf</th>
              <th>Age</th>
            </tr>
          </thead>
          <tbody>
            {tracks.map((track) => (
              <tr
                key={track.track_id}
                data-testid={`track-row-${track.track_id}`}
                className={track.track_id === selectedTrackId ? 'is-selected' : ''}
                onClick={() =>
                  radarStore.setSelectedTrack(
                    track.track_id === selectedTrackId ? null : track.track_id,
                  )
                }
              >
                <td>{track.track_id}</td>
                <td>{(track.range_m / 1000).toFixed(2)} km</td>
                <td>{track.radial_velocity_mps.toFixed(1)} m/s</td>
                <td>{track.snr_db.toFixed(1)} dB</td>
                <td>{track.class_label}</td>
                <td>{track.confidence.toFixed(2)}</td>
                <td>{track.age_frames}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
