import { useRadarSnapshot } from './useRadarStore';

/** Named connection-state banner — hidden while the stream is healthy. */
export default function ConnectionBanner() {
  const { connection } = useRadarSnapshot();
  if (connection === 'open') {
    return null;
  }
  const label =
    connection === 'connecting'
      ? 'Connecting to the radar service…'
      : connection === 'reconnecting'
        ? 'Connection lost — reconnecting…'
        : 'Disconnected from the radar service';
  return (
    <div
      className={`radar-banner radar-banner--${connection}`}
      data-testid="connection-banner"
      role="status"
    >
      {label}
    </div>
  );
}
