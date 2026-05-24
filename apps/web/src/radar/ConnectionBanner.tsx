import { useRadarSnapshot } from './useRadarStore';
import type { ConnectionState } from './radarStore';

/** Named connection-state banner — hidden while the stream is healthy. */
export default function ConnectionBanner() {
  const { connection } = useRadarSnapshot();
  const banners: Record<ConnectionState, { hidden: boolean; label: string }> = {
    connecting: {
      hidden: false,
      label: 'Connecting to the radar service...',
    },
    open: {
      hidden: true,
      label: 'Radar service connected',
    },
    reconnecting: {
      hidden: false,
      label: 'Connection lost; reconnecting...',
    },
    closed: {
      hidden: false,
      label: 'Disconnected from the radar service',
    },
  };
  const banner = banners[connection];
  return (
    <div
      className={`radar-banner radar-banner--${connection}`}
      data-testid="connection-banner"
      hidden={banner.hidden}
      role="status"
    >
      {banner.label}
    </div>
  );
}
