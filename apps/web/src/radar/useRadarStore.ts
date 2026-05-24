// React binding for the radar store. `useSyncExternalStore` is a React
// built-in, so this satisfies the "hooks only, no store library" rule.

import { useSyncExternalStore } from 'react';
import { radarStore, type RadarSnapshot } from './radarStore';

export function useRadarSnapshot(): RadarSnapshot {
  return useSyncExternalStore(radarStore.subscribe, radarStore.getSnapshot);
}
