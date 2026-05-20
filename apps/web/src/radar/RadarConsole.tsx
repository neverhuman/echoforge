import { useEffect } from 'react';
import ConnectionBanner from './ConnectionBanner';
import MicroDopplerWaterfall from './MicroDopplerWaterfall';
import PpiScope from './PpiScope';
import { RadarSocket } from './radarSocket';
import RangeDopplerMap from './RangeDopplerMap';
import ScenarioControlPanel from './ScenarioControlPanel';
import TelemetryPanel from './TelemetryPanel';
import TracksPanel from './TracksPanel';
import './radar.css';

/** The realtime radar-operator console — four live displays fed by the
 *  `/ws/radar` WebSocket stream. */
export default function RadarConsole() {
  useEffect(() => {
    const socket = new RadarSocket();
    socket.connect();
    return () => socket.close();
  }, []);

  return (
    <div className="radar-console" data-testid="radar-console">
      <ConnectionBanner />
      <div className="radar-grid">
        <div className="radar-cell">
          <div className="radar-cell__title">Plan-Position Indicator</div>
          <div className="radar-cell__body">
            <PpiScope />
          </div>
        </div>
        <div className="radar-cell">
          <div className="radar-cell__title">Range · Doppler</div>
          <div className="radar-cell__body">
            <RangeDopplerMap />
          </div>
        </div>
        <div className="radar-cell">
          <div className="radar-cell__title">Micro-Doppler Spectrogram</div>
          <div className="radar-cell__body">
            <MicroDopplerWaterfall />
          </div>
        </div>
        <div className="radar-side">
          <ScenarioControlPanel />
          <TelemetryPanel />
          <TracksPanel />
        </div>
      </div>
    </div>
  );
}
