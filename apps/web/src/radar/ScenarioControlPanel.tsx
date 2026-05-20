import { useCallback, useState } from 'react';
import {
  pauseSim,
  resumeSim,
  setRadarParams,
  setPlaybackSpeed,
  startSim,
  stopSim,
} from './radarControl';
import { useRadarSnapshot } from './useRadarStore';

/** Scenario picker, transport controls, and playback-speed slider. */
export default function ScenarioControlPanel() {
  const { session, status } = useRadarSnapshot();
  const [busy, setBusy] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [power, setPower] = useState(250000);
  const [rain, setRain] = useState(2);

  const scenarios = session?.available_scenarios ?? [];
  const running = session?.running ?? false;
  const paused = session?.paused ?? false;
  const scenarioId = session?.scenario_id ?? scenarios[0]?.id ?? '';

  const run = useCallback((action: () => Promise<unknown>) => {
    setBusy(true);
    action()
      .catch(() => undefined)
      .finally(() => setBusy(false));
  }, []);

  const onScenarioChange = useCallback(
    (event: React.ChangeEvent<HTMLSelectElement>) => {
      run(() => startSim({ scenarioId: event.target.value, mode: 'live' }));
    },
    [run],
  );

  const onSpeedChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number(event.target.value);
      setSpeed(value);
      setPlaybackSpeed(value).catch(() => undefined);
    },
    [],
  );

  const onPowerChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    setPower(value);
    setRadarParams({ transmit_power_w: value }).catch(() => undefined);
  }, []);

  const onRainChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    setRain(value);
    setRadarParams({ rain_rate_mm_per_h: value }).catch(() => undefined);
  }, []);

  return (
    <section className="radar-panel" data-testid="scenario-control">
      <h3 className="radar-panel__title">Scenario control</h3>

      <label className="radar-field">
        <span>Scenario</span>
        <select
          value={scenarioId}
          onChange={onScenarioChange}
          disabled={busy}
          data-testid="scenario-select"
        >
          {scenarios.length === 0 ? (
            <option value="">loading…</option>
          ) : (
            scenarios.map((scenario) => (
              <option key={scenario.id} value={scenario.id}>
                {scenario.label}
              </option>
            ))
          )}
        </select>
      </label>

      <div className="radar-control-row">
        {running ? (
          <button
            type="button"
            className="radar-btn radar-btn--stop"
            onClick={() => run(stopSim)}
            disabled={busy}
            data-testid="sim-stop"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="radar-btn radar-btn--go"
            onClick={() => run(() => startSim({ scenarioId, mode: 'live' }))}
            disabled={busy}
            data-testid="sim-start"
          >
            Start live
          </button>
        )}
        {running &&
          (paused ? (
            <button
              type="button"
              className="radar-btn"
              onClick={() => run(resumeSim)}
              disabled={busy}
              data-testid="sim-resume"
            >
              Resume
            </button>
          ) : (
            <button
              type="button"
              className="radar-btn"
              onClick={() => run(pauseSim)}
              disabled={busy}
              data-testid="sim-pause"
            >
              Pause
            </button>
          ))}
        <button
          type="button"
          className="radar-btn"
          onClick={() => run(() => startSim({ scenarioId, mode: 'replay' }))}
          disabled={busy}
          data-testid="sim-replay"
        >
          Replay bundle
        </button>
      </div>

      <label className="radar-field">
        <span>Playback speed — {speed.toFixed(2)}×</span>
        <input
          type="range"
          min="0.25"
          max="4"
          step="0.25"
          value={speed}
          onChange={onSpeedChange}
          data-testid="speed-slider"
        />
      </label>

      <div className="studio-segment" aria-label="run mode">
        <button type="button" className="is-active">
          Live
        </button>
        <button type="button">Replay</button>
        <button type="button">Campaign</button>
      </div>

      <label className="radar-field">
        <span>Transmit power — {Math.round(power / 1000)} kW</span>
        <input
          type="range"
          min="50000"
          max="1000000"
          step="50000"
          value={power}
          onChange={onPowerChange}
          data-testid="power-slider"
        />
      </label>

      <label className="radar-field">
        <span>Rain attenuation proxy — {rain.toFixed(1)} mm/h</span>
        <input
          type="range"
          min="0"
          max="25"
          step="0.5"
          value={rain}
          onChange={onRainChange}
          data-testid="rain-slider"
        />
      </label>

      <p className="radar-control__status">
        Source: <strong>{session?.source ?? 'idle'}</strong>
        {status ? ` · ${status.code}: ${status.message}` : ''}
      </p>
    </section>
  );
}
