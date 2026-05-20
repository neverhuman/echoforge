import { useState } from 'react';
import App from './App';
import DatasetExportPanel from './radar/DatasetExportPanel';
import DetectorComparisonPanel from './radar/DetectorComparisonPanel';
import JobsPanel from './radar/JobsPanel';
import RadarConsole from './radar/RadarConsole';
import RunsPanel from './radar/RunsPanel';
import ScenarioControlPanel from './radar/ScenarioControlPanel';

type Tab = 'live' | 'scenario' | 'jobs' | 'runs' | 'detectors' | 'export' | 'contracts';

const TABS: Array<[Tab, string]> = [
  ['live', 'Live Console'],
  ['scenario', 'Scenario Builder'],
  ['jobs', 'Jobs'],
  ['runs', 'Runs'],
  ['detectors', 'Detectors'],
  ['export', 'Dataset Export'],
  ['contracts', 'Contracts'],
];

/** Top-level tab host: the realtime radar console plus the existing
 *  contract-surface view. */
export default function AppShell() {
  const [tab, setTab] = useState<Tab>('live');

  return (
    <div className="appshell">
      <header className="appshell__bar">
        <span className="appshell__brand">
          EchoForge <span className="appshell__brand-sub">Radar Console</span>
        </span>
        <nav className="appshell__tabs" aria-label="views">
          {TABS.map(([id, label]) => (
            <button
              type="button"
              key={id}
              className={tab === id ? 'is-active' : ''}
              onClick={() => setTab(id)}
              data-testid={id === 'live' ? 'tab-radar' : `tab-${id}`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="appshell__body">
        {tab === 'live' ? <RadarConsole /> : null}
        {tab === 'scenario' ? (
          <div className="studio-pad">
            <ScenarioControlPanel />
          </div>
        ) : null}
        {tab === 'jobs' ? <JobsPanel /> : null}
        {tab === 'runs' ? <RunsPanel /> : null}
        {tab === 'detectors' ? <DetectorComparisonPanel /> : null}
        {tab === 'export' ? <DatasetExportPanel /> : null}
        {tab === 'contracts' ? <App /> : null}
      </main>
    </div>
  );
}
