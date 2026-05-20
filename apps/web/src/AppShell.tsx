import { useState } from 'react';
import App from './App';
import RadarConsole from './radar/RadarConsole';

type Tab = 'radar' | 'contracts';

/** Top-level tab host: the realtime radar console plus the existing
 *  contract-surface view. */
export default function AppShell() {
  const [tab, setTab] = useState<Tab>('radar');

  return (
    <div className="appshell">
      <header className="appshell__bar">
        <span className="appshell__brand">
          EchoForge <span className="appshell__brand-sub">Radar Console</span>
        </span>
        <nav className="appshell__tabs" aria-label="views">
          <button
            type="button"
            className={tab === 'radar' ? 'is-active' : ''}
            onClick={() => setTab('radar')}
            data-testid="tab-radar"
          >
            Radar Console
          </button>
          <button
            type="button"
            className={tab === 'contracts' ? 'is-active' : ''}
            onClick={() => setTab('contracts')}
            data-testid="tab-contracts"
          >
            Contracts
          </button>
        </nav>
      </header>
      <main className="appshell__body">
        {tab === 'radar' ? <RadarConsole /> : <App />}
      </main>
    </div>
  );
}
