import { createEchoForgeClient } from './generated/contracts-client.generated.mjs';

const root = document.querySelector('#app');
const config = globalThis.__ECHOFORGE_CONFIG__ ?? {};
const client = createEchoForgeClient({ apiBaseUrl: config.apiBaseUrl });

function setState(title, body) {
  root.innerHTML = `
    <article class="status-card">
      <h2>${title}</h2>
      <pre>${body}</pre>
    </article>
  `;
}

async function boot() {
  try {
    const [health, catalog] = await Promise.all([client.getHealth(), client.getCatalog()]);
    const schemaLines = catalog.schemas.map((schema) => `- ${schema.schema}: ${schema.surface}`).join('\n');
    setState(
      'Contract surface loaded',
      [
        `API status: ${health.status}`,
        `API mode: ${health.mode}`,
        `API base URL: ${health.publicBaseUrl}`,
        '',
        'Schemas:',
        schemaLines,
        '',
        'This surface is intentionally small and does not claim completed behavior.',
      ].join('\n'),
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setState('Surface unavailable', `The placeholder client could not reach the API.\n\n${message}`);
  }
}

boot();

