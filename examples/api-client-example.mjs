import { createEchoForgeClient } from '../apps/web/src/generated/contracts-client.generated.mjs';

const client = createEchoForgeClient({
  apiBaseUrl: process.env.ECHOFORGE_API_BASE_URL ?? 'http://localhost:8080',
});

const [health, catalog] = await Promise.all([client.getHealth(), client.getCatalog()]);

process.stdout.write(
  `${JSON.stringify(
    {
      health,
      firstSchema: catalog.schemas[0] ?? null,
    },
    null,
    2,
  )}\n`,
);

