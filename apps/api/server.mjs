import { createServer } from 'node:http';
import { URL } from 'node:url';

const port = Number(process.env.PORT ?? 8080);
const host = process.env.HOST ?? '0.0.0.0';
const publicBaseUrl = process.env.ECHOFORGE_PUBLIC_API_BASE_URL ?? `http://localhost:${port}`;

const schemaCatalog = [
  'object_card.schema.json',
  'material_card.schema.json',
  'mesh_manifest.schema.json',
  'solver_card.schema.json',
  'rcs_campaign.schema.json',
  'echosig_manifest.schema.json',
  'sensor_archetype.schema.json',
  'scenario.schema.json',
  'radar_episode.schema.json',
  'detector_graph.schema.json',
  'dataset_card.schema.json',
  'validation_report.schema.json',
];

function sendJson(res, statusCode, body) {
  const payload = `${JSON.stringify(body, null, 2)}\n`;
  res.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(payload),
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'content-type',
  });
  res.end(payload);
}

function sendText(res, statusCode, body) {
  res.writeHead(statusCode, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
    'Access-Control-Allow-Origin': '*',
  });
  res.end(body);
}

function handler(req, res) {
  const requestUrl = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'content-type',
    });
    res.end();
    return;
  }

  if (req.method !== 'GET') {
    sendJson(res, 405, {
      service: 'echoforge-api',
      status: 'placeholder',
      error: 'method_not_allowed',
    });
    return;
  }

  switch (requestUrl.pathname) {
    case '/':
      sendText(
        res,
        200,
        'EchoForge API placeholder. Use /healthz or /api/catalog for the small deployment surface.\n',
      );
      return;
    case '/healthz':
      sendJson(res, 200, {
        service: 'echoforge-api',
        status: 'ok',
        mode: 'placeholder',
        publicBaseUrl,
      });
      return;
    case '/api/catalog':
      sendJson(res, 200, {
        service: 'echoforge-api',
        status: 'placeholder',
        publicBaseUrl,
        schemas: schemaCatalog.map((schema) => ({
          schema,
          surface: 'contract-placeholder',
          ready: false,
        })),
      });
      return;
    case '/api/contracts':
      sendJson(res, 200, {
        service: 'echoforge-api',
        status: 'placeholder',
        contractClient: 'apps/web/src/generated/contracts-client.generated.mjs',
        endpoints: ['/healthz', '/api/catalog', '/api/contracts'],
      });
      return;
    default:
      sendJson(res, 404, {
        service: 'echoforge-api',
        status: 'placeholder',
        error: 'not_found',
        path: requestUrl.pathname,
      });
  }
}

createServer(handler).listen(port, host, () => {
  process.stdout.write(
    `${JSON.stringify({
      service: 'echoforge-api',
      status: 'listening',
      host,
      port,
      publicBaseUrl,
      mode: 'placeholder',
    })}\n`,
  );
});

