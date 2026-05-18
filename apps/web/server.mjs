import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? '0.0.0.0';
const apiBaseUrl = process.env.ECHOFORGE_API_BASE_URL ?? 'http://localhost:8080';

const __dirname = dirname(fileURLToPath(import.meta.url));
const indexHtml = readFileSync(join(__dirname, 'index.html'), 'utf8').replaceAll(
  '__ECHOFORGE_API_BASE_URL__',
  apiBaseUrl,
);

const assetMap = new Map(
  [
    ['/', ['text/html; charset=utf-8', indexHtml]],
    ['/index.html', ['text/html; charset=utf-8', indexHtml]],
    ['/src/main.mjs', ['text/javascript; charset=utf-8', readFileSync(join(__dirname, 'src/main.mjs'), 'utf8')]],
    ['/src/styles.css', ['text/css; charset=utf-8', readFileSync(join(__dirname, 'src/styles.css'), 'utf8')]],
    [
      '/src/generated/contracts-client.generated.mjs',
      [
        'text/javascript; charset=utf-8',
        readFileSync(join(__dirname, 'src/generated/contracts-client.generated.mjs'), 'utf8'),
      ],
    ],
  ].map(([path, value]) => [path, value]),
);

function send(res, statusCode, contentType, body) {
  res.writeHead(statusCode, {
    'Content-Type': contentType,
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

createServer((req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);

  if (req.method === 'GET' && url.pathname === '/healthz') {
    send(
      res,
      200,
      'application/json; charset=utf-8',
      `${JSON.stringify({
        service: 'echoforge-web',
        status: 'ok',
        mode: 'placeholder',
        apiBaseUrl,
      }, null, 2)}\n`,
    );
    return;
  }

  const asset = assetMap.get(url.pathname);
  if (req.method === 'GET' && asset) {
    send(res, 200, asset[0], asset[1]);
    return;
  }

  send(
    res,
    404,
    'application/json; charset=utf-8',
    `${JSON.stringify({
      service: 'echoforge-web',
      status: 'placeholder',
      error: 'not_found',
      path: url.pathname,
    }, null, 2)}\n`,
  );
}).listen(port, host, () => {
  process.stdout.write(
    `${JSON.stringify({
      service: 'echoforge-web',
      status: 'listening',
      host,
      port,
      apiBaseUrl,
      mode: 'placeholder',
    })}\n`,
  );
});

