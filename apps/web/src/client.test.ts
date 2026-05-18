import { expect, test } from 'vitest';
import { createEchoForgeClient } from './generated/contracts-client.generated.mjs';

test('echo forge client defaults to same-origin paths', async () => {
  const requests: Array<{ url: string; options: RequestInit | undefined }> = [];
  const fetchImpl = async (url: string, options?: RequestInit) => {
    requests.push({ url, options });
    return {
      ok: true,
      status: 200,
      json: async () => ({ ok: true, url }),
    } as Response;
  };

  const client = createEchoForgeClient({ fetchImpl });
  expect(client.baseUrl).toBe('');

  await client.getHealth();
  await client.getCatalog();
  await client.getValidationLatest();
  await client.getContracts();

  expect(requests.map((request) => request.url)).toEqual([
    '/healthz',
    '/api/catalog',
    '/api/validation/latest',
    '/api/contracts',
  ]);
  expect(requests[0]?.options?.headers).toEqual({ Accept: 'application/json' });
});

test('echo forge client trims explicit base URLs', async () => {
  const fetchImpl = async () =>
    ({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
    }) as Response;

  const client = createEchoForgeClient({
    apiBaseUrl: 'http://example.test///',
    fetchImpl,
  });

  expect(client.baseUrl).toBe('http://example.test');
});
