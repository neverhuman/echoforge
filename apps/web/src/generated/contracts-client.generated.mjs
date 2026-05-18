function stripTrailingSlash(value) {
  return value ? value.replace(/\/+$/, '') : '';
}

function resolveBaseUrl(apiBaseUrl) {
  if (apiBaseUrl) {
    return stripTrailingSlash(apiBaseUrl);
  }

  const configBaseUrl = globalThis.__ECHOFORGE_CONFIG__?.apiBaseUrl;
  return stripTrailingSlash(configBaseUrl ?? 'http://localhost:8080');
}

async function requestJson(baseUrl, path, fetchImpl) {
  const response = await fetchImpl(`${baseUrl}${path}`, {
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Request to ${path} failed with HTTP ${response.status}`);
  }

  return response.json();
}

export function createEchoForgeClient({ apiBaseUrl, fetchImpl = globalThis.fetch.bind(globalThis) } = {}) {
  const baseUrl = resolveBaseUrl(apiBaseUrl);

  return {
    baseUrl,
    getHealth() {
      return requestJson(baseUrl, '/healthz', fetchImpl);
    },
    getCatalog() {
      return requestJson(baseUrl, '/api/catalog', fetchImpl);
    },
    getContracts() {
      return requestJson(baseUrl, '/api/contracts', fetchImpl);
    },
  };
}

