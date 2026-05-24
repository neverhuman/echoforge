export interface EchoForgeClientOptions {
  apiBaseUrl?: string;
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
}

export interface EchoForgeClient {
  baseUrl: string;
  getHealth(): Promise<unknown>;
  getCatalog(): Promise<unknown>;
  getValidationLatest(): Promise<unknown>;
  getContracts(): Promise<unknown>;
}

export function createEchoForgeClient(options?: EchoForgeClientOptions): EchoForgeClient;
