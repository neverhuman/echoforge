/**
 * meshService — thin HTTP adapter layer between view components and the studio API.
 *
 * All fetch calls targeting /api/mesh/* are centralised here. Domain types
 * (MeshDescriptor, MeshStats) are re-exported from domainTypes.ts so view
 * components can import type definitions without depending on this adapter.
 */

export type { MeshDescriptor, MeshListResponse, MeshStats } from './domainTypes';

export const MESH_LIST_URL = '/api/mesh/list';

/** Resolves the mesh-list endpoint URL. */
export function meshListUrl(): string {
  return MESH_LIST_URL;
}

/** Resolves the per-primitive STL endpoint URL. */
export function meshStlUrl(primitive: string): string {
  return `/api/mesh/${encodeURIComponent(primitive)}`;
}

import type { MeshDescriptor, MeshListResponse, MeshStats } from './domainTypes';

/**
 * Fetch and validate the list of available mesh primitives from the studio
 * service.  Throws on non-2xx responses or an empty primitive list.
 */
export async function fetchMeshList(fetcher: typeof fetch): Promise<MeshDescriptor[]> {
  const resp = await fetcher(MESH_LIST_URL, { headers: { Accept: 'application/json' } });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status} from ${MESH_LIST_URL}`);
  }
  const body = (await resp.json()) as MeshListResponse;
  if (!body.primitives || body.primitives.length === 0) {
    throw new Error('mesh list is empty');
  }
  return body.primitives;
}

export function parseBinaryStl(buffer: ArrayBuffer): MeshStats {
  if (buffer.byteLength < 84) {
    throw new Error(`STL payload too small (${buffer.byteLength} bytes)`);
  }
  const dv = new DataView(buffer);
  const triangleCount = dv.getUint32(80, true);
  const expectedLen = 84 + triangleCount * 50;
  if (buffer.byteLength !== expectedLen) {
    throw new Error(
      `STL length mismatch: header says ${triangleCount} triangles ` +
        `(want ${expectedLen} bytes), got ${buffer.byteLength}`,
    );
  }
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (let i = 0; i < triangleCount; i++) {
    const base = 84 + i * 50;
    for (let v = 0; v < 3; v++) {
      const vOff = base + 12 + v * 12;
      const x = dv.getFloat32(vOff + 0, true);
      const y = dv.getFloat32(vOff + 4, true);
      const z = dv.getFloat32(vOff + 8, true);
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
  }
  if (triangleCount === 0) {
    return {
      triangleCount,
      boundingBox: { min: [0, 0, 0], max: [0, 0, 0] },
    };
  }
  return {
    triangleCount,
    boundingBox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
  };
}

export interface MeshServiceImpl {
  fetchList(): Promise<MeshDescriptor[]>;
  fetchStl(primitiveId: string, signal: AbortSignal): Promise<{ bytes: ArrayBuffer; stats: MeshStats }>;
}

/** Default service wired to globalThis.fetch. */
export function createDefaultMeshService(): MeshServiceImpl {
  return {
    fetchList: () => fetchMeshList(globalThis.fetch.bind(globalThis)),
    fetchStl: async (id, sig) => {
      const bytes = await fetchMeshStl(globalThis.fetch.bind(globalThis), id, sig);
      const stats = parseBinaryStl(bytes);
      return { bytes, stats };
    },
  };
}

/** Build a test service that injects a custom fetch implementation. */
export function createTestMeshService(fetchImpl: typeof fetch): MeshServiceImpl {
  return {
    fetchList: () => fetchMeshList(fetchImpl),
    fetchStl: async (id, sig) => {
      const bytes = await fetchMeshStl(fetchImpl, id, sig);
      const stats = parseBinaryStl(bytes);
      return { bytes, stats };
    },
  };
}

export async function fetchMeshStl(
  fetcher: typeof fetch,
  primitiveId: string,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  const url = meshStlUrl(primitiveId);
  const resp = await fetcher(url, {
    signal,
    headers: { Accept: 'model/stl, application/octet-stream' },
  });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status} from ${url}`);
  }
  return resp.arrayBuffer();
}
