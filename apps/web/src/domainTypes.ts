/** Shared domain types for the mesh viewer surface — no runtime imports. */

export interface MeshDescriptor {
  primitive_id: string;
  display_name: string;
  defaults: Record<string, number>;
}

export interface MeshListResponse {
  primitives: MeshDescriptor[];
}

export interface MeshStats {
  triangleCount: number;
  boundingBox: {
    min: [number, number, number];
    max: [number, number, number];
  };
}
