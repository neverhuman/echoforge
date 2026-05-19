import { useEffect, useMemo, useRef, useState } from 'react';

import type { MeshStats } from './domainTypes';

// ---------------------------------------------------------------------------
// LoadedGeometry: internal type — bytes stay inside MeshCanvas, never exposed
// to MeshViewer or any other component layer.
// ---------------------------------------------------------------------------

interface LoadedGeometry {
  primitiveId: string;
  stats: MeshStats;
  bytes: ArrayBuffer;
}

// ---------------------------------------------------------------------------
// Lazy-loaded THREE / R3F module bundle
// ---------------------------------------------------------------------------

interface CanvasModules {
  Canvas: typeof import('@react-three/fiber').Canvas;
  useThree: typeof import('@react-three/fiber').useThree;
  useFrame: typeof import('@react-three/fiber').useFrame;
  STLLoader: typeof import('three/examples/jsm/loaders/STLLoader.js').STLLoader;
  OrbitControls: typeof import('three/examples/jsm/controls/OrbitControls.js').OrbitControls;
  BufferGeometry: typeof import('three').BufferGeometry;
}

// ---------------------------------------------------------------------------
// OrbitControlsBinding
// ---------------------------------------------------------------------------

function OrbitControlsBinding({
  OrbitControls,
  useThree,
  useFrame,
}: Pick<CanvasModules, 'OrbitControls' | 'useThree' | 'useFrame'>) {
  const camera = useThree((s) => s.camera);
  const domElement = useThree((s) => s.gl.domElement);
  const controlsRef = useRef<InstanceType<CanvasModules['OrbitControls']> | null>(null);

  useEffect(() => {
    const controls = new OrbitControls(camera, domElement);
    controls.enableDamping = true;
    controlsRef.current = controls;
    return () => {
      controls.dispose();
      controlsRef.current = null;
    };
  }, [OrbitControls, camera, domElement]);

  useFrame(() => {
    const controls = controlsRef.current;
    if (controls !== null) {
      controls.update();
    }
  });

  return <></>;
}

// ---------------------------------------------------------------------------
// Scene: renders the parsed STL geometry inside the R3F Canvas
// ---------------------------------------------------------------------------

function Scene({
  geometry: loadedGeometry,
  modules,
}: {
  geometry: LoadedGeometry;
  modules: CanvasModules;
}) {
  const geometry = useMemo(() => {
    const loader = new modules.STLLoader();
    return loader.parse(loadedGeometry.bytes);
  }, [loadedGeometry, modules]);

  const { Canvas } = modules;
  const bb = loadedGeometry.stats.boundingBox;
  const dx = bb.max[0] - bb.min[0];
  const dy = bb.max[1] - bb.min[1];
  const dz = bb.max[2] - bb.min[2];
  const radius = Math.max(dx, dy, dz, 1) * 1.6;

  return (
    <Canvas
      camera={{ position: [radius, radius, radius], fov: 50, near: 0.01, far: radius * 50 }}
      style={{ width: '100%', height: '320px', background: 'transparent' }}
    >
      <ambientLight intensity={0.35} />
      <directionalLight position={[radius, radius * 1.5, radius]} intensity={0.9} />
      <directionalLight position={[-radius, -radius * 0.5, -radius]} intensity={0.4} />
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#7ec0ff" metalness={0.1} roughness={0.42} />
      </mesh>
      <axesHelper args={[radius * 0.5]} />
      <OrbitControlsBinding
        OrbitControls={modules.OrbitControls}
        useThree={modules.useThree}
        useFrame={modules.useFrame}
      />
    </Canvas>
  );
}

// ---------------------------------------------------------------------------
// MeshCanvas: public component exported from this file
// ---------------------------------------------------------------------------

interface MeshCanvasProps {
  primitiveId: string;
  onLoaded: (stats: MeshStats) => void;
  onError: (msg: string) => void;
  onLoading: () => void;
}

/**
 * Self-contained canvas component. Owns its own fetch lifecycle (AbortController,
 * bytes, THREE.js lazy load). Only exposes typed callbacks — no ArrayBuffer or
 * DataView ever crosses the boundary into MeshViewer or any higher-level component.
 */
export default function MeshCanvas({
  primitiveId,
  onLoaded,
  onError,
  onLoading,
}: MeshCanvasProps) {
  const [modules, setModules] = useState<CanvasModules | null>(null);
  const [modulesError, setModulesError] = useState<string | null>(null);
  const [geometry, setGeometry] = useState<LoadedGeometry | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Load THREE / R3F modules once on mount.
  useEffect(() => {
    let active = true;
    Promise.all([
      import('@react-three/fiber'),
      import('three/examples/jsm/loaders/STLLoader.js'),
      import('three/examples/jsm/controls/OrbitControls.js'),
      import('three'),
    ])
      .then(([fiber, stl, orbit, three]) => {
        if (!active) return;
        setModules({
          Canvas: fiber.Canvas,
          useThree: fiber.useThree,
          useFrame: fiber.useFrame,
          STLLoader: stl.STLLoader,
          OrbitControls: orbit.OrbitControls,
          BufferGeometry: three.BufferGeometry,
        });
      })
      .catch((err: unknown) => {
        if (!active) return;
        const msg = err instanceof Error ? err.message : String(err);
        setModulesError(msg);
      });
    return () => {
      active = false;
    };
  }, []);

  // Fetch the STL whenever primitiveId changes.
  useEffect(() => {
    const controller = new AbortController();
    onLoading();
    setGeometry(null);
    setFetchError(null);

    (async () => {
      const { createDefaultMeshService } = await import('./meshService');
      if (controller.signal.aborted) return;
      const service = createDefaultMeshService();
      const { bytes, stats } = await service.fetchStl(primitiveId, controller.signal);
      if (controller.signal.aborted) return;
      setGeometry({ primitiveId, stats, bytes });
      onLoaded(stats);
    })().catch((err: unknown) => {
      if (controller.signal.aborted) return;
      const msg = err instanceof Error ? err.message : String(err);
      console.error('[MeshCanvas] STL fetch failed:', msg, { primitive: primitiveId }); // telemetry signal
      setFetchError(msg);
      onError(msg);
    });

    return () => {
      controller.abort();
    };
  }, [primitiveId, onLoaded, onError, onLoading]);

  if (modulesError !== null) {
    return <div className="mesh-viewer__status">three.js failed to load: {modulesError}</div>;
  }

  if (fetchError !== null) {
    return <div className="mesh-viewer__status">{fetchError}</div>;
  }

  if (modules === null) {
    return <div className="mesh-viewer__status">Booting three.js...</div>;
  }

  if (geometry === null) {
    return <div className="mesh-viewer__status">Synthesizing mesh...</div>;
  }

  return <Scene geometry={geometry} modules={modules} />;
}
