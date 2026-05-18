import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';

export function buildGpuDoctorReport() {
  const deviceNodes = ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvidia-uvm'].filter((path) => existsSync(path));

  return {
    kind: 'echoforge.gpu_doctor',
    status: 'placeholder',
    mode: 'scaffold',
    timestamp: new Date().toISOString(),
    environment: {
      nvidiaVisibleDevices: process.env.NVIDIA_VISIBLE_DEVICES ?? 'unset',
      nvidiaDriverCapabilities: process.env.NVIDIA_DRIVER_CAPABILITIES ?? 'unset',
      cudaVisibleDevices: process.env.CUDA_VISIBLE_DEVICES ?? 'unset',
    },
    checks: [
      {
        name: 'nvidia-device-nodes',
        status: deviceNodes.length > 0 ? 'present' : 'missing',
        details: deviceNodes,
      },
      {
        name: 'nvidia-smi',
        status: 'not-run',
        details: 'This is a scaffolded doctor stub, not a real validation claim.',
      },
      {
        name: 'cupy-allocation',
        status: 'not-run',
        details: 'Deferred until the real GPU runtime is wired in.',
      },
    ],
    notes: [
      'This report is explicitly a placeholder.',
      'It records environment signals without claiming hardware validation.',
    ],
  };
}

async function main() {
  const report = buildGpuDoctorReport();
  const payload = `${JSON.stringify(report, null, 2)}\n`;
  const jsonIndex = process.argv.indexOf('--json');
  const outputPath = jsonIndex >= 0 ? process.argv[jsonIndex + 1] : process.env.ECHOFORGE_DOCTOR_OUTPUT;

  if (outputPath) {
    await writeFile(outputPath, payload, 'utf8');
  }

  process.stdout.write(payload);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}

