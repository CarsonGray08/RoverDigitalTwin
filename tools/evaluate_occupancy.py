"""CPU replay diagnostics for experimental occupancy (no TSDF/GPU or twin).

Example: PYTHONPATH=src python tools/evaluate_occupancy.py runs/drift_run.scanlog \
    --output /tmp/chair.json --spot 0.32 0.82 -1.90 -1.40

Spot counts are observations, not ground truth about when a person left.
Timing includes mapper work and rebuilds but excludes TSDF reconstruction,
network transport and simulation. It is a lower bound for a deployed mapper.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from rover_twin import config
from rover_twin.mapper.pipeline import Mapper
from rover_twin.mapping.tsdf import Mesh
from rover_twin.metrics.latency import summarize
from rover_twin.phone import protocol, recording


class NoTsdf:
    def integrate(self, *args):
        pass

    def clear(self):
        pass

    def mesh(self):
        return Mesh.empty()


class NoLink:
    def send(self, message):
        pass

    def snapshot_requested(self):
        return False


def evaluate(path, output, spot=None, until=None):
    cfg = config.load(Path(__file__).resolve().parents[1] / 'config/rover_twin.toml')
    cfg = replace(cfg, occupancy=replace(cfg.occupancy, enabled=True))
    mapper = Mapper(cfg, NoTsdf(), NoLink(), output.parent / output.stem)
    series, frames, rebuilds = [], [], []
    first, last_sample = None, -float('inf')
    for arrival in recording.replay(path, speed=0):
        message = protocol.decode(arrival.payload)
        rgbd = isinstance(message, protocol.RgbdFrame)
        if rgbd:
            if first is None:
                first = message.t_phone
            elapsed = message.t_phone - first
            if until is not None and elapsed > until:
                break
        before = mapper.watch.counts['refuses']
        start = time.perf_counter()
        mapper.process(arrival)
        duration = time.perf_counter() - start
        if mapper.watch.counts['refuses'] > before:
            rebuilds.append({'phone_elapsed_s': elapsed if first is not None else None,
                             'process_ms': round(duration * 1000, 2)})
        if rgbd:
            frames.append(duration)
            if elapsed - last_sample >= 2:
                last_sample = elapsed
                series.append({'phone_elapsed_s': round(elapsed, 3),
                               'spot_columns': mapper.occupancy.count_in(*spot) if spot else None})
            if len(frames) % 500 == 0:
                print(f'{path.name}: {len(frames)} frames, {elapsed:.1f} s', flush=True)
    result = {'recording': str(path), 'until_s': until, 'spot_xy': spot,
              'timing_scope': 'CPU mapper, no TSDF, transport or twin',
              'frame_process_ms': summarize(frames), 'metrics': mapper.watch.summary(),
              'rebuilds': rebuilds, 'series': series,
              'occupied_voxels_end': len(mapper.occupancy.occupied())}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{output}: {result["frame_process_ms"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spot', type=float, nargs=4, metavar=('X0', 'X1', 'Y0', 'Y1'))
    parser.add_argument('--until', type=float, help='seconds since first RGB-D frame; omit for full recording')
    args = parser.parse_args()
    evaluate(args.recording, args.output, args.spot, args.until)
