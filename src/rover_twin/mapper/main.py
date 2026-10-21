"""Mapper process.

    python -m rover_twin.mapper.main [--config config/rover_twin.toml]
    python -m rover_twin.mapper.main --replay RUN/phone.scanlog [--speed 1]

Live, it serves the phone and records the stream into the run directory.
``--replay`` feeds a recording instead, paced at its recorded arrival times
(``--speed 0``: as fast as the mapper goes). Each run writes its outputs and
metrics to a fresh directory under ``run.root``.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from rover_twin import config
from rover_twin.link.server import LinkServer
from rover_twin.mapper.pipeline import Mapper
from rover_twin.mapping.tsdf import NvbloxTsdf
from rover_twin.phone import recording
from rover_twin.phone.receiver import PhoneLink

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "rover_twin.toml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)

    cfg = config.load(args.config)
    run_dir = args.run_dir or Path(cfg.run.root) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.toml")
    tsdf = NvbloxTsdf(cfg.tsdf.voxel_m, cfg.depth.max_range_m)
    link = LinkServer(cfg.link.host, cfg.link.port)
    phone = recorder = None
    if args.replay:
        arrivals = recording.replay(args.replay, args.speed)
    else:
        recorder = recording.Recorder(run_dir / "phone.scanlog") if cfg.run.record else None
        arrivals = phone = PhoneLink(cfg.phone.host, cfg.phone.port, recorder)
    print(f"mapper: run {run_dir}, phone {'replay ' + str(args.replay) if args.replay else cfg.phone.port}, "
          f"twin link {link.port}", flush=True)
    try:
        Mapper(cfg, tsdf, link, run_dir).run(arrivals)
    except KeyboardInterrupt:
        pass
    finally:
        for closable in (phone, recorder, link):
            if closable is not None:
                closable.close()
    print(f"mapper: outputs in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
