from pathlib import Path

import pytest
import synthetic_scan


@pytest.fixture(scope="session")
def scan(tmp_path_factory):
    """``scan(kind, **options)``: a synthetic recording (full resolution, 5 Hz),
    generated once per session; its truth file sits next to it."""
    root, made = tmp_path_factory.mktemp("scans"), {}

    def make(kind: str, **options) -> Path:
        key = (kind, tuple(sorted(options.items())))
        if key not in made:
            made[key] = root / f"{kind}-{len(made)}.scanlog"
            synthetic_scan.write(kind, made[key], rgbd_hz=5.0, **options)
        return made[key]

    return make
