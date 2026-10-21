#!/usr/bin/env bash
# Set up a lab workstation for the mapper and the twin, then check it works.
#
#   scripts/setup_workstation.sh [ISAAC_SIM_ROOT]
#
# Ubuntu 22.04 or 24.04 with an NVIDIA GPU. Installs the system packages
# nvblox needs, builds .venv (PyTorch, nvblox_torch, this package) for the
# mapper and the tests, records it and Isaac Sim's path in config/system.env,
# then runs the suite and, if runs/drift_run.scanlog is here, replays it
# through nvblox and compares the map with the reference numbers. Isaac Sim 6
# is not downloaded: install it first (NVIDIA's licence page) and pass its
# folder, or set ISAAC_SIM_ROOT in config/system.env. Safe to run again.
set -euo pipefail
unset PYTHONPATH                 # ROS's breaks pip, pytest and Isaac's Python

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
ENV_FILE=config/system.env

NVBLOX=0.0.9                     # the version the lab maps were made with
TORCH=2.9.1                      # nvblox_torch 0.0.9 needs torch <= 2.9.1
CUDA=cu128                       # the first CUDA PyTorch builds for Blackwell (RTX 50xx)
MIN_DRIVER=570                   # what CUDA 12.8 needs

say() { printf '\n[setup] %s\n' "$*"; }
die() { printf '\n[setup] error: %s\n' "$*" >&2; exit 1; }

set_env() {                      # set_env KEY VALUE: replace or append in config/system.env
  if grep -q "^$1=" "$ENV_FILE"; then
    sed -i "s|^$1=.*|$1=$2|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$1" "$2" >> "$ENV_FILE"
  fi
}

say "checking the machine"
. /etc/os-release
case "$VERSION_ID" in
  24.04) UBUNTU=24; PACKAGES=(python3-venv python3-pip libglib2.0-0t64 libgl1) ;;
  22.04) UBUNTU=22; PACKAGES=(python3-venv python3-pip libglib2.0-0 libgl1) ;;
  *) die "Ubuntu $VERSION_ID: nvblox_torch is published for 22.04 and 24.04 only" ;;
esac
command -v nvidia-smi > /dev/null || die "no NVIDIA driver (nvidia-smi not found)"
GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
DRIVER="${GPU##*, }"
[ "${DRIVER%%.*}" -ge "$MIN_DRIVER" ] || die "driver $DRIVER is too old for CUDA 12.8; install $MIN_DRIVER or newer"
echo "Ubuntu $VERSION_ID, $GPU"

ISAAC="${1:-$(sed -n 's/^ISAAC_SIM_ROOT=//p' "$ENV_FILE")}"
[ -x "$ISAAC/python.sh" ] || die "no Isaac Sim at '$ISAAC': install Isaac Sim 6 and pass its folder"

missing=()
for p in "${PACKAGES[@]}"; do
  dpkg -s "$p" > /dev/null 2>&1 || missing+=("$p")
done
if [ ${#missing[@]} -gt 0 ]; then
  say "installing ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

say "building .venv"
[ -x .venv/bin/python ] || python3 -m venv .venv
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q "torch==$TORCH" torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
"$PY" -m pip install -q "https://github.com/nvidia-isaac/nvblox/releases/download/v$NVBLOX/nvblox_torch-$NVBLOX%2Bcu12ubuntu$UBUNTU-py3-none-linux_x86_64.whl"
"$PY" -m pip install -q -e '.[dev]'
set_env ISAAC_SIM_ROOT "$ISAAC"
set_env MAPPER_PYTHON "$PY"

say "checking nvblox on the GPU"
GLOG_minloglevel=1 "$PY" - <<'EOF'            # nvblox's info logging off, warnings kept
import torch
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
name, arch = torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability(0)
assert arch in torch.cuda.get_arch_list(), f"this PyTorch has no kernels for {name} ({arch})"
from rover_twin.mapping.tsdf import NvbloxTsdf
NvbloxTsdf(0.02, 5.0)
print(f"nvblox on {name} ({arch})")
EOF

say "checking Isaac Sim's Python"
"$ISAAC/python.sh" -c "import isaacsim, numpy; print('Isaac Sim at $ISAAC')"

say "running the tests"
"$PY" -m pytest -q

if [ -f runs/drift_run.scanlog ]; then
  say "replaying runs/drift_run.scanlog through nvblox"
  RUN="$(mktemp -d)"
  "$PY" -m rover_twin.mapper.main --replay runs/drift_run.scanlog --speed 0 --run-dir "$RUN" > "$RUN/mapper.log" 2>&1 \
    || { tail -20 "$RUN/mapper.log"; die "the replay failed (log above)"; }
  "$PY" - "$RUN" <<'EOF'
import sys
from pathlib import Path
from rover_twin.metrics import report
run = Path(sys.argv[1])
s = report.score(report.load_run(run), report.Truth.load(Path("runs/drift_run.truth.toml")))
# The lab workstation's map of this recording; the grid is this package's
# own code, so a working install reproduces it to within run-to-run noise.
reference = {"floor_rms_m": 0.0050, "floor_step_m": 0.0103, "seam_p95_m": 0.0354}
for metric, value in reference.items():
    print(f"{metric}: {s[metric]:.4f} (reference {value:.4f})")
    assert abs(s[metric] - value) <= 0.002, f"{metric} differs from the reference"
assert (run / "mesh.npz").exists(), "nvblox produced no visual mesh"
print("the map matches the reference")
EOF
  rm -rf "$RUN"
else
  echo "runs/drift_run.scanlog is not on this machine: copy runs/ from the lab workstation to check the map too"
fi

say "done: start mapper and twin with scripts/rover_twin.sh"
