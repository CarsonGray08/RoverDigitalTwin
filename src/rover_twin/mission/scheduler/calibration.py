"""Rover calibration files: provenance-tagged parameters for the simulator.

A calibration file holds every rover parameter the simulator needs, each with
its provenance, and a mission file points at it:

    mission_calibrated_preliminary.json
        {"calibration_file": "config/rover_calibration_preliminary.json",
         "rover": {"battery_wh": 2304.0},      # optional overrides (recorded)
         "tasks": [...], ...}

``load_mission(path)`` reads a mission file, resolves ``calibration_file``
relative to it, and returns the ordinary mission dictionary the existing
``parse_mission()`` / ``MissionSimulator`` consume - plus a ``_provenance``
block so every number can be traced. Missions without ``calibration_file``
(the synthetic ``mission.json``) pass through untouched.

Calibration file format (one entry per simulator parameter)::

    {"_meta": {"name": ..., "specification_status": "preliminary", ...},
     "parameters": {
        "speed_m_per_min": {"value": 2.1, "unit": "m/min", "status": "MEASURED",
                            "source": "...", "date": "2026-11-14",
                            "uncertainty": "...", "notes": "..."},
        ...},
     "nominal_battery": {...},     # informational, never used as capacity
     "policy": {...},              # team decisions (reserve, sol) - informational
     "not_modelled": {...}}        # measured things the simulator cannot use yet

Statuses: MEASURED (physical test), DERIVED (computed from measured values),
ASSUMED (no test; reason given), SIMULATED (emulated by design), POLICY
(team decision). Nothing here is used by NF/SA/GA directly; they only see the
numbers in ``mission["rover"]``.
"""
from __future__ import annotations

import json
from pathlib import Path

from .scheduler_core import MissionError

STATUSES = ("MEASURED", "DERIVED", "ASSUMED", "SIMULATED", "POLICY")

# Simulator rover parameters, their required units, and whether they must exist.
ROVER_PARAMETERS: dict[str, tuple[str, bool]] = {
    "battery_capacity_wh": ("Wh", True),
    "battery_wh": ("Wh", True),
    "reserve_wh": ("Wh", True),
    "capacity_kg": ("kg", True),
    "speed_m_per_min": ("m/min", True),
    "travel_power_w": ("W", True),
    "unload_min": ("min", True),
    "charging_power_w": ("W", True),
    "default_excavation_power_w": ("W", False),
}
MISSION_PARAMETERS: dict[str, tuple[str, bool]] = {
    "sol_limit_min": ("min", False),
}

# Accepted spellings for each canonical unit (case-insensitive).
UNIT_ALIASES = {
    "Wh": {"wh", "watt-hour", "watt-hours", "watthour", "watthours"},
    "kg": {"kg", "kilogram", "kilograms"},
    "m/min": {"m/min", "metres/min", "meters/min", "m per min", "metre/min", "meter/min"},
    "W": {"w", "watt", "watts"},
    "min": {"min", "minute", "minutes"},
}


class CalibrationError(MissionError):
    """A calibration file is missing, malformed, or inconsistent."""


def _check_unit(key: str, unit, expected: str) -> None:
    if not isinstance(unit, str) or unit.strip().lower() not in UNIT_ALIASES[expected]:
        raise CalibrationError(f"{key}: unit must be '{expected}', got {unit!r}")


def _check_entry(key: str, entry, expected_unit: str) -> float:
    if not isinstance(entry, dict):
        raise CalibrationError(f"{key}: each parameter must be an object with value/unit/status/source")
    for field in ("value", "unit", "status", "source"):
        if field not in entry:
            raise CalibrationError(f"{key}: missing '{field}'")
    value = entry["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{key}: value must be a number, got {value!r}")
    _check_unit(key, entry["unit"], expected_unit)
    if entry["status"] not in STATUSES:
        raise CalibrationError(f"{key}: status must be one of {STATUSES}, got {entry['status']!r}")
    if not str(entry["source"]).strip():
        raise CalibrationError(f"{key}: source must not be empty")
    return float(value)


def load_calibration(path: str | Path) -> dict:
    """Read and validate a calibration file. Returns the parsed dictionary."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"calibration file {path} is not valid JSON: {exc}") from exc
    validate_calibration(data)
    data.setdefault("_meta", {})["_path"] = str(path)
    return data


def validate_calibration(data: dict) -> None:
    if not isinstance(data, dict) or "parameters" not in data or not isinstance(data["parameters"], dict):
        raise CalibrationError("calibration file must be an object with a 'parameters' block")
    params = data["parameters"]
    for key, (unit, required) in {**ROVER_PARAMETERS, **MISSION_PARAMETERS}.items():
        if key in params:
            _check_entry(key, params[key], unit)
        elif required:
            raise CalibrationError(f"calibration is missing required parameter '{key}'")
    unknown = set(params) - set(ROVER_PARAMETERS) - set(MISSION_PARAMETERS)
    if unknown:
        raise CalibrationError(f"unknown parameter(s) in calibration: {', '.join(sorted(unknown))}")
    meta = data.get("_meta", {})
    if not isinstance(meta, dict) or "specification_status" not in meta:
        raise CalibrationError("calibration _meta must declare specification_status")


def rover_values(data: dict) -> dict[str, float]:
    """Plain numeric rover block for parse_mission()."""
    return {k: float(data["parameters"][k]["value"]) for k in ROVER_PARAMETERS if k in data["parameters"]}


def provenance_table(data: dict) -> list[dict]:
    rows = []
    for key, entry in data["parameters"].items():
        rows.append({"parameter": key, "value": entry["value"], "unit": entry["unit"], "status": entry["status"],
                     "source": entry["source"], "date": entry.get("date", ""),
                     "uncertainty": entry.get("uncertainty", ""), "notes": entry.get("notes", "")})
    return rows


def apply_calibration(mission: dict, calibration: dict) -> dict:
    """Return a new mission dict whose rover block comes from the calibration.

    Precedence: a key present in the mission's own ``rover`` block overrides the
    calibration (and is recorded as an override in ``_provenance``); every other
    key comes from the calibration. ``sol_limit_min`` follows the same rule.
    """
    out = json.loads(json.dumps(mission))      # deep copy, keeps it plain JSON
    cal_rover = rover_values(calibration)
    overrides = dict(out.get("rover") or {})
    out["rover"] = {**cal_rover, **overrides}
    if "sol_limit_min" not in out and "sol_limit_min" in calibration["parameters"]:
        out["sol_limit_min"] = float(calibration["parameters"]["sol_limit_min"]["value"])
    prov = {k: dict(calibration["parameters"][k]) for k in calibration["parameters"]}
    for k, v in overrides.items():
        prov[k] = {"value": v, "unit": ROVER_PARAMETERS.get(k, ("?",))[0], "status": "MISSION_OVERRIDE",
                   "source": "rover block of the mission file", "notes": "overrides calibration file"}
    out["_provenance"] = {
        "calibration_file": calibration.get("_meta", {}).get("_path"),
        "calibration_meta": {k: v for k, v in calibration.get("_meta", {}).items() if k != "_path"},
        "parameters": prov,
        "nominal_battery": calibration.get("nominal_battery"),
        "policy": calibration.get("policy"),
        "not_modelled": calibration.get("not_modelled"),
    }
    meta = out.setdefault("_meta", {})
    meta["specification_status"] = calibration.get("_meta", {}).get("specification_status", "calibrated")
    meta["calibration_file"] = str(calibration.get("_meta", {}).get("_path"))
    return out


def load_mission(path: str | Path) -> dict:
    """Read a mission file; if it names a ``calibration_file``, merge it in.

    Missions without ``calibration_file`` are returned exactly as written, so
    the synthetic ``mission.json`` behaves as before.
    """
    path = Path(path)
    mission = json.loads(path.read_text(encoding="utf-8"))
    cal_ref = mission.get("calibration_file")
    if not cal_ref:
        return mission
    cal_path = Path(cal_ref)
    if not cal_path.is_absolute():
        cal_path = path.parent / cal_path
    calibration = load_calibration(cal_path)
    merged = apply_calibration(mission, calibration)
    merged.pop("calibration_file", None)
    return merged


def summary_lines(mission: dict) -> list[str]:
    """Human-readable provenance for the terminal."""
    prov = mission.get("_provenance")
    if not prov:
        status = mission.get("_meta", {}).get("specification_status", "unknown")
        return [f"Rover parameters: from mission file (specification_status = {status})"]
    lines = [f"Rover parameters: calibration {prov['calibration_file']} "
             f"({prov['calibration_meta'].get('specification_status')})"]
    for k, e in prov["parameters"].items():
        lines.append(f"  {k:<28} {e['value']:>10} {e['unit']:<6} {e['status']:<16} {e['source']}")
    return lines
