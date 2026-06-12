#!/usr/bin/env python3
"""
TinyTouch calibration — max-force reference only.

Squeezes the sensor against a flat surface to capture the peak field.
Saves calibration/calib_<timestamp>.yaml with vmax_ut for normalisation.

Usage:
    uv run calibration.py --demo
    uv run calibration.py /dev/ttyACM0
"""

import argparse
import time
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

BOARD_SLICE  = slice(5, 10)
SAMPLE_SECS  = 3.0
POLL_HZ      = 50


class _DemoReader:
    def __init__(self): self._t = 0.0
    def start(self): pass
    def stop(self):  pass
    def get_data(self):
        self._t += 1 / POLL_HZ
        ang = self._t * 0.35
        d   = np.zeros((10, 4), np.float32)
        XY  = np.array([[0,0],[5,0],[-5,0],[0,5],[0,-5]], float) * 1e-3
        for i in range(5):
            sx, sy   = XY[i] * 1e3
            dist     = np.hypot(sx - 3.8*np.cos(ang), sy - 3.8*np.sin(ang)) + 0.8
            strength = 95.0 / dist ** 1.3
            ph = ang + i * 0.6
            d[5+i, 1] = strength * np.cos(ph)
            d[5+i, 2] = strength * np.sin(ph)
            d[5+i, 3] = strength * 0.45
        return d, None


def collect(reader, duration_s: float) -> np.ndarray:
    """Returns (N, 5, 3) float64."""
    frames   = []
    interval = 1.0 / POLL_HZ
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        t0     = time.monotonic()
        mag, _ = reader.get_data()
        if mag is not None:
            frames.append(mag[BOARD_SLICE, 1:4].astype(np.float64))
        rem = interval - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)
    return np.array(frames) if frames else np.zeros((1, 5, 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("port",   nargs="?", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int,  default=115200)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        reader = _DemoReader()
    else:
        from tinytouch import EFleshMuxReader
        reader = EFleshMuxReader(args.port, args.baud)
    reader.start()

    print("\n╔══════════════════════════════════════════════╗")
    print("║     TinyTouch Calibration — Max Force        ║")
    print("╚══════════════════════════════════════════════╝\n")
    print("  Press the ENTIRE sensor face flat against a")
    print("  hard surface at maximum force and hold it.\n")
    input("  Press ENTER when ready…")

    print("\n  Recording…", end="", flush=True)
    frames = collect(reader, SAMPLE_SECS)
    print(f" {len(frames)} frames collected.")
    reader.stop()

    bmag       = np.linalg.norm(frames, axis=2)   # (N, 5)
    vmax_ut    = float(bmag.max()) * 1.1
    grad_vmax  = vmax_ut * 0.20

    ts  = datetime.now()
    Path("calibration").mkdir(exist_ok=True)
    out = Path(f"calibration/calib_{ts.strftime('%Y%m%d_%H%M%S')}.yaml")

    with open(out, "w") as f:
        yaml.dump({
            "timestamp":    ts.isoformat(),
            "board_slice":  [5, 10],
            "sample_secs":  SAMPLE_SECS,
            "n_frames":     len(frames),
            "derived": {
                "vmax_ut":   vmax_ut,
                "grad_vmax": grad_vmax,
            },
        }, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✓  vmax_ut  = {vmax_ut:.1f} µT")
    print(f"     Saved  → {out}\n")


if __name__ == "__main__":
    main()
