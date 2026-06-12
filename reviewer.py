#!/usr/bin/env python3
"""
TinyTouch — Episode Reviewer

Interactive playback: one window, keyboard navigation between episodes.

Usage:
    uv run reviewer.py                              # list episodes, latest session
    uv run reviewer.py data/session_20260609_*/     # browse from episode 0
    uv run reviewer.py 5                            # start at episode 5, latest session
    uv run reviewer.py 5 data/session_20260609_*/   # start at episode 5, given session

Keys:
    →  /  n     next episode
    ←  /  p     previous episode
    r           restart current episode
    q           quit
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RBFInterpolator

# ── Sensor layout ─────────────────────────────────────────────────────────────
SENSOR_XY_M = np.array([[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]], float) * 1e-3
GRID_N      = 48
GRID_PAD_M  = 7e-3
INTERVAL_MS = 20
TRAIL_LEN   = 60
BG          = "#0d0d1a"

_LO   = (-GRID_PAD_M + SENSOR_XY_M[:, 0].min()) * 1e3
_HI   = ( GRID_PAD_M + SENSOR_XY_M[:, 0].max()) * 1e3
_GX1D = np.linspace(_LO, _HI, GRID_N)
_GX, _GY  = np.meshgrid(_GX1D, _GX1D)
_GRID_PTS = np.column_stack([_GX.ravel(), _GY.ravel()]) * 1e-3
_STEP     = float(_GX1D[1] - _GX1D[0])
_QSTEP    = max(1, GRID_N // 7)

print("Pre-computing RBF weights … ", end="", flush=True)
_RBF_W = np.zeros((GRID_N * GRID_N, len(SENSOR_XY_M)))
for _i in range(len(SENSOR_XY_M)):
    _e = np.zeros(len(SENSOR_XY_M)); _e[_i] = 1.0
    _RBF_W[:, _i] = RBFInterpolator(
        SENSOR_XY_M, _e, kernel="thin_plate_spline", smoothing=0.1
    )(_GRID_PTS)
print("done.")


def _interp(ch):
    return (_RBF_W @ ch).reshape(GRID_N, GRID_N)

def _blob(cx, cy, sigma=2.2):
    return np.exp(-((_GX - cx)**2 + (_GY - cy)**2) / (2 * sigma**2))

def _contact_xy(bmag5, calib):
    R, L, T, B = bmag5[1], bmag5[2], bmag5[3], bmag5[4]
    eps    = 0.5
    asym_x = (R - L) / (R + L + eps)
    asym_y = (T - B) / (T + B + eps)
    half   = (_HI - _LO) * 0.45
    if calib is not None:
        d      = calib["derived"]
        bx     = d.get("baseline_asym_x", 0.0)
        by     = d.get("baseline_asym_y", 0.0)
        sx     = 0.9 * half / max(abs(d.get("asym_x_at_right",  0.3) - bx),
                                   abs(d.get("asym_x_at_left",  -0.3) - bx), 0.01)
        sy     = 0.9 * half / max(abs(d.get("asym_y_at_top",    0.3) - by),
                                   abs(d.get("asym_y_at_bottom",-0.3) - by), 0.01)
        return (float(np.clip((asym_x - bx) * sx, -half, half)),
                float(np.clip((asym_y - by) * sy, -half, half)))
    return float(asym_x * half), float(asym_y * half)


# ── Session helpers ───────────────────────────────────────────────────────────
def _latest_session():
    sessions = sorted(glob.glob("data/session_*/"))
    return sessions[-1] if sessions else None

def _load_session(session_dir):
    d = Path(session_dir)
    npz  = np.load(d / "episodes.npz", allow_pickle=True)
    meta = json.loads((d / "meta.json").read_text())
    return npz, meta

def _load_calib(meta):
    ref = meta.get("calibration", {}).get("file")
    if ref and Path(ref).exists():
        import yaml
        with open(ref) as f:
            return yaml.safe_load(f)
    files = sorted(glob.glob("calibration/calib_*.yaml"))
    if files:
        import yaml
        with open(files[-1]) as f:
            return yaml.safe_load(f)
    return None

def _list(npz, session_dir):
    raw   = npz["raw"]
    bases = npz["ema_baseline"]
    names = npz["label_names"]
    times = npz["timestamps"]
    n     = len(names)
    print(f"\nSession: {session_dir}   ({n} episodes)\n")
    print(f"  {'#':>4}  {'label':<16}  {'timestamp':<26}  peak_delta")
    print("  " + "─" * 62)
    for i in range(n):
        peak = float(np.abs(np.diff(raw[i] - bases[i], axis=0)).max())
        print(f"  {i:>4}  {str(names[i]):<16}  {str(times[i])[:26]}  {peak:7.1f} µT")
    print()


# ── Interactive viewer ────────────────────────────────────────────────────────
class EpisodeViewer:
    def __init__(self, npz, calib, session_dir, start_idx):
        self._npz         = npz
        self._n           = len(npz["labels"])
        self._calib       = calib
        self._session_dir = session_dir

        self.fig = plt.figure(figsize=(12, 10.5), facecolor=BG)
        gs = self.fig.add_gridspec(
            2, 2, hspace=0.42, wspace=0.44,
            left=0.07, right=0.96, top=0.92, bottom=0.08,
        )
        self.axes = [self.fig.add_subplot(gs[r, c])
                     for r in range(2) for c in range(2)]
        ext = [_LO, _HI, _LO, _HI]
        d0  = np.zeros((GRID_N, GRID_N))

        ax = self.axes[0]
        self._style(ax, "|B|  Field Strength  (µT)")
        self._im0 = ax.imshow(d0, origin="lower", extent=ext, cmap="viridis",
                               vmin=0, vmax=120, interpolation="bilinear", aspect="equal")
        self._cb(self._im0, ax, "µT")
        self._peak_dot, = ax.plot([], [], "*", color="white", ms=10, zorder=10)

        ax = self.axes[1]
        self._style(ax, "Bz / |B|  — Axial Ratio")
        self._im1 = ax.imshow(d0, origin="lower", extent=ext, cmap="coolwarm",
                               vmin=-1, vmax=1, interpolation="bilinear", aspect="equal")
        self._cb(self._im1, ax)

        ax = self.axes[2]
        self._style(ax, "|∇|B||  — Contact Boundary")
        self._im2 = ax.imshow(d0, origin="lower", extent=ext, cmap="plasma",
                               vmin=0, vmax=24, interpolation="bilinear", aspect="equal")
        self._cb(self._im2, ax, "µT/mm")
        qx = _GX[::_QSTEP, ::_QSTEP]; qy = _GY[::_QSTEP, ::_QSTEP]
        self._qv = ax.quiver(qx, qy, np.zeros_like(qx), np.zeros_like(qy),
                              color="white", alpha=0.5, scale=28, width=0.007, zorder=5)

        ax = self.axes[3]
        self._style(ax, "Contact Centroid Estimate")
        self._im3 = ax.imshow(d0, origin="lower", extent=ext, cmap="inferno",
                               vmin=0, vmax=1, interpolation="bilinear", aspect="equal")
        ax.scatter(SENSOR_XY_M[:, 0]*1e3, SENSOR_XY_M[:, 1]*1e3,
                   s=22, facecolors="none", edgecolors="white", alpha=0.4, zorder=5)
        self._trail_line, = ax.plot([], [], "o", ms=4, color="#44ffcc",
                                     alpha=0.22, zorder=6, markeredgewidth=0)
        self._est_dot, = ax.plot([], [], "o", ms=14, color="#00ffcc",
                                  markeredgecolor="white", markeredgewidth=1.5, zorder=10)
        self._xy_text = ax.text(0.5, 0.03, "", transform=ax.transAxes,
                                 ha="center", va="bottom", color="#00ffcc",
                                 fontsize=8, fontfamily="monospace")

        for ax in self.axes[:3]:
            ax.scatter(SENSOR_XY_M[:, 0]*1e3, SENSOR_XY_M[:, 1]*1e3,
                       s=14, color="white", alpha=0.35, zorder=10)

        self._title_text  = self.fig.text(0.5, 0.965, "", ha="center", color="white",
                                           fontsize=10, fontweight="bold")
        self._nav_text    = self.fig.text(0.5, 0.945, "", ha="center", color="#888",
                                           fontsize=8)
        self._frame_text  = self.fig.text(0.5, 0.022, "", ha="center", color="#555",
                                           fontsize=8)

        # Progress bar
        prog_ax = self.fig.add_axes([0.07, 0.045, 0.86, 0.012])
        prog_ax.set_xlim(0, 1)
        prog_ax.set_ylim(0, 1)
        prog_ax.set_facecolor("#1a1a2e")
        prog_ax.axis("off")
        self._prog_track = prog_ax.barh(0.5, 1.0, height=1.0,
                                         color="#2a2a3e", align="center")[0]
        self._prog_bar   = prog_ax.barh(0.5, 0.0, height=1.0,
                                         color="#00ffcc", align="center")[0]

        # Clear default arrow-key bindings (back/forward nav) so our handler fires
        try:
            plt.rcParams["keymap.back"]    = [k for k in plt.rcParams["keymap.back"]
                                               if k != "left"]
            plt.rcParams["keymap.forward"] = [k for k in plt.rcParams["keymap.forward"]
                                               if k != "right"]
        except Exception:
            pass

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._load(start_idx)

        self._anim = FuncAnimation(
            self.fig, self._update,
            interval=INTERVAL_MS, cache_frame_data=False, blit=False,
        )

    def _load(self, idx):
        self._ep_idx = idx % self._n
        i = self._ep_idx
        self._raw   = self._npz["raw"][i].astype(np.float32)   # (T, 5, 3)
        self._ema   = self._npz["ema_baseline"][i].astype(np.float64)
        self._T     = len(self._raw)
        self._fi    = 0
        self._trail = []

        vmax_ut = max(float(np.linalg.norm(self._raw, axis=2).max()) * 1.1, 30.0)
        self._vmax0          = vmax_ut
        self._vmax2          = vmax_ut * 0.20
        self._contact_thresh = vmax_ut * 0.05
        self._im0.set_clim(0, vmax_ut)
        self._im2.set_clim(0, vmax_ut * 0.20)

        self._prog_bar.set_width(0)

        label = str(self._npz["label_names"][i])
        ts    = str(self._npz["timestamps"][i])[:19]
        self._title_text.set_text(
            f"Episode {i} / {self._n - 1}  —  {label}   {ts}"
        )
        self._nav_text.set_text(
            f"← / p  prev      → / n  next      r  restart      q  quit"
        )
        print(f"  → ep {i:>3}  {label}  ({self._T} frames)")

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._load(self._ep_idx + 1)
        elif event.key in ("left", "p"):
            self._load(self._ep_idx - 1)
        elif event.key == "r":
            self._load(self._ep_idx)
        elif event.key == "q":
            plt.close(self.fig)

    def _update(self, _):
        fi       = self._fi
        self._fi = (fi + 1) % self._T

        frame = self._raw[fi]
        bx = frame[:, 0].astype(float)
        by = frame[:, 1].astype(float)
        bz = frame[:, 2].astype(float)

        bmag_raw  = np.sqrt(bx**2 + by**2 + bz**2)
        raw_xyz   = np.stack([bx, by, bz], axis=1)
        delta_mag = np.abs(bmag_raw - np.linalg.norm(self._ema, axis=1)).max()
        if delta_mag < self._contact_thresh:
            self._ema = 0.995 * self._ema + 0.005 * raw_xyz

        bx -= self._ema[:, 0]; by -= self._ema[:, 1]; bz -= self._ema[:, 2]

        Bx_g   = _interp(bx); By_g = _interp(by); Bz_g = _interp(bz)
        Bmag_g = np.sqrt(Bx_g**2 + By_g**2 + Bz_g**2)

        self._im0.set_data(Bmag_g)
        pk = np.unravel_index(np.argmax(Bmag_g), Bmag_g.shape)
        self._peak_dot.set_data([_GX[pk]], [_GY[pk]])
        self._vmax0 = max(self._vmax0 * 0.98, float(np.percentile(Bmag_g, 99)))
        self._im0.set_clim(0, self._vmax0)

        self._im1.set_data(Bz_g / (Bmag_g + 1e-6))

        gY, gX   = np.gradient(Bmag_g, _STEP)
        grad_mag = np.hypot(gX, gY)
        self._im2.set_data(grad_mag)
        self._vmax2 = max(self._vmax2 * 0.98, float(np.percentile(grad_mag, 99)))
        self._im2.set_clim(0, self._vmax2)
        gxs = gX[::_QSTEP, ::_QSTEP]; gys = gY[::_QSTEP, ::_QSTEP]
        self._qv.set_UVC(gxs / (np.hypot(gxs, gys) + 1e-6),
                          gys / (np.hypot(gxs, gys) + 1e-6))

        cx, cy = _contact_xy(bmag_raw, self._calib)
        self._im3.set_data(_blob(cx, cy))
        self._est_dot.set_data([cx], [cy])
        self._xy_text.set_text(f"x = {cx:+5.1f} mm    y = {cy:+5.1f} mm")
        self._trail.append((cx, cy))
        if len(self._trail) > TRAIL_LEN: self._trail.pop(0)
        if self._trail:
            self._trail_line.set_data([p[0] for p in self._trail],
                                       [p[1] for p in self._trail])

        suffix = "  [looping]" if fi == self._T - 1 else ""
        self._frame_text.set_text(
            f"frame {fi + 1} / {self._T}   t = {fi / 50:.2f} s{suffix}")
        self._prog_bar.set_width((fi + 1) / self._T)
        self.fig.canvas.draw_idle()

    def _style(self, ax, title):
        ax.set_facecolor(BG)
        ax.set_xlim(_LO, _HI); ax.set_ylim(_LO, _HI)
        ax.set_title(title, color="white", fontsize=9.5, pad=5)
        ax.set_xlabel("X (mm)", color="#666", fontsize=8)
        ax.set_ylabel("Y (mm)", color="#666", fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#2a2a3e")

    def _cb(self, im, ax, label=""):
        cb = self.fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        if label: cb.set_label(label, color="#888", fontsize=7)
        cb.ax.tick_params(colors="#555", labelsize=6)

    def run(self):
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Browse TinyTouch episodes.")
    p.add_argument("args", nargs="*",
                   help="[session_dir] [episode_index] in any order.")
    args = p.parse_args()

    session_dir = None
    start_str   = None
    for a in args.args:
        if a.lstrip("-").isdigit():
            start_str = a
        else:
            session_dir = a

    session_dir = session_dir or _latest_session()
    if session_dir is None:
        print("No sessions found in data/. Run collect.py first.")
        sys.exit(1)

    npz, meta = _load_session(session_dir)
    n         = len(npz["labels"])

    if not args.args:
        _list(npz, session_dir)
        sys.exit(0)

    start = int(start_str) if start_str is not None else 0
    if not (0 <= start < n):
        print(f"Episode {start} out of range (0–{n - 1}).")
        _list(npz, session_dir)
        sys.exit(1)

    calib = _load_calib(meta)
    print(f"\nBrowsing {n} episodes from {Path(session_dir).name}")
    print("Keys: ← → navigate   r restart   q quit\n")

    EpisodeViewer(npz, calib, session_dir, start).run()


if __name__ == "__main__":
    main()
