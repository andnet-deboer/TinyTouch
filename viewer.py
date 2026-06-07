#!/usr/bin/env python3
"""
TinyTouch B-field viewer
Two top-down stream plots (one per board/finger), viridis colour = |Bxy|
Prediction label at top — hook in your CNN by replacing _predict().

Usage:
    uv run viewer.py --demo
    uv run viewer.py /dev/ttyACM0
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RBFInterpolator

# ── Sensor positions (metres, local board frame) ─────────────────────────────
# eFlesh cross: centre + 4 cardinal points at 5 mm pitch
SENSOR_XY_M = np.array([[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]], float) * 1e-3

BOARD_TITLE = ["Finger 0", "Finger 1"]
GRID_N      = 28          # interpolation grid resolution
GRID_PAD_M  = 7e-3        # grid margin beyond sensor extent
INTERVAL_MS = 60


# ── Pluggable predictor ───────────────────────────────────────────────────────
def _predict(mag_10x4: np.ndarray) -> str:
    """
    Replace this with your TFLite Micro / CNN inference call.
    mag_10x4: latest frame, shape (10, 4) — columns t, Bx, By, Bz in µT.
    Return a string label to display, e.g. "slip (0.94)".
    """
    return ""


# ── Demo reader ───────────────────────────────────────────────────────────────
class _DemoReader:
    def __init__(self): self._t = 0.0
    def start(self): pass
    def stop(self): pass
    def get_data(self):
        self._t += INTERVAL_MS / 1000.0
        d = np.zeros((10, 4), np.float32)
        for i in range(10):
            ph = self._t * 2.5 + i * 0.8
            d[i, 1] = 70 * np.cos(ph)
            d[i, 2] = 70 * np.sin(ph)
            d[i, 3] = 30 * np.sin(ph * 0.6 + i)
        return d, None


# ── Viewer ────────────────────────────────────────────────────────────────────
_LO = (-GRID_PAD_M + SENSOR_XY_M[:, 0].min()) * 1e3   # grid extents in mm
_HI = ( GRID_PAD_M + SENSOR_XY_M[:, 0].max()) * 1e3

_GX1D = np.linspace(_LO, _HI, GRID_N)   # 1-D axes for streamplot
_GX, _GY = np.meshgrid(_GX1D, _GX1D)
_GRID_PTS = np.column_stack([_GX.ravel(), _GY.ravel()]) * 1e-3  # back to metres


def _interp_bxy(bxy: np.ndarray):
    """Interpolate Bx, By from 5 sensor points onto the regular grid."""
    rbf_x = RBFInterpolator(SENSOR_XY_M, bxy[:, 0], kernel="thin_plate_spline", smoothing=0.1)
    rbf_y = RBFInterpolator(SENSOR_XY_M, bxy[:, 1], kernel="thin_plate_spline", smoothing=0.1)
    Bx = rbf_x(_GRID_PTS).reshape(GRID_N, GRID_N)
    By = rbf_y(_GRID_PTS).reshape(GRID_N, GRID_N)
    return Bx, By


class TinyTouchViewer:
    def __init__(self, reader):
        self.reader = reader

        self.fig, self.axes = plt.subplots(
            1, 2, figsize=(11, 5.5), facecolor="#0d0d1a"
        )
        self.fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.1, wspace=0.35)

        self._pred = self.fig.text(
            0.5, 0.93, "—",
            ha="center", va="top", color="white",
            fontsize=15, fontweight="bold",
            transform=self.fig.transFigure,
        )

        self._anim = FuncAnimation(
            self.fig, self._update, interval=INTERVAL_MS, cache_frame_data=False
        )

    # ── Per-frame ─────────────────────────────────────────────────────────────
    def _update(self, _frame):
        mag_data, _ = self.reader.get_data()
        if mag_data is None:
            return

        label = _predict(mag_data)
        norms = np.linalg.norm(mag_data[:, 1:3], axis=1)

        for b, ax in enumerate(self.axes):
            ax.cla()
            self._style_ax(ax, b)

            idx = slice(b * 5, b * 5 + 5)
            bxy = mag_data[idx, 1:3].astype(float)  # (5, 2) µT

            try:
                Bx, By = _interp_bxy(bxy)
                speed = np.hypot(Bx, By)
                ax.streamplot(
                    _GX1D, _GX1D, Bx, By,
                    color=speed, cmap="viridis",
                    linewidth=1.3, density=1.4, arrowsize=1.1,
                    norm=plt.Normalize(vmin=0, vmax=max(speed.max(), 1)),
                )
            except Exception:
                pass

            # White quiver arrows at the 5 actual measurement points
            ax.quiver(
                SENSOR_XY_M[:, 0] * 1e3, SENSOR_XY_M[:, 1] * 1e3,
                bxy[:, 0], bxy[:, 1],
                color="white", scale=600, width=0.007, zorder=5,
            )
            # Sensor dots
            ax.scatter(
                SENSOR_XY_M[:, 0] * 1e3, SENSOR_XY_M[:, 1] * 1e3,
                s=30, color="white", zorder=6, alpha=0.7,
            )

            avg_n = norms[idx].mean()
            ax.set_title(
                f"{BOARD_TITLE[b]}   |B| {avg_n:.1f} µT",
                color="white", fontsize=11, pad=6,
            )

        self._pred.set_text(label if label else "")

    def _style_ax(self, ax, b):
        ax.set_facecolor("#0d0d1a")
        ax.set_xlim(_LO, _HI)
        ax.set_ylim(_LO, _HI)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)", color="#666", fontsize=8, labelpad=3)
        ax.set_ylabel("Y (mm)", color="#666", fontsize=8, labelpad=3)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#2a2a3e")

    def run(self):
        self.reader.start()
        plt.show()
        self.reader.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("port", nargs="?", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--demo", action="store_true")
    args = p.parse_args()

    if args.demo:
        reader = _DemoReader()
    else:
        from tinytouch import EFleshMuxReader
        reader = EFleshMuxReader(args.port, args.baud)

    TinyTouchViewer(reader).run()


if __name__ == "__main__":
    main()
