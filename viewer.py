#!/usr/bin/env python3
"""
TinyTouch – Feature Visualizer (single board, four panels)

  Panel 0: |B| field strength   (viridis)  — raw interpolated magnitude
  Panel 1: Bz / |B|             (coolwarm) — axial ratio / height proxy
  Panel 2: |∇|B||               (plasma)   — gradient magnitude, contact boundary
  Panel 3: Contact centroid     (inferno)  — Gaussian blob at estimated touch point

Automatically loads the newest calib_*.yaml in the working directory.
Run calibration.py first for calibrated colour ranges and contact estimate.

Usage:
    uv run viewer2.py --demo
    uv run viewer2.py /dev/ttyACM0
"""

import argparse
import glob
import numpy as np
import yaml
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RBFInterpolator

# ── Sensor layout ─────────────────────────────────────────────────────────────
#   Center=0, Right=1, Left=2, Top=3, Bottom=4  (cross arrangement)
SENSOR_XY_M = np.array([[0, 0], [5, 0], [-5, 0], [0, 5], [0, -5]], float) * 1e-3
BOARD_SLICE = slice(5, 10)   # rows in mag_data for Finger 1

GRID_N      = 48
GRID_PAD_M  = 7e-3
INTERVAL_MS = 30
TRAIL_LEN   = 60
BG          = "#0d0d1a"

# Defaults — overridden at runtime if a calib file is found
_VMAX_UT    = 120.0
_GRAD_VMAX  = 22.0

# ── Pre-compute RBF interpolation weights ─────────────────────────────────────
_LO = (-GRID_PAD_M + SENSOR_XY_M[:, 0].min()) * 1e3   # mm
_HI = ( GRID_PAD_M + SENSOR_XY_M[:, 0].max()) * 1e3

_GX1D     = np.linspace(_LO, _HI, GRID_N)
_GX, _GY  = np.meshgrid(_GX1D, _GX1D)
_GRID_PTS = np.column_stack([_GX.ravel(), _GY.ravel()]) * 1e-3
_STEP     = float(_GX1D[1] - _GX1D[0])   # mm per grid cell

print("Pre-computing RBF weights … ", end="", flush=True)
_N = len(SENSOR_XY_M)
_RBF_W = np.zeros((GRID_N * GRID_N, _N))
for _i in range(_N):
    _e = np.zeros(_N); _e[_i] = 1.0
    _RBF_W[:, _i] = RBFInterpolator(
        SENSOR_XY_M, _e, kernel="thin_plate_spline", smoothing=0.1
    )(_GRID_PTS)
print("done.")

_QSTEP = max(1, GRID_N // 7)


# ── Calibration loader ────────────────────────────────────────────────────────
def _load_calib() -> dict | None:
    """Load the newest calib_*.yaml in cwd. Returns None if none found."""
    files = sorted(glob.glob("calibration/calib_*.yaml"))
    if not files:
        return None
    with open(files[-1]) as f:
        calib = yaml.safe_load(f)
    print(f"Loaded calibration: {files[-1]}")
    return calib


# ── Math helpers ──────────────────────────────────────────────────────────────
def _interp(channel: np.ndarray) -> np.ndarray:
    return (_RBF_W @ channel).reshape(GRID_N, GRID_N)


def _contact_xy(bmag5: np.ndarray, calib: dict | None) -> tuple:
    """
    Differential asymmetry estimate of contact (x, y) in mm.
    Without calib: raw ratio × half-grid.
    With calib:    normalised by measured left/right/top/bottom asymmetry ranges.
    """
    R, L, T, B = bmag5[1], bmag5[2], bmag5[3], bmag5[4]
    eps    = 0.5
    asym_x = (R - L) / (R + L + eps)
    asym_y = (T - B) / (T + B + eps)
    half   = (_HI - _LO) * 0.45

    if calib is not None:
        d      = calib["derived"]
        base_x = d.get("baseline_asym_x", 0.0)
        base_y = d.get("baseline_asym_y", 0.0)
        cx = asym_x - base_x
        cy = asym_y - base_y
        scale_x = 0.9 * half / max(abs(d.get("asym_x_at_right", 0.3) - base_x),
                                    abs(d.get("asym_x_at_left",  -0.3) - base_x), 0.01)
        scale_y = 0.9 * half / max(abs(d.get("asym_y_at_top",    0.3) - base_y),
                                    abs(d.get("asym_y_at_bottom", -0.3) - base_y), 0.01)
        x = float(np.clip(cx * scale_x, -half, half))
        y = float(np.clip(cy * scale_y, -half, half))
    else:
        x = float(asym_x * half)
        y = float(asym_y * half)

    return x, y


def _blob(cx: float, cy: float, sigma: float = 2.2) -> np.ndarray:
    d2 = (_GX - cx) ** 2 + (_GY - cy) ** 2
    return np.exp(-d2 / (2 * sigma ** 2))


# ── Zenoh reader ──────────────────────────────────────────────────────────────
class ZenohReader:
    """Subscribe to rt/tinytouch/raw via Zenoh; same interface as EFleshMuxReader."""

    def __init__(self, key: str = "rt/tinytouch/raw"):
        self._key = key
        self._latest: np.ndarray | None = None
        self._lock = __import__("threading").Lock()
        self._session = None
        self._sub = None

    def _on_sample(self, sample):
        import struct
        raw = bytes(sample.payload)
        # CDR header (4) + layout: dim_len (4) + data_offset (4) + seq_len (4) = 16 bytes before floats
        if len(raw) < 16 + 160:
            return
        floats = struct.unpack(f"<40f", raw[16:16 + 160])
        mag = np.array(floats, dtype=np.float32).reshape(10, 4)
        with self._lock:
            self._latest = mag

    def start(self):
        import zenoh
        self._session = zenoh.open(zenoh.Config())
        self._sub = self._session.declare_subscriber(self._key, self._on_sample)

    def stop(self):
        if self._session:
            self._session.close()

    def get_data(self):
        with self._lock:
            mag = self._latest.copy() if self._latest is not None else None
        return mag, None


# ── ROS2 reader ───────────────────────────────────────────────────────────────
ROS_TOPIC = "/tinytouch/raw"

class RosReader:
    """Subscribe to /tinytouch/raw via rclpy; same interface as EFleshMuxReader.
    Caller must have already called rclpy.init()."""

    def __init__(self, topic: str = ROS_TOPIC):
        import threading
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
        self._rclpy = rclpy
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._node = Node("tinytouch_viewer")
        self._node.create_subscription(Float32MultiArray, topic, self._cb, 10)
        self._thread = threading.Thread(
            target=lambda: rclpy.spin(self._node), daemon=True
        )

    def _cb(self, msg):
        mag = np.array(msg.data, dtype=np.float32).reshape(10, 4)
        with self._lock:
            self._latest = mag

    def start(self):
        self._thread.start()

    def stop(self):
        self._node.destroy_node()
        try:
            self._rclpy.shutdown()
        except Exception:
            pass

    def get_data(self):
        with self._lock:
            return (self._latest.copy() if self._latest is not None else None), None


def _try_ros_reader(timeout: float = 2.0):
    """Return a RosReader if /tinytouch/raw is discoverable, else None."""
    try:
        import time
        import rclpy
        from rclpy.node import Node
        rclpy.init()
        probe = Node("tinytouch_probe")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            rclpy.spin_once(probe, timeout_sec=0.1)
            if ROS_TOPIC in dict(probe.get_topic_names_and_types()):
                probe.destroy_node()
                print(f"ROS2 topic {ROS_TOPIC} found — subscribing via ROS2")
                return RosReader()
        probe.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass
    return None


# ── Demo reader ───────────────────────────────────────────────────────────────
class _DemoReader:
    def __init__(self): self._t = 0.0

    def start(self): pass
    def stop(self):  pass

    def get_data(self):
        self._t += INTERVAL_MS / 1000.0
        ang    = self._t * 0.35
        r_mm   = 3.8
        cx, cy = r_mm * np.cos(ang), r_mm * np.sin(ang)

        d = np.zeros((10, 4), np.float32)
        for i in range(5):
            sx, sy   = SENSOR_XY_M[i] * 1e3
            dist     = np.hypot(sx - cx, sy - cy) + 0.8
            strength = 95.0 / dist ** 1.3
            ph = ang + i * 0.6
            d[5 + i, 1] = strength * np.cos(ph)
            d[5 + i, 2] = strength * np.sin(ph)
            d[5 + i, 3] = strength * 0.45 * np.cos(ang * 0.7)
        return d, None


# ── Viewer ────────────────────────────────────────────────────────────────────
class FeatureViewer:
    def __init__(self, reader):
        self.reader  = reader
        self._trail: list[tuple] = []
        self._calib  = _load_calib()

        # Pull calibrated limits (fall back to module defaults)
        vmax_ut   = float(self._calib["derived"]["vmax_ut"])  \
                    if self._calib else _VMAX_UT
        grad_vmax = float(self._calib["derived"]["grad_vmax"]) \
                    if self._calib else _GRAD_VMAX

        # Rolling EMA baseline — tracks slow TPU drift, freezes during contact.
        # Updated only when |ΔBmag| < 5% of vmax (no active contact).
        # Viewer-only: not used as model input.
        self._ema      = None   # (5, 3) float64, initialised on first frame
        self._ema_alpha = 0.005  # ~200 frames ≈ 6 s to fully track a slow drift
        self._contact_thresh = vmax_ut * 0.05

        calib_label = (
            f"vmax: {vmax_ut:.0f} µT  calib: {self._calib['timestamp'][:10]}"
            if self._calib else "no calib — run calibration.py"
        )

        # Adaptive colour-scale trackers (grow instantly, decay slowly)
        self._vmax0 = vmax_ut    # panel 0: |B|
        self._vmax2 = grad_vmax  # panel 2: |∇|B||

        self.fig = plt.figure(figsize=(12, 10.5), facecolor=BG)
        self.fig.suptitle(
            f"TinyTouch — Feature Representations    [{calib_label}]",
            color="white" if self._calib else "#ff8844",
            fontsize=11, fontweight="bold", y=0.985,
        )

        gs = self.fig.add_gridspec(
            2, 2, hspace=0.42, wspace=0.44,
            left=0.07, right=0.96, top=0.935, bottom=0.06,
        )
        self.axes = [self.fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
        ext = [_LO, _HI, _LO, _HI]
        d0  = np.zeros((GRID_N, GRID_N))

        # ── Panel 0: |B| field strength ───────────────────────────────────────
        ax = self.axes[0]
        self._style(ax, "|B|  Field Strength  (µT)")
        self._im0 = ax.imshow(
            d0, origin="lower", extent=ext, cmap="viridis",
            vmin=0, vmax=vmax_ut, interpolation="bilinear", aspect="equal",
        )
        self._cb(self._im0, ax, "µT")
        self._peak_dot, = ax.plot([], [], "*", color="white", ms=10, zorder=10)

        # ── Panel 1: Bz / |B| axial ratio ────────────────────────────────────
        ax = self.axes[1]
        self._style(ax, "Bz / |B|  — Axial Ratio  (height proxy)")
        self._im1 = ax.imshow(
            d0, origin="lower", extent=ext, cmap="coolwarm",
            vmin=-1, vmax=1, interpolation="bilinear", aspect="equal",
        )
        self._cb(self._im1, ax)

        # ── Panel 2: |∇|B|| gradient magnitude ───────────────────────────────
        ax = self.axes[2]
        self._style(ax, "|∇|B||  — Contact Boundary")
        self._im2 = ax.imshow(
            d0, origin="lower", extent=ext, cmap="plasma",
            vmin=0, vmax=grad_vmax, interpolation="bilinear", aspect="equal",
        )
        self._cb(self._im2, ax, "µT/mm")
        qx = _GX[::_QSTEP, ::_QSTEP]
        qy = _GY[::_QSTEP, ::_QSTEP]
        self._qv = ax.quiver(
            qx, qy, np.zeros_like(qx), np.zeros_like(qy),
            color="white", alpha=0.5, scale=28, width=0.007, zorder=5,
        )

        # ── Panel 3: contact centroid ─────────────────────────────────────────
        ax = self.axes[3]
        self._style(ax, "Contact Centroid Estimate")
        self._im3 = ax.imshow(
            d0, origin="lower", extent=ext, cmap="inferno",
            vmin=0, vmax=1, interpolation="bilinear", aspect="equal",
        )
        ax.scatter(
            SENSOR_XY_M[:, 0] * 1e3, SENSOR_XY_M[:, 1] * 1e3,
            s=22, facecolors="none", edgecolors="white", alpha=0.4, zorder=5,
        )
        self._trail_line, = ax.plot(
            [], [], "o", ms=4, color="#44ffcc", alpha=0.22,
            zorder=6, markeredgewidth=0,
        )
        self._est_dot, = ax.plot(
            [], [], "o", ms=14, color="#00ffcc",
            markeredgecolor="white", markeredgewidth=1.5, zorder=10,
        )
        self._xy_text = ax.text(
            0.5, 0.03, "", transform=ax.transAxes,
            ha="center", va="bottom", color="#00ffcc",
            fontsize=8, fontfamily="monospace",
        )

        # Sensor dots on panels 0–2
        for ax in self.axes[:3]:
            ax.scatter(
                SENSOR_XY_M[:, 0] * 1e3, SENSOR_XY_M[:, 1] * 1e3,
                s=14, color="white", alpha=0.35, zorder=10,
            )

        self._anim = FuncAnimation(
            self.fig, self._update,
            interval=INTERVAL_MS, cache_frame_data=False, blit=False,
        )

    def _style(self, ax, title: str):
        ax.set_facecolor(BG)
        ax.set_xlim(_LO, _HI)
        ax.set_ylim(_LO, _HI)
        ax.set_title(title, color="white", fontsize=9.5, pad=5)
        ax.set_xlabel("X (mm)", color="#666", fontsize=8)
        ax.set_ylabel("Y (mm)", color="#666", fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#2a2a3e")

    def _cb(self, im, ax, label: str = ""):
        cb = self.fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        if label:
            cb.set_label(label, color="#888", fontsize=7)
        cb.ax.tick_params(colors="#555", labelsize=6)

    def _update(self, _frame):
        mag_data, _ = self.reader.get_data()
        if mag_data is None:
            return

        s  = mag_data[BOARD_SLICE]
        bx = s[:, 1].astype(float)
        by = s[:, 2].astype(float)
        bz = s[:, 3].astype(float)

        # Raw per-sensor |B| — used for contact estimate (stable denominator)
        bmag_raw = np.sqrt(bx**2 + by**2 + bz**2)

        # Rolling EMA baseline: tracks slow TPU drift for display only.
        # Freezes when contact is detected so events don't corrupt the baseline.
        raw_xyz = np.stack([bx, by, bz], axis=1)   # (5, 3)
        if self._ema is None:
            self._ema = raw_xyz.copy()
        else:
            delta_mag = np.abs(bmag_raw - np.linalg.norm(self._ema, axis=1)).max()
            if delta_mag < self._contact_thresh:
                self._ema = (1 - self._ema_alpha) * self._ema + self._ema_alpha * raw_xyz

        bx = bx - self._ema[:, 0]
        by = by - self._ema[:, 1]
        bz = bz - self._ema[:, 2]
        bmag = np.sqrt(bx**2 + by**2 + bz**2)

        Bx_g   = _interp(bx)
        By_g   = _interp(by)
        Bz_g   = _interp(bz)
        Bmag_g = np.sqrt(Bx_g**2 + By_g**2 + Bz_g**2)

        # Panel 0 — |B|  (adaptive colour scale: grows instantly, decays at 2%/frame)
        self._im0.set_data(Bmag_g)
        pk = np.unravel_index(np.argmax(Bmag_g), Bmag_g.shape)
        self._peak_dot.set_data([_GX[pk]], [_GY[pk]])
        self._vmax0 = max(self._vmax0 * 0.98, float(np.percentile(Bmag_g, 99)))
        self._im0.set_clim(0, self._vmax0)

        # Panel 1 — Bz / |B|  (fixed −1…1, no autoscale needed)
        self._im1.set_data(Bz_g / (Bmag_g + 1e-6))

        # Panel 2 — |∇|B|| + normalised direction arrows  (adaptive)
        gY, gX = np.gradient(Bmag_g, _STEP)
        grad_mag = np.hypot(gX, gY)
        self._im2.set_data(grad_mag)
        self._vmax2 = max(self._vmax2 * 0.98, float(np.percentile(grad_mag, 99)))
        self._im2.set_clim(0, self._vmax2)
        gxs   = gX[::_QSTEP, ::_QSTEP]
        gys   = gY[::_QSTEP, ::_QSTEP]
        gnorm = np.hypot(gxs, gys) + 1e-6
        self._qv.set_UVC(gxs / gnorm, gys / gnorm)

        # Panel 3 — contact centroid blob + trail
        cx, cy = _contact_xy(bmag_raw, self._calib)
        self._im3.set_data(_blob(cx, cy))
        self._est_dot.set_data([cx], [cy])
        self._xy_text.set_text(f"x = {cx:+5.1f} mm    y = {cy:+5.1f} mm")

        self._trail.append((cx, cy))
        if len(self._trail) > TRAIL_LEN:
            self._trail.pop(0)
        if self._trail:
            self._trail_line.set_data(
                [p[0] for p in self._trail],
                [p[1] for p in self._trail],
            )

        self.fig.canvas.draw_idle()

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
    p.add_argument("--zenoh", action="store_true", help="Subscribe via Zenoh instead of serial")
    args = p.parse_args()

    if args.demo:
        reader = _DemoReader()
    elif args.zenoh:
        reader = ZenohReader()
    else:
        reader = _try_ros_reader()
        if reader is None:
            print(f"ROS2 topic not found — falling back to serial {args.port}")
            from tinytouch import EFleshMuxReader
            reader = EFleshMuxReader(args.port, args.baud)

    FeatureViewer(reader).run()


if __name__ == "__main__":
    main()
