#!/usr/bin/env python3
"""
TinyTouch — Tactile Event Data Collection

Classes (from proposal):
  idle           sensor at rest, no contact
  contact_make   press down and hold
  contact_break  start pressed, lift finger during window
  slip           press down then slide laterally while maintaining contact
  tap            quick press and release

Controls:
  i   idle            m   contact_make
  b   contact_break   s   slip
  t   tap
  v   view last episode (ΔB per sensor)
  d   discard last
  q   export + quit

Stored per episode:
  raw (T, 5, 3) float32  — Bx, By, Bz × 5 sensors (before EMA subtraction)
  ema_baseline (5, 3)    — EMA state at recording start (for reproducible subtraction)

Training input:
  delta = diff(raw - ema_baseline, axis=0)  →  (T-1, 15)  normalized by vmax_ut

Usage:
  uv run collect.py --demo
  uv run collect.py /dev/ttyACM0
  uv run collect.py /dev/ttyACM0 --vmax 1200
"""

import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path

BOARD_SLICE   = slice(5, 10)
WINDOW_SECS   = 3.0
POLL_HZ       = 50
WINDOW_FRAMES = int(WINDOW_SECS * POLL_HZ)   # 100 frames

EMA_ALPHA          = 0.005   # ~200 frames ≈ 6 s to track drift
CONTACT_THRESH_FRAC = 0.05   # fraction of vmax_ut — EMA freezes above this

CLASSES = {
    "i": ("idle",          "Rest sensor on table. Do NOT touch it."),
    "m": ("contact_make",  "Press firmly and HOLD for the full window."),
    "b": ("contact_break", "Start with finger pressed — LIFT during the window."),
    "s": ("slip",          "Press down then SLIDE laterally while staying in contact."),
    "t": ("tap",           "Quick sharp TAP — press and release within ~0.3 s."),
}
LABEL_IDX = {k: i for i, k in enumerate(CLASSES)}   # i=0 m=1 b=2 s=3 t=4
SENSOR_NAMES = ["center", "right", "left", "top", "bottom"]


# ── Demo reader ───────────────────────────────────────────────────────────────
class _DemoReader:
    def __init__(self): self._t = 0.0
    def start(self): pass
    def stop(self):  pass
    def get_data(self):
        self._t += 1 / POLL_HZ
        ang = self._t * 0.5
        cx, cy = 4.0 * np.cos(ang), 4.0 * np.sin(ang)
        d  = np.zeros((10, 4), np.float32)
        XY = np.array([[0,0],[5,0],[-5,0],[0,5],[0,-5]], float) * 1e-3
        for i in range(5):
            sx, sy   = XY[i] * 1e3
            dist     = np.hypot(sx - cx, sy - cy) + 0.8
            strength = 95.0 / dist ** 1.3
            ph = ang + i * 0.6
            d[5+i, 1] = strength * np.cos(ph)
            d[5+i, 2] = strength * np.sin(ph)
            d[5+i, 3] = strength * 0.45
        return d, None


# ── EMA tracker ──────────────────────────────────────────────────────────────
class EMABaseline:
    """Rolling baseline that tracks slow TPU drift and freezes during contact."""
    def __init__(self, vmax_ut: float):
        self._ema   = None
        self._thresh = vmax_ut * CONTACT_THRESH_FRAC

    def update(self, xyz: np.ndarray) -> np.ndarray:
        """xyz: (5,3). Returns current EMA state (5,3)."""
        if self._ema is None:
            self._ema = xyz.copy()
        else:
            bmag = np.linalg.norm(xyz, axis=1)
            ema_mag = np.linalg.norm(self._ema, axis=1)
            if np.abs(bmag - ema_mag).max() < self._thresh:
                self._ema = (1 - EMA_ALPHA) * self._ema + EMA_ALPHA * xyz
        return self._ema.copy()

    @property
    def state(self) -> np.ndarray | None:
        return self._ema.copy() if self._ema is not None else None


# ── Recording ─────────────────────────────────────────────────────────────────
def record_episode(reader, ema: EMABaseline):
    """
    Returns dict with raw (T,5,3), ema_baseline (5,3), and stats.
    Returns None on failure.
    """
    frames    = []
    baseline_snapshot = ema.state   # EMA state at recording start

    interval = 1.0 / POLL_HZ
    n_target = WINDOW_FRAMES

    while len(frames) < n_target:
        t0     = time.monotonic()
        mag, _ = reader.get_data()
        if mag is not None:
            xyz = mag[BOARD_SLICE, 1:4].astype(np.float32)
            frames.append(xyz)
            ema.update(xyz.astype(np.float64))
            done = len(frames)
            bar  = "█" * (done * 24 // n_target) + "░" * (24 - done * 24 // n_target)
            print(f"\r  [{bar}] {done}/{n_target}", end="", flush=True)
        rem = interval - (time.monotonic() - t0)
        if rem > 0:
            time.sleep(rem)

    print()
    if len(frames) < n_target // 2:
        print("  ✗  Too few frames.")
        return None

    raw = np.array(frames[:n_target], dtype=np.float32)   # (T, 5, 3)

    if baseline_snapshot is None:
        baseline_snapshot = raw[0].astype(np.float64)

    # Quick delta stats for feedback
    sub   = raw - baseline_snapshot.astype(np.float32)
    delta = np.diff(sub, axis=0)                          # (T-1, 5, 3)
    peak  = float(np.abs(delta).max())
    dominant = SENSOR_NAMES[int(np.abs(delta).reshape(len(delta), 5, 3)
                                .max(axis=(0,2)).argmax())]

    return {
        "raw":           raw,
        "ema_baseline":  baseline_snapshot.astype(np.float32),
        "peak_delta_ut": peak,
        "dominant_sensor": dominant,
    }


# ── Display ───────────────────────────────────────────────────────────────────
def _cls():
    print("\033[2J\033[H", end="")


def print_header(episodes: list, vmax: float):
    counts = {k: 0 for k in CLASSES}
    for ep in episodes:
        counts[ep["key"]] = counts.get(ep["key"], 0) + 1

    total = len(episodes)
    print("╔══════════════════════════════════════════════════╗")
    print(f"║  TinyTouch Collection          {datetime.now():%H:%M:%S}          ║")
    print(f"║  Episodes: {total:<5}   vmax ref: {vmax:>8.1f} µT           ║")
    print("╠══════════════════════════════════════════════════╣")
    for key, (name, _) in CLASSES.items():
        n   = counts.get(key, 0)
        bar = "▪" * min(n, 30)
        print(f"║  [{key}] {name:<14}  {n:>3}  {bar:<30} ║")
    print("╚══════════════════════════════════════════════════╝")


def view_last(episodes: list):
    if not episodes:
        print("  No episodes yet.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available.")
        return

    ep  = episodes[-1]
    raw = ep["raw"]                                          # (T, 5, 3)
    sub = raw - ep["ema_baseline"]
    delta = np.diff(sub, axis=0)                            # (T-1, 5, 3)
    t = np.arange(len(delta)) / POLL_HZ

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), facecolor="#0d0d1a", sharex=True)
    fig.suptitle(f"Episode review — {ep['label_name']}  "
                 f"(peak Δ={ep['peak_delta_ut']:.1f} µT, "
                 f"dominant={ep['dominant_sensor']})",
                 color="white", fontsize=10)

    colors = ["#ff6b6b","#ffd93d","#6bcb77","#4d96ff","#c77dff"]
    for ci, (ax, cname) in enumerate(zip(axes, ["ΔBx","ΔBy","ΔBz"])):
        for si in range(5):
            ax.plot(t, delta[:, si, ci], color=colors[si],
                    alpha=0.85, lw=1.2, label=SENSOR_NAMES[si])
        ax.set_ylabel(f"{cname} (µT)", color="#aaa", fontsize=8)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="#555", labelsize=7)
        ax.axhline(0, color="#333", lw=0.5)
        for sp in ax.spines.values(): sp.set_color("#2a2a3e")
        if ci == 0:
            ax.legend(fontsize=7, framealpha=0.2, labelcolor="white", loc="upper right")
    axes[-1].set_xlabel("Time (s)", color="#aaa", fontsize=8)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


# ── Export ────────────────────────────────────────────────────────────────────
def export(episodes: list, vmax: float, session_dir: Path):
    import json
    if not episodes:
        print("  Nothing to export.")
        return None

    session_dir.mkdir(parents=True, exist_ok=True)

    raw       = np.stack([ep["raw"]          for ep in episodes])  # (N,T,5,3)
    baselines = np.stack([ep["ema_baseline"] for ep in episodes])  # (N,5,3)
    labels    = np.array([LABEL_IDX[ep["key"]] for ep in episodes], dtype=np.int32)

    np.savez(session_dir / "episodes.npz",
             raw=raw.astype(np.float32),
             ema_baseline=baselines.astype(np.float32),
             labels=labels,
             label_names=np.array([ep["label_name"] for ep in episodes]),
             timestamps=np.array([ep["timestamp"]   for ep in episodes]))

    counts = {}
    for ep in episodes:
        counts[ep["label_name"]] = counts.get(ep["label_name"], 0) + 1

    import glob as _glob
    calib_files = sorted(_glob.glob("calibration/calib_*.yaml"))

    meta = {
        "session_id":   session_dir.name,
        "created_at":   datetime.now().isoformat(),
        "n_episodes":   len(episodes),
        "sensor": {
            "board_slice":    [5, 10],
            "poll_hz":        POLL_HZ,
            "window_frames":  WINDOW_FRAMES,
            "window_secs":    WINDOW_SECS,
            "n_sensors":      5,
            "n_axes":         3,
        },
        "calibration": {
            "file":    calib_files[-1] if calib_files else None,
            "vmax_ut": vmax,
        },
        "ema": {
            "alpha":                  EMA_ALPHA,
            "contact_thresh_fraction": CONTACT_THRESH_FRAC,
            "contact_thresh_ut":       vmax * CONTACT_THRESH_FRAC,
        },
        "model_input": {
            "description": "diff(raw - ema_baseline, axis=0).reshape(T-1, 15) / vmax_ut",
            "shape":       [WINDOW_FRAMES - 1, 15],
        },
        "label_map":    {str(i): name for i, (key, (name, _))
                         in enumerate(CLASSES.items())},
        "label_counts": counts,
        "video": {
            "panels":      ["gradient", "centroid"],
            "fps":         POLL_HZ,
            "note":        "render offline with render_videos.py",
        },
    }
    with open(session_dir / "meta.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)

    print(f"\n  ✓  {len(episodes)} episodes → {session_dir}/")
    print(f"     episodes.npz : raw{list(raw.shape)}  labels{list(labels.shape)}")
    print(f"     ΔB input     : ({len(episodes)}, {WINDOW_FRAMES-1}, 15)  after diff")
    for name, n in sorted(counts.items()):
        print(f"     {name:<16}: {n}")
    return session_dir


# ── Session resume ────────────────────────────────────────────────────────────
def _latest_session() -> Path | None:
    sessions = sorted(Path("data").glob("*/episodes.npz"))
    return sessions[-1].parent if sessions else None


def load_session(session_dir: Path) -> tuple[list[dict], float]:
    import json as _json
    d    = np.load(session_dir / "episodes.npz", allow_pickle=True)
    raw  = d["raw"]           # (N, T, 5, 3)
    bas  = d["ema_baseline"]  # (N, 5, 3)
    lab_names  = d["label_names"]
    timestamps = d["timestamps"]

    name_to_key = {name: key for key, (name, _) in CLASSES.items()}

    episodes = []
    for i in range(len(raw)):
        nm  = str(lab_names[i])
        key = name_to_key.get(nm, "i")
        sub   = raw[i] - bas[i]
        delta = np.diff(sub, axis=0)
        peak  = float(np.abs(delta).max())
        dominant = SENSOR_NAMES[int(
            np.abs(delta).reshape(len(delta), 5, 3).max(axis=(0, 2)).argmax()
        )]
        episodes.append({
            "raw":            raw[i],
            "ema_baseline":   bas[i],
            "peak_delta_ut":  peak,
            "dominant_sensor": dominant,
            "key":            key,
            "label_name":     nm,
            "timestamp":      str(timestamps[i]),
        })

    meta_path = session_dir / "meta.json"
    vmax = 500.0
    if meta_path.exists():
        vmax = _json.loads(meta_path.read_text())["calibration"].get("vmax_ut", 500.0)

    return episodes, float(vmax)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("port",   nargs="?", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int,  default=115200)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--vmax", type=float, default=500.0,
                   help="Peak field ref from calibration.py (µT)")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   metavar="SESSION_DIR",
                   help="Resume a session. Omit path to resume the latest one.")
    args = p.parse_args()

    if args.demo:
        reader = _DemoReader()
    else:
        from tinytouch import EFleshMuxReader
        reader = EFleshMuxReader(args.port, args.baud)
    reader.start()

    # ── Resolve session dir and optionally pre-load episodes ──────────────────
    if args.resume is not None:
        if args.resume == "latest":
            session_dir = _latest_session()
            if session_dir is None:
                print("  No existing sessions found — starting fresh.")
                session_dir = Path("data") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                episodes, vmax = [], args.vmax
            else:
                episodes, vmax = load_session(session_dir)
                print(f"  Resumed {session_dir.name}  ({len(episodes)} episodes loaded)")
        else:
            session_dir = Path(args.resume)
            if not (session_dir / "episodes.npz").exists():
                print(f"  ERROR: {session_dir} has no episodes.npz")
                return
            episodes, vmax = load_session(session_dir)
            print(f"  Resumed {session_dir.name}  ({len(episodes)} episodes loaded)")
    else:
        session_dir = Path("data") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        episodes, vmax = [], args.vmax

    vmax        = max(vmax, args.vmax)
    ema         = EMABaseline(vmax)
    current_key = None

    def _prompt():
        """Print the in-mode prompt line."""
        name = CLASSES[current_key][0]
        switches = "  ".join(f"[{k}]" for k in CLASSES if k != current_key)
        print(f"\n  ── {name.upper()} ──  "
              f"ENTER=record   [v]=view   [d]=discard   {switches}   [q]=quit")

    try:
        while True:
            _cls()
            print_header(episodes, vmax)

            # ── Class selection (shown only before first class chosen) ─────────
            if current_key is None:
                print()
                for key, (name, instr) in CLASSES.items():
                    print(f"  [{key}] {name:<16}  {instr}")
                print("\n  Select a class to begin.   [q] to quit.")
                print()
                cmd = input("  > ").strip().lower()
                if cmd in CLASSES:
                    current_key = cmd
                elif cmd == "q":
                    break
                continue

            # ── In-class recording loop ────────────────────────────────────────
            _prompt()
            cmd = input("  > ").strip().lower()

            # Switch class
            if cmd in CLASSES:
                current_key = cmd
                continue

            # Quit
            if cmd == "q":
                break

            # View last
            if cmd == "v":
                view_last(episodes)
                input("  ENTER to continue…")
                continue

            # Discard last saved episode
            if cmd == "d":
                if episodes:
                    r = episodes.pop()
                    print(f"  Discarded ep #{len(episodes)+1}  ({r['label_name']})")
                else:
                    print("  Nothing to discard.")
                time.sleep(0.6)
                continue

            # ENTER → record
            if cmd == "":
                name, instruction = CLASSES[current_key]
                print(f"\n  ➤  {instruction}")
                print("  ◉ Recording…")

                ep = record_episode(reader, ema)
                if ep is None:
                    input("  Press ENTER to continue…")
                    continue

                vmax = max(vmax, float(np.linalg.norm(ep["raw"], axis=2).max()))
                print(f"  ✓  peak Δ={ep['peak_delta_ut']:.1f} µT   "
                      f"dominant sensor: {ep['dominant_sensor']}")

                keep = input("  ENTER=save   d=discard  >  ").strip().lower()
                if keep == "d":
                    print("  Discarded.")
                else:
                    ep["key"]        = current_key
                    ep["label_name"] = name
                    ep["timestamp"]  = datetime.now().isoformat()
                    episodes.append(ep)
                    print(f"  Saved — ep #{len(episodes)}")

                    if len(episodes) % 20 == 0:
                        export(episodes, vmax, session_dir)

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        reader.stop()
        export(episodes, vmax, session_dir)


if __name__ == "__main__":
    main()
