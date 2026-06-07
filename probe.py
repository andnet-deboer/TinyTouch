#!/usr/bin/env python3
"""
TinyTouch serial probe — verify the chip is streaming before using the viewer.

Usage:
    uv run probe.py /dev/ttyACM0
    uv run probe.py /dev/ttyACM0 --raw     # dump hex bytes instead of decoded values
"""

import argparse
import struct
import sys
import time
import serial

HEADER     = b"\xAA\x55"
N_SENSORS  = 10
FRAME_SIZE = N_SENSORS * 4 * 4   # 10 sensors × 4 floats × 4 bytes = 160
# Note: no IMU in probe — just confirm sensor data

SENSOR_LABELS = [f"B{b}S{i}" for b in range(2) for i in range(5)]


def probe_raw(ser, seconds=5):
    """Dump raw hex for `seconds` to show what the chip is actually sending."""
    print(f"=== RAW HEX DUMP for {seconds}s (Ctrl-C to stop early) ===\n")
    deadline = time.time() + seconds
    total = 0
    try:
        while time.time() < deadline:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                total += len(chunk)
                hex_str = chunk.hex(" ")
                print(hex_str)
    except KeyboardInterrupt:
        pass
    print(f"\n--- {total} bytes received ---")


def probe_frames(ser):
    """Sync to 0xAA 0x55 header and decode sensor values."""
    print("=== FRAME DECODE (Ctrl-C to stop) ===")
    print("Waiting for first frame (up to 5 s)…\n")

    buf = b""
    frames = 0
    t_start = time.time()
    t_last  = t_start

    try:
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                if time.time() - t_start > 5 and frames == 0:
                    print("ERROR: no frames found in 5 s.")
                    print("  • Check the port and baud rate.")
                    print("  • Make sure the updated firmware (with 0xAA 0x55 header) is flashed.")
                    print("  • Try --raw to see what the chip is actually sending.")
                    return
                continue

            buf += chunk

            while True:
                idx = buf.find(HEADER)
                if idx == -1:
                    buf = buf[-1:] if buf else b""
                    break
                buf = buf[idx + len(HEADER):]
                if len(buf) < FRAME_SIZE:
                    break

                frame = buf[:FRAME_SIZE]
                buf   = buf[FRAME_SIZE:]
                frames += 1

                now = time.time()
                fps = 1.0 / (now - t_last) if frames > 1 else 0.0
                t_last = now

                vals = struct.unpack(f"<{N_SENSORS * 4}f", frame)
                data = [vals[i*4:(i+1)*4] for i in range(N_SENSORS)]   # (t, x, y, z) per sensor

                # Print a compact table every frame
                lines = [f"Frame {frames:5d}  {fps:5.1f} fps\n"]
                lines.append(f"  {'sensor':6s}  {'Bx':>8s}  {'By':>8s}  {'Bz':>8s}  {'|B|':>8s}")
                lines.append(f"  {'------':6s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
                for i, (t, x, y, z) in enumerate(data):
                    norm = (x**2 + y**2 + z**2) ** 0.5
                    board_sep = "\n" if i == 4 else ""
                    lines.append(f"{board_sep}  {SENSOR_LABELS[i]:6s}  {x:8.1f}  {y:8.1f}  {z:8.1f}  {norm:8.1f}")

                # Move cursor up to overwrite previous frame
                if frames > 1:
                    n_lines = len(lines) + 1   # +1 for the blank board separator
                    print(f"\033[{n_lines}A", end="")

                print("\n".join(lines))
                sys.stdout.flush()

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\n\n--- {frames} frames in {elapsed:.1f} s ({frames/elapsed:.1f} fps avg) ---")


def main():
    p = argparse.ArgumentParser(description="TinyTouch serial probe")
    p.add_argument("port", nargs="?", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--raw", action="store_true", help="Hex dump instead of decoded frames")
    p.add_argument("--raw-seconds", type=int, default=5, metavar="S")
    args = p.parse_args()

    print(f"Opening {args.port} at {args.baud} baud…")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05, dsrdtr=False, rtscts=False)
        ser.dtr = False
    except serial.SerialException as e:
        sys.exit(f"Cannot open port: {e}")

    time.sleep(4.0)   # QT Py ESP32-S3 needs ~3 s for USB CDC + firmware init
    ser.reset_input_buffer()

    if args.raw:
        probe_raw(ser, args.raw_seconds)
    else:
        probe_frames(ser)

    ser.close()


if __name__ == "__main__":
    main()
