"""
eFlesh serial reader — PCA9546A mux edition
10 sensors (2 boards × 5), + IMU
 
Frame layout (after 0xAA 0x55 header):
  160 bytes  — 10 × MLX90393::txyz (each 16 bytes: float t, x, y, z)
   24 bytes  — 6 floats: ax, ay, az, gx, gy, gz
  ─────────
  184 bytes total payload per frame
"""
 
import struct
import threading
import time
import numpy as np
import serial
 
 
HEADER        = b'\xAA\x55'
NUM_SENSORS   = 10
FLOATS_PER_MAG = 4          # t, x, y, z
MAG_BYTES     = NUM_SENSORS * FLOATS_PER_MAG * 4   # 160
IMU_BYTES     = 0                              # 0
FRAME_SIZE    = MAG_BYTES + IMU_BYTES               # 184
 
 
class EFleshMuxReader:
    def __init__(self, port: str, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.lock = threading.Lock()
        self.latest_mag: np.ndarray | None = None   # shape (10, 4)  t,x,y,z
        self.latest_imu: np.ndarray | None = None   # shape (6,)     ax,ay,az,gx,gy,gz
        self.running = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
 
    def start(self):
        self.running = True
        time.sleep(3.0)                 # let firmware finish init prints
        self.ser.reset_input_buffer()
        self._thread.start()
 
    def stop(self):
        self.running = False
        self._thread.join(timeout=2.0)
        self.ser.close()
 
    def get_data(self):
        """Returns (mag_array, imu_array) or (None, None) if no data yet."""
        with self.lock:
            return (
                self.latest_mag.copy() if self.latest_mag is not None else None,
                self.latest_imu.copy() if self.latest_imu is not None else None,
            )
 
    # ── internal ────────────────────────────────────────────────────────────
    def _read_loop(self):
        buf = b''
        while self.running:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
 
            while True:
                idx = buf.find(HEADER)
                if idx == -1:
                    buf = buf[-1:] if buf else b''
                    break
                buf = buf[idx + len(HEADER):]
                if len(buf) < FRAME_SIZE:
                    break   # wait for more bytes
 
                frame = buf[:FRAME_SIZE]
                buf = buf[FRAME_SIZE:]
 
                mag_raw = struct.unpack(f'<{NUM_SENSORS * FLOATS_PER_MAG}f', frame[:MAG_BYTES])
                # imu_raw = struct.unpack('<6f', frame[MAG_BYTES:])
 
                with self.lock:
                    self.latest_mag = np.array(mag_raw, dtype=np.float32).reshape(NUM_SENSORS, FLOATS_PER_MAG)
                    # self.latest_imu = np.array(imu_raw, dtype=np.float32)
 
 
# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
    reader = EFleshMuxReader(port)
    reader.start()
    print(f'Reading from {port}. Ctrl-C to stop.')
    try:
        while True:
            mag, imu = reader.get_data()
            if mag is not None:
                # Print board 0 sensor 0 xyz, board 1 sensor 0 xyz
                # print(f'B0S0 xyz=({mag[0,1]:7.1f},{mag[0,2]:7.1f},{mag[0,3]:7.1f})  '
                #       f'B1S0 xyz=({mag[5,1]:7.1f},{mag[5,2]:7.1f},{mag[5,3]:7.1f})  '
                #       f'IMU ax={imu[0]:5.2f}')
                print(f'B0S0 xyz=({mag[0,1]:7.1f},{mag[0,2]:7.1f},{mag[0,3]:7.1f})  '
                        f'B1S0 xyz=({mag[5,1]:7.1f},{mag[5,2]:7.1f},{mag[5,3]:7.1f})')
            time.sleep(0.05)
    except KeyboardInterrupt:
        reader.stop()
 





