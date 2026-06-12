#!/usr/bin/env python3
"""
TinyTouch serial→ROS2 bridge.

Reads the MODE_SERIAL binary frames from the ESP32-S3 and publishes
Float32MultiArray on /tinytouch/raw at ~50 Hz.

Usage (source jazzy first):
    source /opt/ros/jazzy/setup.bash
    python3 ros_bridge.py [/dev/ttyACM0]

Message layout: 40 floats, 10 sensors × 4 values each (t, x, y, z),
boards 0-1, sensors 0-4 each — identical to the MODE_MICROROS firmware output.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from tinytouch import EFleshMuxReader


class TinyTouchBridge(Node):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__("tinytouch_bridge")
        self.pub = self.create_publisher(Float32MultiArray, "/tinytouch/raw", 10)
        self.reader = EFleshMuxReader(port, baud)
        self.reader.start()
        self.create_timer(0.02, self._publish)  # 50 Hz
        self.get_logger().info(f"Bridging {port} → /tinytouch/raw at 50 Hz")

    def _publish(self):
        mag, _ = self.reader.get_data()
        if mag is None:
            return
        msg = Float32MultiArray()
        msg.data = mag.flatten().tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        self.reader.stop()
        super().destroy_node()


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    rclpy.init()
    node = TinyTouchBridge(port)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
