# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) 2022 Florida Space Institute
#     - Renamed the simple subscriber to RassorDriverSubscruber to subscribe
#       to 'ezrassor/wheel_instructions' instead of 'topic'
#     - Created the PySerial code for serial variable 'usb', and properly
#       opened and terminated it in the init function and main function
#     - Modified the callback function to support the change
#     - Created functions "bin_command_format" and "bin_command_send"
#     - Merged obstacle-reactive sensor loop (originally rassor_obstacle_driver.py):
#       reads telemetry from the wheel Arduino and on an obstacle < 0.5 m
#       halts, reverses 0.5 m, and captures a snapshot.
# Modified code is under the MIT License
#
#
#
# MIT License
#
# Copyright (c) 2022 Florida Space Institute

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import socket
import threading
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Int16MultiArray
from geometry_msgs.msg import Twist

import serial

try:
    import cv2
except ImportError:
    cv2 = None


# ============ CONFIG ============
WHEEL_PORT        = "/dev/arduino_wheel"
DRUM_PORT         = "/dev/arduino_drum"
BAUD              = 115200

STOP_DISTANCE_CM  = 50            # 0.5 m obstacle trigger
REVERSE_STEPS     = 1070          # reverse travel on obstacle (rover-specific calibration)

CAMERA_INDEX      = 1
SNAPSHOT_DIR      = "snapshots"

MODE_B_HEADER     = 0xA0
CMD_STOP          = 0x00
CMD_REV           = 0x02
CMD_HALT          = 0xFF

REVERSE_TIMEOUT_S = 20.0          # safety ceiling waiting for reverse to finish
# =================================


class RassorDriverSubscriber(Node):
    usb1 = serial.Serial()
    usb1.baudrate = BAUD
    usb1.port = WHEEL_PORT
    usb1.timeout = 1
    usb1.write_timeout = 0.1
    usb2 = None

    def get_ip_address(self):
        try:
            # Create a dummy socket connection to get the IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Connect to a public IP address (Google's DNS), but we don't actually send any data
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
            formatted_ip =  "ip_" + ip_address.replace(".", "_")
            return formatted_ip
        except Exception as e:
            self.get_logger().error(f"Error finding IP for potato: {e}")
            return None  # Return None or a default value

    def __init__(self):
        RassorDriverSubscriber.usb1.open() # opens the serial connection

        if os.path.exists(DRUM_PORT):
            RassorDriverSubscriber.usb2 = serial.Serial()
            RassorDriverSubscriber.usb2.baudrate = BAUD
            RassorDriverSubscriber.usb2.port = DRUM_PORT
            RassorDriverSubscriber.usb2.write_timeout = 0.1
            RassorDriverSubscriber.usb2.open()

        super().__init__('rassor_driver_subscriber')
        self.get_logger().info("Starting the RE-RASSOR Serial forwarding service.")

        if os.path.exists(DRUM_PORT):
            self.get_logger().info("Arduino Drum is Connected")

        # ---- Obstacle-reaction state ----
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self.usb1_lock = threading.Lock()
        self.usb2_lock = threading.Lock()
        self.obstacle_mode = threading.Event()
        self.reverse_done = threading.Event()
        self._shutdown = threading.Event()

        # ---- Camera (same probe order as datacollection.py) ----
        if cv2 is None:
            self.get_logger().warn("cv2 not available — snapshots disabled")
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.get_logger().warn("No camera found — snapshots disabled")
                self.cap = None
            else:
                self.get_logger().info("Camera connected")

        ROVER_NAME = self.get_ip_address()

        STATUS_TOPIC = f'/{ROVER_NAME}/command_status'
        self.status_publisher = self.create_publisher(String, STATUS_TOPIC, 10)
        self.get_logger().info(f"Status topic initialized: {STATUS_TOPIC}")

        # Publish an initial message to create the topic
        status_msg = String()
        status_msg.data = "Node started"
        self.status_publisher.publish(status_msg)
        self.get_logger().info(f"Published initial status message")



        WHEEL_ACTIONS_TOPIC = f'/{ROVER_NAME}/wheel_instructions'
        WHEEL_ACTIONS_LEGACY_TOPIC = f'/{ROVER_NAME}/wheel_instructions_legacy'
        self.subscription = self.create_subscription(
            Int16MultiArray,
            WHEEL_ACTIONS_TOPIC,
            self.listener_callback,
            100)
        self.subscription_legacy = self.create_subscription(
            Twist,
            WHEEL_ACTIONS_LEGACY_TOPIC,
            self.listener_callback_legacy,
            100)
        self.subscription  # prevent unused variable warning
        self.subscription_legacy  # prevent unused variable warning
        SHOULDER_ACTIONS_TOPIC = f'/{ROVER_NAME}/shoulder_instructions'
        self.subscription2 = self.create_subscription(
            Twist,
            SHOULDER_ACTIONS_TOPIC,
            self.listener_callback2,
            100)
        self.subscription2  # prevent unused variable warning
        FRONT_DRUM_ACTIONS_TOPIC = f'/{ROVER_NAME}/front_drum_instructions'
        self.subscription3 = self.create_subscription(
            Twist,
            FRONT_DRUM_ACTIONS_TOPIC,
            self.listener_callback3,
            100)
        self.subscription3  # prevent unused variable warning

        # ---- Sensor reader thread ----
        self.sensor_thread = threading.Thread(target=self._sensor_loop, daemon=True)
        self.sensor_thread.start()

    def publish_status(self, message):
        self.get_logger().info('Publishing the status of the last command...')
        status_msg = String()
        status_msg.data = message
        self.status_publisher.publish(status_msg)
        self.get_logger().info(f"Published: '{message}'")

    def listener_callback(self, msg):
        if self.obstacle_mode.is_set():
            self.publish_status("Ignored wheel command: obstacle mode")
            return
        if not os.path.exists(WHEEL_PORT):
            self.publish_status("Missing Arduino for Wheels...")
            return
        if len(msg.data) < 3:
            self.get_logger().error("Wheel command expects [command, speed, distance]")
            self.publish_status("Invalid wheel command")
            return
        command, speed, distance = msg.data[0], msg.data[1], msg.data[2]
        self.get_logger().info('command: %d' % command)
        self.get_logger().info('speed: %d' % speed)
        self.get_logger().info('distance: %d' % distance)
        packet = self.bin_command_format(command, speed, distance)
        self.bin_command_send(packet, RassorDriverSubscriber.usb1, self.usb1_lock)  # Send wheel packet to usb1

    def listener_callback2(self, msg):
        if not os.path.exists(DRUM_PORT):
            self.publish_status("Missing Arduino for Drums...")
            return
        self.get_logger().info('linear.y: %f' % msg.linear.y)
        self.get_logger().info('angular.y: %f' % msg.angular.y)
        lin_y = msg.linear.y
        ang_y = msg.angular.y
        packet = self.bin_command_format_shoulder(lin_y, ang_y)
        self.bin_command_send(packet, RassorDriverSubscriber.usb2, self.usb2_lock)  # Send shoulder packet to usb2

    def listener_callback3(self, msg):
        if not os.path.exists(DRUM_PORT):
            self.publish_status("Missing Arduino for Drums...")
            return
        self.get_logger().info('linear.x: %f' % msg.linear.x)
        lin_x = msg.linear.x
        packet = self.bin_command_format_drum(lin_x)
        self.bin_command_send(packet, RassorDriverSubscriber.usb2, self.usb2_lock)  # Send drum packet to usb2

    def bin_command_format(self, command, speed, distance):
        # STOP → Mode A 1-byte graceful deceleration
        if command == CMD_STOP:
            return bytearray([CMD_STOP])
        # All other commands → Mode B 4-byte packet:
        #   [0xA0, direction, steps_hi, steps_lo]
        # distance is interpreted as raw motor steps (0-65535).
        # Speed is fixed at MODE_B_RPM on the Arduino, so speed is unused here.
        steps = min(65535, max(0, int(distance)))
        return bytearray([MODE_B_HEADER, command, (steps >> 8) & 0xFF, steps & 0xFF])

    def listener_callback_legacy(self, msg):
        if self.obstacle_mode.is_set():
            self.publish_status("Ignored wheel command: obstacle mode")
            return
        if not os.path.exists(WHEEL_PORT):
            self.publish_status("Missing Arduino for Wheels...")
            return
        self.get_logger().info('legacy linear.x: %f' % msg.linear.x)
        self.get_logger().info('legacy angular.z: %f' % msg.angular.z)
        lin_x = msg.linear.x
        ang_z = msg.angular.z
        packet = self.bin_command_format_legacy(lin_x, ang_z)
        self.bin_command_send(packet, RassorDriverSubscriber.usb1, self.usb1_lock)

    def bin_command_format_legacy(self, lin_x, ang_z):
        packet = bytearray()
        if lin_x == 0.0 and ang_z == 0.0:
            packet.append(0x00) # stop
        elif lin_x > 0:
            packet.append(0x01) # forward
        elif lin_x < 0:
            packet.append(0x02) # reverse
        elif ang_z > 0:
            packet.append(0x03) # left
        else:
            packet.append(0x04) # right
        return packet

    def bin_command_format_shoulder(self, lin_y, ang_y):
        packet = bytearray()
        if lin_y == 0.0 and ang_y == 0.0:
            packet.append(0x00) # Do nothing
        elif lin_y > 0:
            packet.append(0x05) # Raise Front
        elif lin_y < 0:
            packet.append(0x06) # Lower Front
        elif ang_y > 0:
            packet.append(0x07) # Raise Back
        else:
            packet.append(0x08) # Lower Back
        return packet

    def bin_command_format_drum(self, lin_x):
        packet = bytearray()
        if lin_x == 0.0:
            packet.append(0x00) # stop
        elif lin_x > 0:
            packet.append(0x01) # dump
        else:
            packet.append(0x02) # dig
        return packet

    def bin_command_send(self, packet, usb, lock=None):
        try:
            self.get_logger().info("Sending packet: " + str(packet))
            if lock is not None:
                with lock:
                    usb.write(packet)
            else:
                usb.write(packet)
            self.publish_status("Success")  # If command sent successfully, publish success
        except serial.SerialTimeoutException as e:
            self.get_logger().error(f"Write timeout occurred: {e}")
            self.publish_status("Timeout")
            usb.flushInput()
            usb.flushOutput()
        except Exception as e:
            self.get_logger().error(f"Other exception occurred: {e}")
            self.publish_status("Exception")
            usb.flushInput()
            usb.flushOutput()

    # ==================== sensor loop ====================
    def _sensor_loop(self):
        """Continuously read sensor telemetry from the wheel Arduino.

        Expected line format from firmware (when PRINT_SENSOR_DEBUG is on):
            Ultrasonic: X.X cm | LiDAR: Y.Y cm | STATUS
        """
        while not self._shutdown.is_set():
            try:
                raw = RassorDriverSubscriber.usb1.readline()
            except Exception as e:
                self.get_logger().error(f"Serial read error: {e}")
                time.sleep(0.5)
                continue
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Mode B completion marker — release any thread waiting on reverse.
            if "target distance reached" in line:
                self.reverse_done.set()
                continue

            if "Ultrasonic:" not in line or "LiDAR:" not in line:
                continue

            u_cm = self._parse_field(line, 0, "Ultrasonic:")
            l_cm = self._parse_field(line, 1, "LiDAR:")

            if self.obstacle_mode.is_set():
                continue  # already reacting

            if self._is_obstacle(u_cm, l_cm):
                self.get_logger().warn(
                    f"Obstacle < {STOP_DISTANCE_CM} cm (ultrasonic={u_cm}, lidar={l_cm}) — reacting")
                threading.Thread(target=self._handle_obstacle, daemon=True).start()

    @staticmethod
    def _parse_field(line, idx, key):
        try:
            part = line.split("|")[idx].replace(key, "").strip()
            return float(part.replace("cm", "").strip())
        except Exception:
            return -1.0

    @staticmethod
    def _is_obstacle(u_cm, l_cm):
        # -1 means "no reading" / out of range — don't treat as obstacle.
        return (0 <= u_cm < STOP_DISTANCE_CM) or (0 <= l_cm < STOP_DISTANCE_CM)

    # ==================== obstacle response ====================
    def _handle_obstacle(self):
        self.obstacle_mode.set()
        try:
            self.publish_status("OBSTACLE: halting")
            self.bin_command_send(bytearray([CMD_HALT]), RassorDriverSubscriber.usb1, self.usb1_lock)
            time.sleep(0.2)

            self.publish_status(f"OBSTACLE: reversing {REVERSE_STEPS} steps (~0.5 m)")
            self.reverse_done.clear()
            packet = bytearray([MODE_B_HEADER, CMD_REV,
                                (REVERSE_STEPS >> 8) & 0xFF, REVERSE_STEPS & 0xFF])
            self.bin_command_send(packet, RassorDriverSubscriber.usb1, self.usb1_lock)

            if not self.reverse_done.wait(timeout=REVERSE_TIMEOUT_S):
                self.get_logger().warn("Reverse timeout — forcing halt")
                self.bin_command_send(bytearray([CMD_HALT]), RassorDriverSubscriber.usb1, self.usb1_lock)

            self._capture_snapshot()
            self.publish_status("OBSTACLE: cleared")
        finally:
            self.obstacle_mode.clear()

    def _capture_snapshot(self):
        if self.cap is None or cv2 is None:
            self.get_logger().warn("No camera — skipping snapshot")
            return
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Camera read failed")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(SNAPSHOT_DIR, f"obstacle_{ts}.jpg")
        cv2.imwrite(path, frame)
        self.get_logger().info(f"Snapshot saved: {path}")
        self.publish_status(f"Snapshot: {path}")

    def shutdown(self):
        self._shutdown.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)

    rassor_driver_subscriber = RassorDriverSubscriber()

    try:
        rclpy.spin(rassor_driver_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        rassor_driver_subscriber.shutdown()
        RassorDriverSubscriber.usb1.close()
        if RassorDriverSubscriber.usb2 is not None:
             RassorDriverSubscriber.usb2.close()
        rassor_driver_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
