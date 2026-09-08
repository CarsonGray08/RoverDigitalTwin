# Workstation and rover setup

## Host workstation

1. Install the NVIDIA driver, CUDA stack, `nvblox_torch`, and cuVSLAM environment
   used by the lab workstation.
2. Install Isaac Sim 6 with the Newton extensions used by `assets/rover.usd`.
3. Edit `config/system.env`, especially `ISAAC_SIM_ROOT` and `ROVER_IP`.
4. Create the host Python environment and run the tests:

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -e '.[dev]'
   pytest
   ```

`torch`, `nvblox_torch`, cuVSLAM, Isaac, and ROS 2 are platform installations;
they are not installed by `pip`. The ordinary unit suite does not require a GPU
or Isaac.

## Rover

The ROS 2 serial bridge is in `robot/ros2_serial_bridge`. Build it with the
rover's ROS 2 workspace and install the systemd service used on the rover. The
wheel Arduino must appear at `/dev/arduino_wheel`.

The matching Mode-B wheel firmware and its `DRV8825` driver are in
`robot/firmware`. Do not flash the older four-byte command sketch from the
paper-one archive; its byte order does not match this ROS bridge.

The host sends HTTP commands to the rover controller on port 5000. The controller
publishes `std_msgs/Int16MultiArray` to the serial bridge's ROS topic:

```text
/<ip-derived-rover-name>/wheel_instructions
[command, speed, distance]
```

The namespace for `192.168.0.157` is `ip_192_168_0_157`.

## iPhone

Build `ios/LiveScanner` with Theos, install it on a LiDAR-capable iPhone, and
enter the workstation IP and port shown by the dashboard. Phone and workstation
must be on the same access point.
