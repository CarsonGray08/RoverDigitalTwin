# Lab setup

For an already configured workstation, go straight to [operations](operations.md).

## Workstation

Ubuntu 22.04 or 24.04, NVIDIA GPU, driver 570+, and Isaac Sim 6 installed:

```bash
scripts/setup_workstation.sh ~/isaac-sim-6.0.0
```

The script installs mapper dependencies, builds `.venv`, writes workstation
paths to `config/system.env`, and checks nvblox, Isaac's Python and tests.
If the lab recordings are present, it also checks replay against the reference
map. Set `ROVER_IP` in `config/system.env`. Runtime settings are in
`config/rover_twin.toml`.

## Rover and phone

- Build `robot/ros2_serial_bridge` in the rover's ROS 2 workspace. Use the
  matching Mode-B firmware in `robot/firmware`; the wheel Arduino must appear
  at `/dev/arduino_wheel`. Enable the rover's controller and serial-bridge services.
- The controller listens on port 5000 and publishes `[command, speed, distance]`
  as `std_msgs/Int16MultiArray` to `/<rover-name>/wheel_instructions`.
  The default name for `192.168.0.157` is `ip_192_168_0_157`; a custom
  `ROVER_NAME` must also be passed to the drive tool with `--rover-name`.
- The existing rover uses ROS 2 Humble on Ubuntu 22.04. Moving it to Ubuntu
  24.04 requires Jazzy, updated service paths and rebuilt workspaces.
- Build and install [LiveScanner](../ios/LiveScanner/README.md) on a LiDAR iPhone.
  Connect it to the workstation IP on port 9099, on the same access point.
