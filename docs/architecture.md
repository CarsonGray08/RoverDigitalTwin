# Architecture

The live twin is two long-running processes connected by one session directory.

```text
iPhone RGB-D stream
        |
        v
digital_twin.mapper  --->  manifest.json, tiles/, rover_pose.json
                                      |
                                      v
                         digital_twin.isaac_service
                                      |
                         /observation, /validate_plan
                                      |
                          scan and drive commands
                                      |
                            rover HTTP controller
                                      |
                           ROS 2 serial bridge
                                      |
                              wheel firmware
```

The mapper owns the phone connection, nvblox TSDF, map tiles, and physical rover
pose. The Isaac process owns USD authoring, simulated command playback, and plan
approval. The drive command always uses `rover_pose.json` as physical pose truth;
the visible Isaac articulation may temporarily show validation playback.

The filesystem boundary is intentional because nvblox and Isaac use different
Python environments. Writes to the session contract are atomic so both processes
can run independently.

## Runtime defaults

- Online localization uses cuVSLAM `depth-gray`.
- A new launch always creates a new map session.
- The 360 scan fills the occluded floor beneath the rover and stabilizes the
  floor before driving.
- Mapping pauses during physical driving; localization continues.
- Armed dashboard turns and drives stop from fresh live-pose odometry.
- Rover output defaults to continual Mode-A commands; bounded Mode-B step
  commands remain selectable from the dashboard.
- The static room scene is hidden and non-collidable during a cold start.

The `mapper.py` and `isaac_service.py` entry points configure these defaults. The
dashboard exposes connection, rover command mode, map resolution, update rate,
and floor stabilization settings.
