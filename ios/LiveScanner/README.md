# LiveScanner

Objective-C ARKit app streaming LiDAR depth, confidence, JPEG, camera pose
and IMU to the workstation. Requires a LiDAR-capable iPhone.

## Linux build and install

Install [Theos](https://theos.dev/docs/installation), its Linux iPhone toolchain,
and a recent [iOS SDK](https://github.com/theos/sdks). Export `THEOS`; put the
SDK under `$THEOS/sdks/`. The UI is programmatic because Linux lacks Apple's
storyboard and asset-catalog tools.

```bash
cd ios/LiveScanner
make package
```

Install the IPA from `packages/` using Dadoum/Sideloader, an Apple ID and an
Anisette server. Enable iPhone Developer Mode and trust the developer
certificate. Free-account installs expire after about seven days and need
re-signing.

## Connect

Start the host as in [lab operations](../../docs/operations.md). Enter the
workstation IP and port 9099 in the app, then tap **Connect**. Keep phone and
workstation on the same access point.

The host fuses the map; the phone displays its camera view. RGB-D streams at
up to 10 Hz with at most one frame queued. ARKit transforms use column-major
order; host frame conversion lives in `src/rover_twin/frames.py`.
