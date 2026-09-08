# LiveScanner (Theos / no-Mac build)

A native ARKit app that streams posed iPhone RGB-D frames to the host nvblox
mapper. It is
written in Objective-C so it can be **built on Linux with Theos** (no Xcode/Mac)
and installed with **Dadoum/Sideloader** using a free Apple ID.

## What it does

- Runs continuous RGB-D mode: JPEG, metric LiDAR depth, confidence, camera
  intrinsics, and an ARKit camera pose at up to 10 Hz for online cuVSLAM.
- Skips ARKit scene reconstruction in this mode. The host integrates depth and
  color into an nvblox TSDF; mesh anchors are unnecessary work and bandwidth.
- Keeps at most one RGB-D frame queued for TCP, favoring current geometry over a
  delayed backlog.
- Emits camera transforms in ARKit column-major order. The host applies the
  documented ARKit-to-CV and twin-frame transforms before TSDF integration.

The operational contract and host commands are in
[`README.md`](../../README.md).

## Prerequisites (Linux)

1. **Theos** installed and `$THEOS` exported — see https://theos.dev/docs/installation
2. **An iOS SDK** for Theos. The toolchain on Linux does not ship one; grab a
   recent `iPhoneOSxx.x.sdk` from https://github.com/theos/sdks into
   `$THEOS/sdks/`. Scene reconstruction needs the iOS 13.4+ ARKit headers, so
   use a modern SDK (e.g. iPhoneOS16.x).
   If Theos errors with `toolchain/linux/iphone/bin/clang: No such file or
   directory`, install the Linux iPhone toolchain:
   ```bash
   mkdir -p "$THEOS/toolchain"
   curl -L https://github.com/L1ghtmann/llvm-project/releases/latest/download/iOSToolchain-x86_64.tar.xz \
     | tar -xJf - -C "$THEOS/toolchain/"
   rm -rf ~/.cache/clang/ModuleCache
   ```
3. **Sideloader** (Dadoum) built/installed — needs `libimobiledevice`,
   `libplist`, `OpenSSL`, and the D toolchain (`ldc2`/`dmd`) to build.
4. **An Anisette server** for Apple auth (Sideloader fetches a dev cert like
   Xcode does):
   ```bash
   docker run -d -p 6969:6969 --restart=always \
     -v lib_cache:/opt/lib/ --name anisette dadoum/anisette-server:latest
   ```
5. A **free Apple ID** (does not need to be the one on your phone) and an
   **iPhone Pro with LiDAR**.

## Build the .ipa

```bash
cd ios/LiveScanner
make package      # PACKAGE_FORMAT=ipa in the Makefile emits a .ipa
```

The `.ipa` lands in `./packages/`.

Set the host IP either by editing `kDefaultHost` in
`LiveScanner/ScannerViewController.m` before building, or by typing it into the
app's text field at runtime and tapping **Connect**.

## Install it (Sideloader, Linux)

1. Enable **Developer Mode** on the iPhone (Settings → Privacy & Security →
   Developer Mode) — required on iOS 16+.
2. Plug the phone in (USB; `libimobiledevice` must see it).
3. Run Sideloader, sign in with the free Apple ID, and install the `.ipa`.
4. Trust the developer cert on the phone (Settings → General → VPN & Device
   Management).

## Caveats

- **Free account: 7-day expiry.** The app stops launching after ~7 days; re-sign
  / reinstall. Free accounts also cap you at ~3 sideloaded apps.
- **No storyboards / asset catalogs.** `ibtool`/`actool` are macOS-only, so the
  UI here is 100% programmatic and there is no app icon. Do not add a storyboard.
- On-device the screen shows the camera feed + ARKit's world-origin overlay; the
  fused mesh is rendered/colored on the **Isaac** side, not on the phone.
- Watch `ProcessInfo.thermalState` for long sessions; LiDAR capture and RGB-D
  streaming can heat the phone.
- If the app closes during scanning, reinstall the latest IPA from this folder:
  the streamer disables socket `SIGPIPE` and uses a conservative send throttle,
  but heavy LiDAR sessions can still hit thermal/memory limits.

## Run order with the host

```bash
# From the repository root, using the lab's mapper Python environment:
./scripts/live_twin.sh
hostname -I                       # the IP to type into the app

# phone: launch LiveScanner, enter that IP, tap Connect, and scan
# Isaac displays the map beneath /World/NvbloxMap in the configured rover scene.
```
