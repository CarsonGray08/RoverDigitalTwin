# Field operation

## Start

```bash
./scripts/dashboard.sh
```

Open the printed URL. Leave the arm switch off during setup.

1. Launch the live twin.
2. Connect the iPhone app to the displayed workstation address on port 9099.
3. Start a rehearsal 360 scan and verify the expected four turn commands.
4. Clear the rover area, enter the rover IP, arm, and run the physical 360.
5. Wait for map readiness, under-rover fill, and tracking mode.
6. Rehearse a validated drive before arming it.

The scan command (`python -m digital_twin.scan`) starts and stops the phone
scan by default, including during a dry motion rehearsal; pass `--no-auto-scan`
when checking commands without changing the phone stream or map. The dashboard
always uses fresh live-pose odometry to stop armed turns and drives. When using
the command-line entry points directly, add `--telemetry-stop` for the same
behavior.

The **Rover command mode** setting defaults to `continual`. This publishes the
direction command until live odometry reaches the target and sends STOP. The
`step` option retains the bounded step-command path for comparison or recovery.

Floor stabilization is enabled by default in **Runtime settings**. Disable it
only for a controlled comparison, and restart the live twin after changing it.
The mapper entry point sets the stabilization band to 0.04 m.

## Safety behavior

- Real motion requires the explicit arm switch or `--arm`.
- Every drive command is validated in Isaac before publication.
- Armed dashboard motion always uses fresh-pose odometry stopping.
- STOP bypasses confirmation and is published during interruption or failure.
- The operator can stop each managed job from the dashboard.

## Shutdown

Stop active scan or drive jobs first, then stop the live twin. The mapper saves
the current map before exiting. Runtime data lives below `MAP_ROOT` and is not
part of the source repository.

## Fast checks

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/observation
curl http://127.0.0.1:8090/healthz
```
