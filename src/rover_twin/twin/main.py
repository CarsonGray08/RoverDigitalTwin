"""Twin process: the rover in Isaac Sim, on the live terrain.

    $ISAAC_SIM_ROOT/python.sh -m rover_twin.twin.main --experience NEWTON_KIT [--run-dir RUN]

One loop owns the stage. It applies what the link delivered, terrain first,
then the pose the sim rover follows. Physics runs only in a rollout: between
them the sim rover is held where the real one is, since live tiles become
solid only at the reload before a rollout (it would fall through them). Validation
requests from the HTTP API run as jobs in the same loop, so the terrain never
changes under a rollout and the next one starts from the latest map; the
operator's view is drawn as a rollout runs, so a rehearsal plays on screen,
and captured for the dashboard (``GET /view.jpg``). The live pose is answered
at once, never behind a job: the drive loop stops the real rover on it.
"""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rover_twin import config
from rover_twin.link.client import LinkClient, Update
from rover_twin.link.messages import LivePose
from rover_twin.metrics.latency import Stopwatch
from rover_twin.mission import planner
from rover_twin.mission.scheduler.scheduler_core import MissionError
from rover_twin.twin import route
from rover_twin.twin.terrain import StageTerrain, TerrainMap
from rover_twin.twin.validation import Request, Result, SimRover, Validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "config" / "rover_twin.toml"
STALL_S = 0.05      # a loop iteration slower than this is a stall
IDLE_S = 0.002      # between updates, wait this long for a job
RENDER_HZ = 60.0    # the operator's view, drawn between physics steps (~9 ms a frame on the 5090)
REHEARSAL_RENDER_HZ = 30.0   # and during a rehearsal, where every frame drawn slows the rollout
POSE_HZ = 10.0      # the sim rover's pose, read for /observation


class Twin:
    def __init__(self, cfg: config.Config, rover: SimRover, terrain: StageTerrain, link: LinkClient,
                 clock=time.monotonic, view=None) -> None:
        self.cfg, self.rover, self.terrain, self.link, self.clock = cfg, rover, terrain, link, clock
        self.view = view                  # the viewport's capture for the dashboard, if any
        self.map = TerrainMap()
        self.validator = Validator(cfg, _Drawn(rover, lambda: self._draw(self._rehearsal_render_s)), self.map)
        self.jobs: queue.Queue = queue.Queue()
        self.live: LivePose | None = None
        self.watch = Stopwatch(clock)
        self._render_s = 1.0 / RENDER_HZ if cfg.twin.gui else 0.0
        self._rehearsal_render_s = 1.0 / REHEARSAL_RENDER_HZ if cfg.twin.gui else 0.0
        self._render_due = 0.0
        self._speedups: list[float] = []
        self._terrain_changed = False     # since physics last read the stage
        self.rover_pose, self._pose_due = rover.pose(), 0.0   # read here, served from other threads

    def tick(self) -> None:
        """One iteration: the latest updates, then a queued job, and a frame of
        the operator's view when one is due. Updates go first so a validation
        runs on the newest map."""
        start = self.clock()
        update = self.link.take()
        if update is not None:
            self._apply(update)
            for t in update.tile_arrivals:        # solid for the next rollout, which reloads first
                self.watch.since("collider_live", t)
        try:
            self._serve(*self.jobs.get(timeout=IDLE_S if update is None else 0.0))
            return
        except queue.Empty:
            pass
        wall = self.clock() - start
        self.watch.took("tick", wall)
        if wall > STALL_S:
            self.watch.count("stalls")
        self._draw(self._render_s)
        if self.clock() >= self._pose_due:
            self.rover_pose, self._pose_due = self.rover.pose(), self.clock() + 1.0 / POSE_HZ

    def submit(self, kind: str, payload=None) -> tuple[int, dict]:
        """From any thread: run a job in the loop and wait for its reply."""
        reply: queue.Queue = queue.Queue(maxsize=1)
        self.jobs.put((kind, payload, reply))
        return reply.get()

    def write_metrics(self, path: Path) -> None:
        if self._speedups:
            self.watch.values["validation_speedup_min"] = round(min(self._speedups), 2)
            self.watch.values["validation_speedup_median"] = round(statistics.median(self._speedups), 2)
        self.watch.write(path)

    def _draw(self, every_s: float) -> None:
        """A frame of the operator's view, if one is due at this rate."""
        if every_s and self.clock() >= self._render_due:
            t0 = self.clock()
            self.rover.render()
            if self.view is not None:
                self.view.frame_drawn()
            self.watch.took("render", self.clock() - t0)
            # From when it was due, not when it finished: a 9 ms frame would cut 60 Hz to 39.
            self._render_due = max(self._render_due + every_s, self.clock())

    def _apply(self, update: Update) -> None:
        self.map.apply(update)
        t0 = self.clock()
        self.terrain.apply(update)
        self.watch.took("stage_tiles", self.clock() - t0)
        self._terrain_changed |= update.reset or bool(update.patches)
        if update.mesh is not None:
            t0 = self.clock()
            self.terrain.show(update.mesh.mesh)
            self.watch.took("stage_mesh", self.clock() - t0)
        for t in update.tile_arrivals:
            self.watch.since("tiles_applied", t)
        if update.pose is not None:     # followed before published: other threads read self.live
            self.rover.follow(update.pose.pose, update.pose.z_m + self.rover.ride_height_m)
            self.live = update.pose
            self.watch.since("pose_applied", update.pose.t_arrival)

    def _serve(self, kind: str, payload, reply: queue.Queue) -> None:
        try:
            if kind in ("validate", "route"):
                answer = self._rehearse(payload).to_dict() if kind == "validate" else self._route(payload)
                self._return_to_live()
                reply.put((200, answer))
            elif kind == "route_costs":
                reply.put((200, route.costs(route.free_space(self.map, self.cfg), payload)))
            else:
                reply.put((200, self.map.dense(self.cfg.rover.max_step_m)))
        except Exception as exc:  # noqa: BLE001 - one bad job must not stop the twin
            reply.put((500, {"error": "twin_error", "reason": repr(exc)}))

    def _rehearse(self, request: Request) -> Result:
        if self._terrain_changed:
            t0 = self.clock()
            self.rover.reload()
            self.watch.took("physics_reload", self.clock() - t0)
            self._terrain_changed = False
        start = self.clock()
        result = self.validator.validate(request)
        wall = self.clock() - start
        self.watch.took("validation", wall)
        self._speedups.append(result.duration_s / max(wall, 1e-9))
        return result

    def _return_to_live(self) -> None:
        if self.live is not None:          # the sim rover goes back to the real one
            self.rover.follow(self.live.pose, self.live.z_m + self.rover.ride_height_m)

    def _route(self, request: route.RouteRequest) -> dict:
        """Plan a route to the goal, then rehearse its plans one after another,
        each from where the last one ended. docs/twin-api.md is the format."""
        t0 = self.clock()
        found = route.plan(route.free_space(self.map, self.cfg), self.cfg, request)
        self.watch.took("route_planning", self.clock() - t0)
        out = {"approved": found.found, "reason": found.reason, "rehearsed": False,
               "waypoints": [{"x": round(x, 4), "y": round(y, 4)} for x, y in found.waypoints],
               "length_m": round(found.length_m, 3), "turn_deg": found.turn_deg,
               "plans": list(found.plans), "rehearsals": []}
        if not found.found:
            out.update(failure_type=found.failure_type, failed_plan_index=None)
            return out
        if not request.rehearse:
            return out
        start = request.start
        for index, p in enumerate(found.plans):
            result = self._rehearse(Request.from_dict({"start_pose": _pose_dict(start), **p}))
            d = result.to_dict()
            out["rehearsals"].append({k: d[k] for k in ("approved", "reason", "failure_type", "failed_command_index",
                                                        "duration_s", "final_pose", "min_clearance_m") if k in d})
            if not result.approved:
                out.update(approved=False, failure_type=result.failure_type, failed_plan_index=index,
                           reason=f"plan {index}: {result.reason}")
                break
            start = result.final_pose
        out["rehearsed"] = True
        if out["approved"]:
            out["reason"] = found.reason + ", rehearsed"
        return out

    def plan_mission(self, request: planner.MissionRequest) -> tuple[int, dict]:
        """From an HTTP thread: the students' scheduler runs here, outside the
        loop, so it never holds up the live view; route costs and each leg's
        rehearsal queue in the loop like any request."""
        def ask(kind, payload):
            status, answer = self.submit(kind, payload)
            if status != 200:
                raise RuntimeError(answer.get("reason"))
            return answer

        try:
            return 200, planner.plan_mission(request, lambda points: ask("route_costs", points),
                                             lambda leg: ask("route", leg), self.cfg.limits.goal_tolerance_m)
        except MissionError as exc:
            return 400, {"error": "schema_error", "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one bad mission must not stop the twin
            return 500, {"error": "twin_error", "reason": repr(exc)}

    def rover_info(self) -> dict:
        """What a client needs to write commands and size a footprint."""
        rc, lim = self.cfg.rover, self.cfg.limits
        return {"distance_units_per_m": rc.distance_units_per_m, "speed_range": [lim.min_speed, lim.max_speed],
                "max_distance_units": lim.max_distance, "max_turn_deg": lim.max_turn_deg,
                "max_commands": lim.max_commands, "footprint_half_m": [rc.footprint_half_length_m,
                                                                        rc.footprint_half_width_m],
                "max_step_m": rc.max_step_m,
                "goal_tolerance": [lim.goal_tolerance_m, lim.goal_tolerance_yaw_deg]}

    def observation(self) -> dict:
        """From any thread: it reads only what the loop replaces whole."""
        pose, live = self.rover_pose, self.live
        return {
            "rover_pose": {"x": pose.x, "y": pose.y, "yaw_deg": pose.yaw_deg},
            "live_pose": None if live is None else {
                "x": live.pose.x, "y": live.pose.y, "yaw_deg": live.pose.yaw_deg, "rev": live.rev,
                "age_s": round(self.clock() - live.t_arrival, 3)},
            "terrain_tiles": len(self.map.patches),
            "mapper_connected": self.link.connected.is_set(),
        }


def _pose_dict(p) -> dict:
    return {"x": p.x, "y": p.y, "yaw_deg": p.yaw_deg}


class _Drawn:
    """The sim rover as validation drives it: each physics step also draws the
    operator's view when a frame is due, so the rehearsal is seen as it runs."""

    def __init__(self, rover: SimRover, draw) -> None:
        self._rover, self._draw = rover, draw

    def __getattr__(self, name: str):
        return getattr(self._rover, name)

    def step(self) -> None:
        self._rover.step()
        self._draw()


def serve(twin: Twin, host: str, port: int) -> ThreadingHTTPServer:
    """``GET /health`` (with the rover's units and limits), ``GET /observation``,
    ``GET /map``, ``POST /validate_plan``, ``POST /plan_route``, ``POST /route_costs``
    and ``POST /plan_mission``. docs/twin-api.md is the contract.
    ``GET /view.jpg`` is the operator's view, its age (s) in ``X-Age-S``."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._reply(200, {"status": "ok", "rover": twin.rover_info()})
            elif self.path == "/observation":
                self._reply(200, twin.observation())
            elif self.path == "/map":
                self._reply(*twin.submit("map"))
            elif self.path == "/view.jpg":
                frame = twin.view.latest() if twin.view is not None else None
                if frame is None:
                    self._reply(404, {"error": "no_view", "reason": "the view is off or not captured yet"})
                else:
                    self._send(200, frame[0], "image/jpeg", {"X-Age-S": f"{frame[1]:.2f}"})
            else:
                self._reply(404, {"error": "not_found"})

        def do_POST(self) -> None:
            parse = {"/validate_plan": ("validate", Request.from_dict),
                     "/plan_route": ("route", route.RouteRequest.from_dict),
                     "/route_costs": ("route_costs", _points),
                     "/plan_mission": ("mission", planner.MissionRequest.from_dict)}.get(self.path)
            if parse is None:
                self._reply(404, {"error": "not_found"})
                return
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request = parse[1](json.loads(body))
            except (ValueError, KeyError, TypeError) as exc:
                self._reply(400, {"error": "schema_error", "reason": str(exc)})
                return
            self._reply(*(twin.plan_mission(request) if parse[0] == "mission" else twin.submit(parse[0], request)))

        def _reply(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="twin-http", daemon=True).start()
    return server


def _points(data: dict) -> list[tuple[float, float]]:
    """``/route_costs``'s body: ``{"points": [{"x": .., "y": ..}, ...]}``."""
    points = [(route._number(p["x"]), route._number(p["y"])) for p in data["points"]]
    if not 1 <= len(points) <= route.MAX_POINTS:
        raise ValueError(f"points must hold 1 to {route.MAX_POINTS} points, got {len(points)}")
    return points


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experience", default="", help="the Newton kit file for SimulationApp")
    parser.add_argument("--run-dir", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    cfg = config.load(args.config)

    from rover_twin.twin.rover import IsaacRover, boot
    from rover_twin.twin.view import ViewportCapture

    app, stage = boot(cfg, args.experience)
    rover = IsaacRover(stage, cfg)
    if cfg.twin.gui and cfg.twin.camera:
        rover.watch(cfg.twin.camera)
    link = LinkClient(cfg.link.host, cfg.link.port)
    view = ViewportCapture(cfg.twin.view_hz) if cfg.twin.gui and cfg.twin.view_hz > 0 else None
    twin = Twin(cfg, rover, StageTerrain(stage, friction=cfg.twin.terrain_friction), link, view=view)
    server = serve(twin, cfg.twin.host, cfg.twin.port)
    metrics, next_write = args.run_dir / "twin_metrics.json", time.monotonic()
    print(f"twin: http://{cfg.twin.host}:{cfg.twin.port}, mapper link {cfg.link.port}", flush=True)
    try:
        while app.is_running():
            twin.tick()
            if time.monotonic() >= next_write:
                twin.write_metrics(metrics)
                next_write += 5.0
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()       # app.close() below ends the process before Python would print it
        raise
    finally:
        twin.write_metrics(metrics)
        server.shutdown()
        if view is not None:
            view.close()
        link.close()
        rover.close()
        app.close()


if __name__ == "__main__":
    main()
