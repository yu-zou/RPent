# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""One owned, serialized MicroDuck controller and recorded body session."""

import hashlib
import itertools
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger(__name__)
ROOT = Path("/opt/microduck")
POLICIES = {
    "walk": "velstand.onnx",
    "sitstand": "alpha_sitstand.onnx",
    "ground_pick": "alpha_ground_pick.onnx",
    "kick_left": "ball_kick_left.onnx",
    "kick_right": "ball_kick_right.onnx",
    "roulade": "roulade.onnx",
}


def _terminal_exit() -> None:
    # Continuing to serve after a native/daemon worker failed to join is unsafe.
    os._exit(1)


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        process.wait(timeout=2)
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _healthy(value: dict[str, Any]) -> bool:
    hz = value.get("control_loop", {}).get("achieved_hz")
    return value.get("healthy") is True and _number(hz) and hz >= 45


def _standing(sample: dict[str, Any]) -> bool:
    return sample["trunk_xyz"][2] > 0.095 and sample["gravity_xyz"][2] < -0.9


class _RPC:
    """Bounded newline JSON-RPC; a socket belongs to exactly one reader."""

    def __init__(self, sock: socket.socket, ids: Any = None) -> None:
        self.sock = sock
        self.sock.settimeout(2)
        self.ids = ids if ids is not None else itertools.count(1)
        self.buffer = b""

    def _send(self, message: dict[str, Any], timeout_s: float = 2) -> None:
        self.sock.settimeout(timeout_s)
        self.sock.sendall(
            (json.dumps({"jsonrpc": "2.0", **message}, allow_nan=False) + "\n").encode()
        )

    def receive(self, *, deadline: float | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + 2 if deadline is None else deadline
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Controller reply exceeded two seconds")
            self.sock.settimeout(remaining)
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Controller socket closed")
            self.buffer += chunk
            if len(self.buffer) > 1_048_576:
                raise ValueError("Controller message exceeds size bound")
        line, self.buffer = self.buffer.split(b"\n", 1)
        result = json.loads(line)
        if not isinstance(result, dict) or result.get("jsonrpc") != "2.0":
            raise ValueError("Invalid controller JSON-RPC")
        return result

    def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout_s: float = 2
    ) -> dict[str, Any]:
        identity = next(self.ids)
        message = {"id": identity, "method": method}
        if params is not None:
            message["params"] = params
        deadline = time.monotonic() + min(2, timeout_s)
        self._send(message, min(2, timeout_s))
        reply = self.receive(deadline=deadline)
        if type(reply.get("id")) is not int or reply["id"] != identity:
            raise RuntimeError("Controller reply ID mismatch")
        if "error" in reply or not isinstance(reply.get("result"), dict):
            raise RuntimeError(f"Controller rejected {method}: {reply.get('error')}")
        return reply["result"]

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class MicroduckSession:
    """Own the simulator, controller, recording and offline renderer.

    Args:
        output_dir: Private session directory (normally /output/.microduck).
        build_info: Image provenance produced by the pinned image recipe.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        build_info: Path = Path("/opt/microduck/build-info.json"),
    ) -> None:
        self.output_dir, self.build_info = output_dir, build_info
        self._state = "NEW"
        self._operation = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._cancel, self._reader_stop = threading.Event(), threading.Event()
        self._failure: BaseException | None = None
        self._background_failure: Exception | None = None
        self._body: Any = None
        self._controller: subprocess.Popen | None = None
        self._renderer: subprocess.Popen | None = None
        self._rpc: _RPC | None = None
        self._subscription: _RPC | None = None
        self._reader: threading.Thread | None = None
        self._runtime_dir: Path | None = None
        self._log: Any = None
        self._diagnostic: dict[str, Any] | None = None
        self._diagnostic_at = 0.0
        self._monitor_after: float | None = None
        self._enable_at: float | None = None
        self._health_states: list[dict[str, Any]] = []
        self._ids = itertools.count(1)
        self._video: Path | None = None
        self._info: dict[str, Any] = {}
        self._sit_sample: dict[str, Any] = {}
        self._stand: dict[str, Any] = {}
        self._last_motion: dict[str, Any] = {}

    def _validate_image(self) -> None:
        self._info = json.loads(self.build_info.read_text())
        pins = json.loads(Path(__file__).with_name("dependencies.json").read_text())
        if self._info.get("protocol") != 1 or self._info.get("dependencies") != pins:
            raise ValueError("Image protocol/dependency provenance mismatch")
        self._scene = Path(self._info["scene"])
        ort = Path(self._info["ort_dylib_path"])
        for path in (self._scene, ort):
            if (
                not path.is_absolute()
                or not path.resolve().is_relative_to((ROOT / "rl").resolve())
                or not path.is_file()
            ):
                raise ValueError(
                    "Image scene/ORT path missing or outside pinned simulator"
                )
        if self._scene.name != "scene_backlash.xml":
            raise ValueError("Image scene is not scene_backlash.xml")
        current = ROOT / "policies/current"
        actual = {
            str(p.relative_to(current)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(current.rglob("*"))
            if p.is_file()
        }
        if (
            actual != self._info.get("policy_sha256")
            or not {"manifest.json", *POLICIES.values()} <= actual.keys()
        ):
            raise ValueError("Pinned policy inventory mismatch")
        for name in POLICIES.values():
            if not (current / name).stat().st_size:
                raise ValueError("Empty pinned policy")
        executable = ROOT / "runtime/bin/robotd"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Missing pinned executable robotd")

    def _make_body(self) -> Any:
        from robots.microduck.body import MicroduckBody

        return MicroduckBody(self._scene, output_dir=self.output_dir)

    def _launch_controller(self, port: int) -> None:
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="md-", dir="/tmp"))
        self._socket_path = self._runtime_dir / "r.sock"
        params = self._runtime_dir / "params.toml"
        lines = [
            "[control]",
            "hz = 50",
            "[policy]",
            "enabled = true",
            'mode = "walk"',
            'stand = "none"',
        ]
        lines.extend(
            f"{key} = {json.dumps(str(ROOT / 'policies/current' / name))}"
            for key, name in POLICIES.items()
        )
        params.write_text("\n".join(lines) + "\n")
        self._log = (self.output_dir / "controller.log").open("xb")
        self._controller = subprocess.Popen(
            [
                str(ROOT / "runtime/bin/robotd"),
                "--sim",
                f"127.0.0.1:{port}",
                "--params",
                str(params),
                "--socket",
                str(self._socket_path),
            ],
            env={
                **os.environ,
                "DUCK_RUNTIME_DIR": str(self._runtime_dir),
                "ORT_DYLIB_PATH": self._info["ort_dylib_path"],
            },
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def _new_rpc(self, *, timeout_s: float = 2) -> _RPC:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout_s)
            sock.connect(str(self._socket_path))
            return _RPC(sock, self._ids)
        except BaseException:
            sock.close()
            raise

    def _connect_controller(self) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self._check(background_only=True)
            if self._rpc is None:
                try:
                    self._rpc = self._new_rpc(
                        timeout_s=min(2, deadline - time.monotonic())
                    )
                except (FileNotFoundError, ConnectionRefusedError):
                    time.sleep(max(0.0, min(0.05, deadline - time.monotonic())))
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            health = self._rpc.request("robot.health", timeout_s=min(2, remaining))
            if _healthy(health):
                return
            time.sleep(max(0.0, min(0.05, deadline - time.monotonic())))
        raise TimeoutError("Controller socket/healthy 45Hz startup exceeded 30 seconds")

    def _subscribe(self) -> None:
        self._subscription = self._new_rpc()
        result = self._subscription.request("robot.subscribe", {"hz": 25})
        if (
            result.get("accepted") is not True
            or result.get("walk") != "velstand.onnx"
            or result.get("unavailable") is not None
        ):
            raise RuntimeError("Controller subscription/walk policy unavailable")
        self._reader = threading.Thread(
            target=self._read_states, name="microduck-diagnostics", daemon=True
        )
        self._reader.start()

    @staticmethod
    def _validate_state(state: dict[str, Any]) -> None:
        try:
            move, safety = state["move"], state["safety"]
            for key in ("requested", "applied"):
                vector = move[key]
                if (
                    not isinstance(vector, list)
                    or len(vector) != 3
                    or not all(_number(value) for value in vector)
                ):
                    raise ValueError("Invalid move vector")
            if "limited_by" in move:
                limited_by = move["limited_by"]
                if not isinstance(limited_by, list) or not all(
                    isinstance(value, str) for value in limited_by
                ):
                    raise ValueError("Invalid move limiter diagnostic")
            if state["policy"] is None:
                raise ValueError("Unavailable policy/move diagnostic")
            if safety["fallen"] is not False:
                raise ValueError("Controller reports fallen or unknown safety")
            if len(safety["gravity"]) != 3 or not all(
                _number(x) for x in safety["gravity"]
            ):
                raise ValueError("Unknown gravity diagnostic")
            if not _number(state["loop"]["hz"]):
                raise ValueError("Unknown loop frequency")
        except (KeyError, TypeError) as exc:
            raise ValueError("Unknown controller state schema") from exc

    def _read_states(self) -> None:
        try:
            while not self._reader_stop.is_set():
                try:
                    message = self._subscription.receive()
                except Exception:
                    if self._reader_stop.is_set():
                        return
                    raise
                if message.get("method") != "robot.state" or "id" in message:
                    raise ValueError("Unknown controller subscription message")
                state = message["params"]
                self._validate_state(state)
                at = time.monotonic()
                with self._diagnostic_lock:
                    self._health_states.append(
                        {"at": at, "hz": state["loop"]["hz"], "state": state}
                    )
                    if self._monitor_after is not None and at >= self._monitor_after:
                        if state["loop"]["hz"] < 45:
                            raise RuntimeError("Received controller loop below 45Hz")
                        if at - max(self._monitor_after, self._diagnostic_at) > 1:
                            raise RuntimeError(
                                "Controller diagnostic gap exceeded one second"
                            )
                    self._diagnostic, self._diagnostic_at = state, at
        except Exception as exc:
            if self._background_failure is None:
                self._background_failure = exc

    def _check(self, *, background_only: bool = False) -> None:
        if self._cancel.is_set():
            raise RuntimeError("MicroDuck session interrupted")
        if self._background_failure:
            raise RuntimeError(
                "Controller diagnostics failed"
            ) from self._background_failure
        if self._body is not None and self._body._failure:
            raise RuntimeError("Body recording failed") from self._body._failure
        if self._controller is not None and self._controller.poll() is not None:
            raise RuntimeError("Controller exited unexpectedly")
        if not background_only:
            with self._diagnostic_lock:
                if (
                    self._diagnostic is None
                    or time.monotonic() - self._diagnostic_at > 1
                ):
                    raise RuntimeError("Controller diagnostic state is stale or absent")
                self._validate_state(self._diagnostic)
                if self._diagnostic["loop"]["hz"] < 45:
                    raise RuntimeError("Controller diagnostic loop below 45Hz")

    def _accepted(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._rpc.request(method, params).get("accepted") is not True:
            raise RuntimeError(f"Controller rejected {method}")

    def _health(self) -> dict[str, Any]:
        health = self._rpc.request("robot.health")
        if not _healthy(health):
            raise RuntimeError("Controller unhealthy or achieved_hz below 45")
        return health

    def _wait_standing(self) -> None:
        begin = self._enable_at if self._enable_at is not None else time.monotonic()
        while time.monotonic() - begin <= 10:
            self._check(background_only=True)
            sample = self._body.sample()
            if _standing(sample) and sample["monotonic_time_s"] - begin <= 10:
                at = sample["monotonic_time_s"]
                self._check()
                self._stand = {"sample": sample, "elapsed_s": at - begin}
                self._monitor_after = at + 1
                self._health()
                self._check()
                return
            time.sleep(0.02)
        raise TimeoutError("MicroDuck did not stand within ten seconds")

    def _require_ready(self) -> None:
        if self._failure is not None:
            raise RuntimeError("MicroDuck session failed") from self._failure
        if self._state != "READY":
            raise RuntimeError(f"MicroDuck session is {self._state}, not READY")

    def start(self) -> None:
        """Record held SIT, enable the controller and wait for healthy standing."""
        with self._operation:
            if self._state != "NEW":
                raise RuntimeError("Session can only start once")
            self._state = "STARTING"
            try:
                self._validate_image()
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._body = self._make_body()
                port = self._body.start()
                self._body.begin_recording()
                self._sit_sample = self._body.sample()
                self._launch_controller(port)
                self._connect_controller()
                self._subscribe()
                self._enable_at = time.monotonic()
                self._accepted("robot.enable", {"on": True})
                self._wait_standing()
                self._state = "READY"
            except BaseException as exc:
                self._fail(exc)
                raise

    def move(
        self,
        vx: float = 0.4,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = 3.0,
    ) -> dict[str, Any]:
        """Refresh bounded velocity intent at 10Hz, then acknowledge stop.

        Args:
            vx: Forward velocity intent, default/absolute maximum 0.4 m/s.
            vy: Lateral velocity, absolute maximum 0.10 m/s.
            vyaw: Yaw velocity, absolute maximum 0.5 rad/s.
            duration_s: Monotonic duration in (0, 30] seconds.

        Returns:
            Empty observations, not a verified displacement claim.
        """
        for value, bound in ((vx, 0.4), (vy, 0.10), (vyaw, 0.5)):
            if not _number(value) or abs(value) > bound:
                raise ValueError(
                    "Velocity must be finite, nonboolean and within bounds"
                )
        if not _number(duration_s) or not 0 < duration_s <= 30:
            raise ValueError("duration_s must be finite, nonboolean and in (0, 30]")
        with self._operation:
            self._require_ready()
            try:
                self._check()
                health = self._health()
                self._state = "MOVING"
                # No request or sleep between this sample and first notification.
                before = self._body.sample()
                start = time.monotonic()
                deadline, tick = start + duration_s, 0
                try:
                    while time.monotonic() < deadline - 1e-9:
                        self._check()
                        self._rpc.notify(
                            "robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw}
                        )
                        tick = max(
                            tick + 1, math.floor((time.monotonic() - start) * 10) + 1
                        )
                        time.sleep(
                            max(
                                0.0,
                                min(deadline, start + tick * 0.1) - time.monotonic(),
                            )
                        )
                finally:
                    notification_end = time.monotonic()
                    # One attempt only, including socket-loss/interruption paths.
                    self._accepted("robot.stop")
                after = self._body.sample()
                self._check()
                requested = self._wait_zero()
                self._last_motion = {
                    "before": before,
                    "after": after,
                    "notification_start_s": start,
                    "notification_end_s": notification_end,
                    "health_before": health,
                    "health_after": self._health(),
                    "requested_after_stop": requested,
                }
                self._state = "READY"
                return {}
            except BaseException as exc:
                self._fail(exc)
                raise

    def _wait_zero(self) -> list[float]:
        # A post-ack state is necessary; a stale pre-move zero is not evidence.
        boundary = time.monotonic()
        while time.monotonic() - boundary < 2:
            self._check()
            with self._diagnostic_lock:
                requested = self._diagnostic["move"]["requested"].copy()
                if self._diagnostic_at >= boundary and all(
                    value == 0 for value in requested
                ):
                    return requested
            time.sleep(0.02)
        raise TimeoutError("Controller stop did not report fresh zero intent")

    def finish(self) -> Path:
        """Settle, drain stopped physics/writer, then render with a 300s bound."""
        with self._operation:
            if self._video is not None:
                return self._video
            self._require_ready()
            self._state = "FINALIZING"
            try:
                self._check()
                self._health()
                self._accepted("robot.stop")
                self._wait_zero()
                end = time.monotonic() + 0.5
                while time.monotonic() < end - 1e-9:
                    self._check()
                    time.sleep(min(0.02, end - time.monotonic()))
                self._health()
                metadata = self._body.finish()
                self._cleanup()
                if self._failure is not None:
                    raise RuntimeError(
                        "Cleanup failed; replay forbidden"
                    ) from self._failure
                output = self.output_dir / "video.mp4"
                with (self.output_dir / "renderer.log").open("xb") as render_log:
                    self._renderer = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "robots.microduck.recording",
                            "--scene",
                            str(self._scene),
                            "--metadata",
                            str(metadata),
                            "--output",
                            str(output),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=render_log,
                        stderr=subprocess.STDOUT,
                    )
                    deadline = time.monotonic() + 300
                    while True:
                        if self._cancel.is_set():
                            raise RuntimeError("MicroDuck rendering interrupted")
                        remaining = deadline - time.monotonic()
                        if remaining <= 1e-9:
                            raise subprocess.TimeoutExpired(self._renderer.args, 300)
                        try:
                            code = self._renderer.wait(timeout=min(0.1, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    if code != 0:
                        raise RuntimeError("Offline renderer failed")
                self._renderer = None
                if not output.is_file() or not output.stat().st_size:
                    raise RuntimeError("Renderer did not publish video.mp4")
                self._video, self._state = output, "CLOSED"
                return output
            except BaseException as exc:
                self._fail(exc)
                raise

    def _fail(self, exc: BaseException) -> None:
        if self._failure is None:
            self._failure = exc
        self._state = "FAILED"
        self._cleanup()

    def _cleanup(self) -> None:
        self._reader_stop.set()
        errors: list[Exception] = []
        body_timeout = False
        for rpc in (self._subscription, self._rpc):
            if rpc is not None:
                try:
                    rpc.close()
                except Exception as exc:
                    errors.append(exc)
        for process in (self._renderer, self._controller):
            if process is not None:
                try:
                    _terminate(process)
                except Exception as exc:
                    errors.append(exc)
        if self._reader is not None:
            try:
                self._reader.join(2)
            except RuntimeError as exc:
                errors.append(exc)
        # Joining the reader closes the finalization race: no diagnostic failure
        # received during the final health RPC or body drain may become success.
        if self._background_failure is not None:
            errors.append(self._background_failure)
        if self._body is not None:
            try:
                self._body.close()
            except Exception as exc:
                errors.append(exc)
                # Body.close also bounds a private server-shutdown helper that
                # is not exposed on the body. Its timeout is terminal as well.
                cause: BaseException | None = exc
                while cause is not None:
                    body_timeout |= isinstance(cause, TimeoutError)
                    cause = cause.__cause__
        workers = [self._reader]
        if self._body is not None:
            workers.extend(
                getattr(self._body, name, None) for name in ("_physics", "_serving")
            )
            writer = getattr(self._body, "_writer", None)
            if writer is not None:
                workers.append(writer._thread)
        stuck = body_timeout or any(
            worker is not None and worker.is_alive() for worker in workers
        )
        if errors or stuck:
            self._state = "FAILED"
            if self._failure is None:
                self._failure = (
                    errors[0] if errors else RuntimeError("Owned thread failed to join")
                )
            for error in errors:
                logger.error("MicroDuck cleanup failed: %s", error)
        if stuck:
            logger.critical(
                "Owned MicroDuck daemon thread is stuck; terminating process"
            )
            _terminal_exit()
            return
        if self._log is not None:
            self._log.close()
        if self._runtime_dir is not None:
            shutil.rmtree(self._runtime_dir, ignore_errors=True)

    def close(self) -> None:
        """Serialize emergency cleanup; it does not promise a video artifact."""
        self._cancel.set()
        with self._operation:
            self._cleanup()
            if self._state != "FAILED":
                self._state = "CLOSED"
