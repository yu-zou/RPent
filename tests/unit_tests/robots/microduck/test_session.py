# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline checks of the owned session; no native dependencies."""

import json
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from robots.microduck import session as mod


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def diagnostic(*, moving=False):
    return {
        "move": {
            "requested": [0.15 if moving else 0.0, 0.0, 0.0],
            "applied": [0.054 if moving else 0.0, 0.0, 0.0],
        },
        "policy": "walk",
        "safety": {"fallen": False, "gravity": [0, 0, -1]},
        "loop": {"hz": 50},
    }


class RPC:
    accepted = True
    healthy = True

    def __init__(self):
        self.calls = []

    def request(self, method, params=None, **kwargs):
        self.calls.append((method, params))
        if method == "robot.health":
            return {"healthy": self.healthy, "control_loop": {"achieved_hz": 50}}
        return {"accepted": self.accepted}

    def notify(self, method, params):
        self.calls.append((method, params))

    def close(self):
        pass


class Body:
    _failure = None
    closed = False
    finished = False

    def sample(self):
        return {
            "trunk_xyz": [0, 0, 0.10],
            "gravity_xyz": [0, 0, -1],
            "monotonic_time_s": mod.time.monotonic(),
        }

    def finish(self):
        self.finished = True
        return Path("trajectory.json")

    def close(self):
        self.closed = True


@pytest.fixture
def ready(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mod.time, "monotonic", clock.monotonic)
    s = mod.MicroduckSession(tmp_path)
    s._state, s._rpc, s._body = "READY", RPC(), Body()
    s._diagnostic, s._diagnostic_at = diagnostic(), 0.0

    def sleep(seconds):
        clock.sleep(seconds)
        s._diagnostic_at = clock.now

    monkeypatch.setattr(mod.time, "sleep", sleep)
    return s, clock


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vx": True},
        {"vy": float("nan")},
        {"vyaw": float("inf")},
        {"vx": 0.40001},
        {"vx": -0.40001},
        {"vx": float("nan")},
        {"vx": float("inf")},
        {"vx": -float("inf")},
        {"vy": 0.101},
        {"vyaw": 0.501},
        {"duration_s": 0},
        {"duration_s": 30.01},
        {"duration_s": False},
    ],
)
def test_invalid_move_never_sends(ready, kwargs):
    s, _ = ready
    with pytest.raises(ValueError):
        s.move(**kwargs)
    assert s._rpc.calls == []
    assert s._state == "READY"


@pytest.mark.parametrize(
    "vx,duration,count", [(0.4, 5, 50), (-0.4, 3, 30), (0.15, 3, 30)]
)
def test_refresh_duration_stop_and_empty_success(ready, vx, duration, count):
    s, clock = ready
    assert s.move(vx=vx, duration_s=duration) == {}
    moves = [c for c in s._rpc.calls if c[0] == "robot.move"]
    assert len(moves) == count
    assert all(c[1] == {"vx": vx, "vy": 0.0, "vyaw": 0.0} for c in moves)
    assert clock.now == pytest.approx(duration)
    assert ("robot.stop", None) in s._rpc.calls
    assert s._state == "READY"
    assert s._last_motion["notification_start_s"] == 0
    assert s._last_motion["notification_end_s"] == pytest.approx(duration)


def test_defaults(ready):
    s, clock = ready
    s.move()
    assert clock.now == pytest.approx(3)
    assert ("robot.move", {"vx": 0.4, "vy": 0.0, "vyaw": 0.0}) in s._rpc.calls


@pytest.mark.parametrize(
    "fault", ["stop", "health", "stale", "unknown", "cap", "signal"]
)
def test_failures_latch_without_replay(ready, fault):
    s, clock = ready
    if fault == "stop":
        s._rpc.accepted = False
    elif fault == "health":
        s._rpc.healthy = False
    elif fault == "stale":
        clock.now = 2
    elif fault == "unknown":
        s._diagnostic = {"other": True}
    elif fault == "cap":
        s._body._failure = RuntimeError("cap")
    else:
        s._cancel.set()
    with pytest.raises(Exception):
        s.move(duration_s=0.2)
    assert s._state == "FAILED"
    count = len(s._rpc.calls)
    with pytest.raises(Exception):
        s.move()
    assert len(s._rpc.calls) == count
    assert s._body.closed


def test_socket_rpc_ids_and_stop_omits_params():
    client, server = socket.socketpair()
    rpc = mod._RPC(client)
    messages = []

    def serve():
        with server.makefile("rb") as stream:
            for _ in range(2):
                msg = json.loads(stream.readline())
                messages.append(msg)
                server.sendall(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": msg["id"],
                                "result": {"accepted": True},
                            }
                        )
                        + "\n"
                    ).encode()
                )
        server.close()

    thread = threading.Thread(target=serve)
    thread.start()
    rpc.request("robot.enable", {"on": True})
    rpc.request("robot.stop")
    rpc.close()
    thread.join(2)
    assert [m["id"] for m in messages] == [1, 2]
    assert "params" not in messages[1]


def test_socket_wrong_id_fails():
    client, server = socket.socketpair()
    rpc = mod._RPC(client)
    server.sendall(b'{"jsonrpc":"2.0","id":99,"result":{}}\n')
    with pytest.raises(RuntimeError, match="ID"):
        rpc.request("robot.health")
    rpc.close()
    server.close()


class Process:
    args = ["fake"]

    def __init__(self, timeouts=()):
        self.timeouts, self.events, self.returncode = list(timeouts), [], None

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.timeouts and self.timeouts.pop(0):
            raise subprocess.TimeoutExpired("fake", timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.events.append("term")

    def kill(self):
        self.events.append("kill")


def test_process_term_kill_reap_bounds():
    p = Process([True, False])
    mod._terminate(p)
    assert p.events == ["term", ("wait", 5), "kill", ("wait", 2)]


def test_finish_drains_before_renderer_and_caches(ready, monkeypatch):
    s, clock = ready
    s._scene = Path("scene.xml")
    seen = []

    def popen(argv, **kwargs):
        assert s._body.finished and s._body.closed
        assert clock.now >= 0.5
        assert argv[0] == mod.sys.executable
        assert kwargs["stdout"].name == str(s.output_dir / "renderer.log")
        assert kwargs["stderr"] == subprocess.STDOUT
        seen.append(argv)
        (s.output_dir / "video.mp4").write_bytes(b"video")
        return Process()

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    assert s.finish() == s.output_dir / "video.mp4"
    assert s.finish() == s.output_dir / "video.mp4"
    assert len(seen) == 1
    assert s._state == "CLOSED"


def test_render_timeout_is_cached_failure(ready, monkeypatch):
    s, clock = ready
    s._scene = Path("scene.xml")

    class Renderer(Process):
        def wait(self, timeout):
            if "term" in self.events:
                return super().wait(timeout)
            self.events.append(("wait", timeout))
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("renderer", timeout)

    p = Renderer([True, False])
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: p)
    with pytest.raises(subprocess.TimeoutExpired):
        s.finish()
    boundary = p.events.index("term")
    assert sum(event[1] for event in p.events[:boundary]) == pytest.approx(300)
    assert p.events[boundary:] == ["term", ("wait", 5), "kill", ("wait", 2)]
    with pytest.raises(Exception):
        s.finish()
    assert s._state == "FAILED"


def test_stuck_thread_uses_terminal_seam(ready, monkeypatch):
    s, _ = ready

    class Stuck:
        def join(self, seconds):
            assert seconds == 2

        def is_alive(self):
            return True

    s._reader = Stuck()
    exits = []
    monkeypatch.setattr(mod, "_terminal_exit", lambda: exits.append(True))
    s.close()
    assert exits == [True]
    assert s._state == "FAILED"


def test_move_close_are_serialized(ready, monkeypatch):
    s, _ = ready
    entered, release = threading.Event(), threading.Event()

    def sleep(seconds):
        entered.set()
        assert release.wait(2)
        s._cancel.set()

    monkeypatch.setattr(mod.time, "sleep", sleep)
    errors = []

    def move():
        try:
            s.move()
        except Exception as exc:
            errors.append(exc)

    mover = threading.Thread(target=move)
    mover.start()
    assert entered.wait(2)
    closer = threading.Thread(target=s.close)
    closer.start()
    assert not s._body.closed
    release.set()
    mover.join(2)
    closer.join(2)
    assert errors and s._body.closed


@pytest.mark.parametrize(
    "stage",
    ["validate", "body", "record", "spawn", "connect", "subscribe", "enable", "stand"],
)
def test_startup_failure_cleans_owned_resources(tmp_path, monkeypatch, stage):
    s, body = mod.MicroduckSession(tmp_path), Body()

    def fail(*args):
        raise RuntimeError(stage)

    body.start = fail if stage == "body" else lambda: 1234
    body.begin_recording = fail if stage == "record" else lambda: None
    monkeypatch.setattr(
        s, "_validate_image", fail if stage == "validate" else lambda: None
    )
    monkeypatch.setattr(s, "_make_body", lambda: body)
    monkeypatch.setattr(
        s, "_launch_controller", fail if stage == "spawn" else lambda port: None
    )

    def connect():
        s._rpc = RPC()
        if stage == "enable":
            s._rpc.accepted = False

    monkeypatch.setattr(
        s, "_connect_controller", fail if stage == "connect" else connect
    )
    monkeypatch.setattr(s, "_subscribe", fail if stage == "subscribe" else lambda: None)
    monkeypatch.setattr(s, "_wait_standing", fail if stage == "stand" else lambda: None)
    with pytest.raises(RuntimeError):
        s.start()
    assert s._state == "FAILED"
    if stage != "validate":
        assert body.closed


def test_recording_and_sit_sample_precede_enable(tmp_path, monkeypatch):
    s, body, events = mod.MicroduckSession(tmp_path), Body(), []
    body.start = lambda: 123
    body.begin_recording = lambda: events.append("record")
    body.sample = lambda: events.append("sit") or {"trunk_xyz": [0, 0, 0.07]}
    monkeypatch.setattr(s, "_validate_image", lambda: None)
    monkeypatch.setattr(s, "_make_body", lambda: body)
    monkeypatch.setattr(s, "_launch_controller", lambda port: None)
    monkeypatch.setattr(s, "_connect_controller", lambda: setattr(s, "_rpc", RPC()))
    monkeypatch.setattr(s, "_subscribe", lambda: None)
    monkeypatch.setattr(s, "_wait_standing", lambda: events.append("stand"))
    s.start()
    assert events == ["record", "sit", "stand"]
    assert s._sit_sample["trunk_xyz"][2] == 0.07
    assert s._state == "READY"
    s.close()


@pytest.mark.parametrize("component", range(3))
def test_stop_requires_fresh_zero_intent(ready, component):
    s, _ = ready
    s._diagnostic["move"]["requested"][component] = 0.15
    with pytest.raises(TimeoutError, match="zero intent"):
        s.move(duration_s=0.1)
    assert s._state == "FAILED"


@pytest.mark.parametrize("key", ["requested", "applied"])
def test_controller_move_vectors_accept_finite_numeric_lists(key):
    state = diagnostic()
    state["move"][key] = [0.15, -1, 0.5]
    mod.MicroduckSession._validate_state(state)


def test_rejected_walk_state_without_limiter_is_recorded(ready):
    s, _ = ready
    # Exact validator-consumed fields from the executed rejected-state payload.
    state = {
        "move": {"applied": [0.054, 0.0, 0.0], "requested": [0.15, 0.0, 0.0]},
        "policy": "walk",
        "loop": {"hz": 49.98579232262315, "missed": 0},
        "safety": {
            "fallen": False,
            "gain": 200,
            "gravity": [
                0.016262831513628905,
                0.007048968406935125,
                -0.9998429038381766,
            ],
            "limp": False,
            "picked_up": False,
        },
    }

    class Subscription(RPC):
        def receive(self):
            s._reader_stop.set()
            return {"method": "robot.state", "params": state}

    s._subscription = Subscription()
    s._read_states()
    assert s._background_failure is None
    assert s._diagnostic == state
    assert s._health_states[-1]["hz"] == 49.98579232262315


@pytest.mark.parametrize("limited_by", [[], ["deadman"], ["deadman", "other"]])
def test_controller_limiter_accepts_string_lists(limited_by):
    state = diagnostic()
    state["move"]["limited_by"] = limited_by
    mod.MicroduckSession._validate_state(state)


@pytest.mark.parametrize(
    "limited_by",
    [None, "deadman", {}, (), 0, False, [None], [1], [False], ["deadman", []]],
)
def test_controller_limiter_rejects_malformed_optional_value(limited_by):
    state = diagnostic(moving=True)
    state["move"]["limited_by"] = limited_by
    with pytest.raises(ValueError):
        mod.MicroduckSession._validate_state(state)


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_null_policy_still_rejects_unlimited_motion(missing):
    state = diagnostic(moving=True)
    if missing:
        del state["policy"]
    else:
        state["policy"] = None
    with pytest.raises(ValueError):
        mod.MicroduckSession._validate_state(state)


def test_move_checks_moving_diagnostics_without_limiter(ready, monkeypatch):
    s, clock = ready
    notify, request = s._rpc.notify, s._rpc.request

    def move(method, params):
        notify(method, params)
        s._diagnostic = diagnostic(moving=True)

    def stop(method, params=None):
        result = request(method, params)
        if method == "robot.stop":
            s._diagnostic = diagnostic()
        return result

    def sleep(seconds):
        clock.sleep(seconds)
        s._diagnostic_at = clock.now
        s._check()

    s._rpc.notify, s._rpc.request = move, stop
    monkeypatch.setattr(mod.time, "sleep", sleep)
    assert s.move(duration_s=0.2) == {}
    assert s._state == "READY"
    assert s._last_motion["requested_after_stop"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("key", ["requested", "applied"])
@pytest.mark.parametrize(
    "vector",
    [
        {"vx": 0, "vy": 0, "vyaw": 0},
        [],
        [0, 0],
        [0, 0, 0, 0],
        (0, 0, 0),
        [False, 0, 0],
        [0, True, 0],
        [0, 0, False],
        [float("nan"), 0, 0],
        [0, float("inf"), 0],
        [0, 0, -float("inf")],
        ["0", 0, 0],
        None,
    ],
)
def test_controller_move_vectors_reject_unknown_schema(key, vector):
    state = diagnostic()
    state["move"][key] = vector
    with pytest.raises(ValueError):
        mod.MicroduckSession._validate_state(state)


def test_stop_preserves_post_ack_list_evidence(ready, monkeypatch):
    s, clock = ready
    request = s._rpc.request
    acknowledged = False

    def stop(method, params=None):
        nonlocal acknowledged
        result = request(method, params)
        if method == "robot.stop":
            acknowledged = True
            # Pre-ack zero must not satisfy the post-ack diagnostic boundary.
            s._diagnostic_at = clock.now - 0.01
        return result

    def sleep(seconds):
        clock.sleep(seconds)
        if acknowledged:
            s._diagnostic_at = clock.now

    s._rpc.request = stop
    monkeypatch.setattr(mod.time, "sleep", sleep)
    s.move(duration_s=0.1)
    requested = s._last_motion["requested_after_stop"]
    assert requested == [0.0, 0.0, 0.0]
    assert requested is not s._diagnostic["move"]["requested"]
    assert clock.now == pytest.approx(0.12)


def test_pre_ack_zero_without_fresh_state_cannot_confirm_stop(ready, monkeypatch):
    s, clock = ready
    monkeypatch.setattr(mod.time, "sleep", clock.sleep)
    with pytest.raises(RuntimeError, match="stale"):
        s.move(duration_s=0.1)
    assert s._state == "FAILED"
    assert [c[0] for c in s._rpc.calls].count("robot.stop") == 1


def test_notification_loss_stops_once_and_never_replays(ready):
    s, _ = ready
    sends = []

    def notify(method, params):
        sends.append(params)
        raise ConnectionError("lost")

    s._rpc.notify = notify
    with pytest.raises(ConnectionError):
        s.move()
    assert len(sends) == 1
    assert [c for c in s._rpc.calls if c[0] == "robot.stop"] == [("robot.stop", None)]


def test_outside_ready_does_not_move(tmp_path):
    s = mod.MicroduckSession(tmp_path)
    with pytest.raises(RuntimeError, match="READY"):
        s.move()


def test_body_cap_prevents_render_and_caches_failure(ready, monkeypatch):
    s, _ = ready
    s._body._failure = RuntimeError("cap")
    rendered = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: rendered.append(a))
    with pytest.raises(RuntimeError, match="Body"):
        s.finish()
    with pytest.raises(RuntimeError):
        s.finish()
    assert not rendered and s._body.closed


def test_unknown_subscription_and_unavailable_walk_fail(tmp_path, monkeypatch):
    s = mod.MicroduckSession(tmp_path)
    rpc = RPC()
    monkeypatch.setattr(s, "_new_rpc", lambda: rpc)
    with pytest.raises(RuntimeError, match="unavailable"):
        s._subscribe()
    assert s._reader is None
    s.close()


def test_reader_uses_separate_socket_and_latches_malformed_state(tmp_path):
    s = mod.MicroduckSession(tmp_path)
    client, server = socket.socketpair()
    s._subscription = mod._RPC(client)
    s._rpc = RPC()
    server.sendall(
        b'{"jsonrpc":"2.0","method":"robot.state","params":{"unknown":true}}\n'
    )
    s._read_states()
    assert isinstance(s._background_failure, ValueError)
    assert s._rpc.calls == []
    s.close()
    server.close()


@pytest.mark.parametrize("fault", ["hz", "gap"])
def test_received_unhealthy_state_latches_even_if_next_state_recovers(ready, fault):
    s, clock = ready
    s._monitor_after = 0
    clock.now = 2
    state = diagnostic()
    if fault == "hz":
        state["loop"]["hz"] = 40
        s._diagnostic_at = 1.9

    class Subscription:
        calls = 0

        def receive(self):
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("Must latch unhealthy state before reading again")
            return {"jsonrpc": "2.0", "method": "robot.state", "params": state}

    s._subscription = Subscription()
    s._read_states()
    assert isinstance(s._background_failure, RuntimeError)
    s._subscription = None


def test_null_startup_health_waits_and_total_startup_is_bounded(ready, monkeypatch):
    s, clock = ready
    values = [None, 44, 50]
    s._rpc.request = lambda *a, **kw: {
        "healthy": True,
        "control_loop": {"achieved_hz": values.pop(0)},
    }
    s._connect_controller()
    assert clock.now == pytest.approx(0.1)
    s._rpc.request = lambda *a, **kw: {
        "healthy": False,
        "control_loop": {"achieved_hz": None},
    }
    with pytest.raises(TimeoutError, match="30 seconds"):
        s._connect_controller()
    assert clock.now < 30.2


def test_seated_vertical_is_not_standing_and_timeout_is_bounded(ready):
    s, clock = ready
    s._body.sample = lambda: {"trunk_xyz": [0, 0, 0.07], "gravity_xyz": [0, 0, -1]}
    with pytest.raises(TimeoutError, match="ten seconds"):
        s._wait_standing()
    assert clock.now <= 10.03


def test_robotd_argv_private_toml_and_environment(tmp_path, monkeypatch):
    import os

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    s = mod.MicroduckSession(tmp_path)
    s._info = {"ort_dylib_path": "/pinned/ort.so"}
    seen = []
    monkeypatch.setattr(
        mod.subprocess, "Popen", lambda argv, **kw: seen.append((argv, kw)) or Process()
    )
    s._launch_controller(1234)
    argv, kwargs = seen[0]
    assert argv == [
        "/opt/microduck/runtime/bin/robotd",
        "--sim",
        "127.0.0.1:1234",
        "--params",
        str(s._runtime_dir / "params.toml"),
        "--socket",
        str(s._runtime_dir / "r.sock"),
    ]
    assert os.stat(s._runtime_dir).st_mode & 0o777 == 0o700
    assert len(str(s._runtime_dir / "r.sock")) < 100
    config = tomllib.loads((s._runtime_dir / "params.toml").read_text())
    assert set(config) == {"control", "policy"}
    assert config["control"] == {"hz": 50}
    assert config["policy"]["stand"] == "none"
    assert config["policy"]["walk"] == "/opt/microduck/policies/current/velstand.onnx"
    assert kwargs["env"]["ORT_DYLIB_PATH"] == "/pinned/ort.so"
    s.close()
    assert not s._runtime_dir.exists()


@pytest.fixture
def image(tmp_path, monkeypatch):
    import hashlib
    import os

    root = tmp_path / "image"
    monkeypatch.setattr(mod, "ROOT", root)
    scene = root / "rl/scene_backlash.xml"
    ort = root / "rl/.venv/lib/ort.so"
    exe = root / "runtime/bin/robotd"
    for path in (scene, ort, exe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
    os.chmod(exe, 0o700)
    current = root / "policies/current"
    current.mkdir(parents=True)
    for name in ["manifest.json", *mod.POLICIES.values()]:
        (current / name).write_bytes(b"data")
    info = {
        "protocol": 1,
        "dependencies": json.loads(
            Path(mod.__file__).with_name("dependencies.json").read_text()
        ),
        "scene": str(scene),
        "ort_dylib_path": str(ort),
        "policy_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in current.iterdir()
        },
    }
    info_path = root / "build-info.json"
    info_path.write_text(json.dumps(info))
    return mod.MicroduckSession(tmp_path / "output", build_info=info_path), info


def test_valid_image_inventory(image):
    s, _ = image
    s._validate_image()


@pytest.mark.parametrize(
    "fault", ["pin", "protocol", "scene", "ort", "policy", "executable"]
)
def test_pinned_image_validation_precedes_body(image, fault):
    s, info = image
    if fault == "pin":
        info["dependencies"] = {}
    elif fault == "protocol":
        info["protocol"] = 2
    elif fault == "scene":
        info["scene"] = "/etc/passwd"
    elif fault == "ort":
        info["ort_dylib_path"] = "/etc/passwd"
    elif fault == "policy":
        (mod.ROOT / "policies/current/velstand.onnx").write_bytes(b"changed")
    else:
        (mod.ROOT / "runtime/bin/robotd").unlink()
    s.build_info.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        s.start()
    assert s._body is None and s._state == "FAILED"


def test_socket_loss_and_two_second_answer_deadline():
    client, server = socket.socketpair()
    rpc = mod._RPC(client)
    assert client.gettimeout() == 2
    server.close()
    with pytest.raises((ConnectionError, OSError)):
        rpc.request("robot.health")
    rpc.close()


def test_failed_cleanup_never_returns_normal_finish(ready, monkeypatch):
    s, _ = ready
    s._scene = Path("scene.xml")
    s._body.close = lambda: (_ for _ in ()).throw(RuntimeError("cleanup"))
    rendered = []
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: rendered.append(a))
    with pytest.raises(RuntimeError, match="replay forbidden"):
        s.finish()
    assert not rendered


def test_body_shutdown_timeout_terminates_owner_via_seam(ready, monkeypatch):
    s, _ = ready

    def close():
        raise RuntimeError("cleanup incomplete") from TimeoutError(
            "Body server shutdown timeout"
        )

    s._body.close = close
    exits = []
    monkeypatch.setattr(mod, "_terminal_exit", lambda: exits.append(True))
    s.close()
    assert exits == [True]


def test_rejected_stop_not_retried(ready):
    s, _ = ready
    s._rpc.accepted = False
    with pytest.raises(RuntimeError, match="robot.stop"):
        s.move(duration_s=0.1)
    assert [c[0] for c in s._rpc.calls].count("robot.stop") == 1


def test_answer_deadline_includes_send_time(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(mod.time, "monotonic", clock.monotonic)

    class SlowSocket:
        def settimeout(self, value):
            self.timeout = value

        def sendall(self, data):
            clock.sleep(1.5)

        def recv(self, count):
            assert self.timeout == pytest.approx(0.5)
            raise TimeoutError("reply")

    rpc = mod._RPC(SlowSocket())
    with pytest.raises(TimeoutError):
        rpc.request("robot.health")


def test_startup_health_request_cannot_overrun_thirty_seconds(ready):
    s, clock = ready
    deadlines = []

    def health(method, params=None, *, timeout_s=2):
        deadlines.append(timeout_s)
        clock.sleep(timeout_s)
        return {"healthy": False, "control_loop": {"achieved_hz": None}}

    s._rpc.request = health
    with pytest.raises(TimeoutError):
        s._connect_controller()
    assert clock.now <= 30.000001
    assert all(0 < timeout <= 2 for timeout in deadlines)


@pytest.mark.parametrize("failure", ["exit", "missing"])
def test_renderer_failure_never_publishes_success(ready, monkeypatch, failure):
    s, _ = ready
    s._scene = Path("scene.xml")
    p = Process()
    if failure == "exit":
        p.wait = lambda timeout: 1
        p.returncode = 1
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: p)
    with pytest.raises(RuntimeError):
        s.finish()
    assert s._state == "FAILED" and s._video is None


def test_spawned_child_is_reaped_on_connect_failure(tmp_path, monkeypatch):
    s = mod.MicroduckSession(tmp_path)
    body = Body()
    body.start, body.begin_recording = lambda: 1234, lambda: None
    process = Process()
    monkeypatch.setattr(s, "_validate_image", lambda: None)
    s._info = {"ort_dylib_path": "/pinned/ort.so"}
    monkeypatch.setattr(s, "_make_body", lambda: body)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: process)

    def connect():
        raise RuntimeError("socket")

    monkeypatch.setattr(s, "_connect_controller", connect)
    with pytest.raises(RuntimeError):
        s.start()
    assert process.events == ["term", ("wait", 5)]
    assert body.closed and s._log.closed and not s._runtime_dir.exists()


@pytest.mark.parametrize("extra", [{}, {"unavailable": None}])
def test_valid_subscription_keeps_reader_separate_and_records_state(
    tmp_path, monkeypatch, extra
):
    s = mod.MicroduckSession(tmp_path)
    s._rpc = RPC()

    class Subscription(RPC):
        def request(self, method, params=None):
            assert method == "robot.subscribe" and params == {"hz": 25}
            return {"accepted": True, "walk": "velstand.onnx", **extra}

        def receive(self):
            s._reader_stop.set()
            return {"jsonrpc": "2.0", "method": "robot.state", "params": diagnostic()}

    subscription = Subscription()
    monkeypatch.setattr(s, "_new_rpc", lambda: subscription)
    s._subscribe()
    s._reader.join(2)
    assert s._diagnostic == diagnostic()
    assert len(s._health_states) == 1 and s._rpc.calls == []
    s.close()


def test_cancel_during_render_is_polled_and_child_reaped(ready, monkeypatch):
    s, _ = ready
    s._scene = Path("scene.xml")

    class Renderer(Process):
        def wait(self, timeout):
            if "term" in self.events:
                return super().wait(timeout)
            assert timeout <= 0.1
            self.events.append(("wait", timeout))
            s._cancel.set()
            raise subprocess.TimeoutExpired("renderer", timeout)

    p = Renderer()
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: p)
    with pytest.raises(RuntimeError, match="interrupted"):
        s.finish()
    assert p.events[-2:] == ["term", ("wait", 5)]
    assert s._state == "FAILED"


def test_finish_rejects_unhealthy_controller(ready, monkeypatch):
    s, _ = ready
    s._scene = Path("scene.xml")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: Process())
    s._rpc.healthy = False
    with pytest.raises(RuntimeError, match="unhealthy"):
        s.finish()
    assert not s._body.finished and s._state == "FAILED"


def test_unstarted_reader_does_not_prevent_body_cleanup(ready):
    s, _ = ready
    s._reader = threading.Thread(target=lambda: None)
    s.close()
    assert s._body.closed and s._state == "FAILED"


@pytest.mark.parametrize("unavailable", [False, "", [], {}, "policy missing"])
def test_subscription_rejects_every_non_null_unavailable(
    tmp_path, monkeypatch, unavailable
):
    s = mod.MicroduckSession(tmp_path)
    rpc = RPC()
    rpc.request = lambda *a: {
        "accepted": True,
        "walk": "velstand.onnx",
        "unavailable": unavailable,
    }
    monkeypatch.setattr(s, "_new_rpc", lambda: rpc)
    with pytest.raises(RuntimeError, match="unavailable"):
        s._subscribe()
    assert s._reader is None
    s.close()


@pytest.mark.parametrize("stage", ["health", "drain", "in_flight", "reader_join"])
def test_terminal_diagnostic_failure_forbids_replay_and_is_cached(
    ready, monkeypatch, stage
):
    s, clock = ready
    s._scene = Path("scene.xml")
    s._monitor_after = 0
    rendered = []

    class Subscription(RPC):
        def receive(self):
            if stage == "in_flight":
                # Shutdown starts while this already-issued receive completes.
                s._reader_stop.set()
            state = diagnostic()
            state["loop"]["hz"] = 40
            return {"method": "robot.state", "params": state}

    s._subscription = Subscription()
    if stage == "health":
        health = s._health

        def final_health():
            if clock.now >= 0.5:
                s._read_states()
            return health()

        monkeypatch.setattr(s, "_health", final_health)
    elif stage in ("drain", "in_flight"):
        finish = s._body.finish

        def drain():
            s._read_states()
            return finish()

        s._body.finish = drain
    else:

        class Reader:
            def join(self, timeout):
                # A receive already in flight completes while cleanup joins.
                s._background_failure = RuntimeError(
                    "Received controller loop below 45Hz"
                )

            def is_alive(self):
                return False

        s._reader = Reader()

    monkeypatch.setattr(
        mod.subprocess, "Popen", lambda *a, **k: rendered.append(a) or Process()
    )
    with pytest.raises(RuntimeError):
        s.finish()
    assert s._state == "FAILED" and s._video is None and not rendered
    if stage != "reader_join":
        assert s._health_states[-1]["hz"] == 40
    failure = s._failure
    with pytest.raises(RuntimeError, match="failed"):
        s.finish()
    assert s._failure is failure


def test_standing_sample_not_health_round_trip_sets_deadline_and_warmup(ready):
    s, clock = ready
    s._enable_at = 0.0
    clock.now = 9.9
    s._diagnostic_at = 9.9

    def health():
        clock.sleep(0.2)

    s._health = health
    s._wait_standing()
    assert s._stand["elapsed_s"] == pytest.approx(9.9)
    assert s._monitor_after == pytest.approx(10.9)
    assert clock.now == pytest.approx(10.1)


def test_enable_ack_latency_is_in_standing_deadline(ready, monkeypatch):
    s, clock = ready
    s._state = "NEW"
    s._body.start = lambda: 1234
    s._body.begin_recording = lambda: None
    monkeypatch.setattr(s, "_validate_image", lambda: None)
    monkeypatch.setattr(s, "_make_body", lambda: s._body)
    monkeypatch.setattr(s, "_launch_controller", lambda port: None)
    monkeypatch.setattr(s, "_connect_controller", lambda: None)
    monkeypatch.setattr(s, "_subscribe", lambda: None)

    def enable(method, params):
        clock.sleep(0.2)
        return {"accepted": True}

    s._rpc.request = enable
    s._health = lambda: None
    s.start()
    assert s._stand["elapsed_s"] == pytest.approx(0.2)
    assert s._monitor_after == pytest.approx(1.2)
    s.close()


def test_move_notification_duration_excludes_stop_ack(ready):
    s, clock = ready
    request = s._rpc.request

    def delayed_stop(method, params=None):
        if method == "robot.stop":
            clock.sleep(0.15)
            s._diagnostic_at = clock.now
        return request(method, params)

    s._rpc.request = delayed_stop
    s.move(duration_s=5)
    assert s._last_motion["notification_end_s"] == pytest.approx(5)
    assert s._last_motion["after"]["monotonic_time_s"] == pytest.approx(5.15)
