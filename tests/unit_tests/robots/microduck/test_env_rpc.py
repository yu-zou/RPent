# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline MicroDuck RPC and entrypoint contracts; no simulator imports."""

import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient, RpcError

RUN_ID = "52b9103c-9266-4815-ade2-f5ec5e00fd45"
META = {
    "robot": "microduck",
    "protocol": 1,
    "run_id": "52b9103c92664815ade2f5ec5e00fd45",
    "recording_fps": 25,
}


class Session:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.moves = []
        self.finishes = self.closes = 0
        self.error = None

    def move(self, **kwargs):
        self.moves.append(kwargs)
        if self.error:
            raise self.error
        return {}

    def finish(self):
        self.finishes += 1
        if self.error:
            raise self.error
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "video.mp4"
        path.write_bytes(b"video")
        return path

    def close(self):
        self.closes += 1


@pytest.fixture
def facade(tmp_path):
    from robots.microduck.env_server import MicroduckEnvFacade

    return MicroduckEnvFacade(Session(tmp_path / ".microduck"), run_id=RUN_ID)


def dispatch(facade, method, **kwargs):
    return facade._dispatch(method, (), kwargs)


def test_exact_routes_meta_and_builtin_lifecycle(facade):
    assert set(facade._rpc) == {"env.get_env_meta", "env.move", "env.finish"}
    assert dispatch(facade, "env.get_env_meta") == META
    assert dispatch(facade, "healthz") == {"status": "ok"}
    assert not facade._enable_sessions
    assert dispatch(facade, "shutdown") == {"ok": True}
    facade.close()
    facade.close()
    assert facade.session.closes == 1
    assert facade.session.finishes == 0


@pytest.mark.parametrize(
    "method",
    [
        "env.reset",
        "env.step",
        "env.chunk_step",
        "env.render_camera",
        "env.get_camera_meta",
        "env.get_action_spec",
        "session.register",
        "env.interrupt",
    ],
)
def test_unsupported_routes(facade, method):
    with pytest.raises(ValueError, match="unknown RPC"):
        dispatch(facade, method)


def test_move_defaults_empty_success_and_error(facade):
    assert dispatch(facade, "env.move") == {}
    assert facade.session.moves == [
        {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0}
    ]
    facade.session.error = RuntimeError("controller lost")
    with pytest.raises(RuntimeError, match="controller lost"):
        dispatch(facade, "env.move", vx=-0.4)
    assert len(facade.session.moves) == 2


def test_facade_never_leaks_session_observations(facade):
    facade.session.move = lambda **kw: {"state": b"private"}
    assert dispatch(facade, "env.move") == {}


def test_finish_cached_metadata_is_not_mutable(facade):
    result = dispatch(facade, "env.finish")
    assert result == {"video": ".microduck/video.mp4", "bytes": 5}
    result["video"] = "other"
    assert dispatch(facade, "env.finish") == {
        "video": ".microduck/video.mp4",
        "bytes": 5,
    }
    assert facade.session.finishes == 1


def test_finish_failure_cached_without_retry(facade):
    facade.session.error = RuntimeError("render failed")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="render failed"):
            dispatch(facade, "env.finish")
    assert facade.session.finishes == 1


@pytest.mark.parametrize("kind", ["outside", "empty", "symlink", "directory"])
def test_finish_rejects_wrong_or_incomplete_artifact(facade, tmp_path, kind):
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"private")
    path = facade.session.output_dir / "video.mp4"
    path.parent.mkdir()
    if kind == "empty":
        path.touch()
    elif kind == "symlink":
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    facade.session.finish = lambda: target if kind == "outside" else path
    with pytest.raises((ValueError, RuntimeError)):
        dispatch(facade, "env.finish")


@pytest.mark.parametrize(
    "first,second",
    [("env.move", "env.finish"), ("env.finish", "env.move"), ("env.move", "env.move")],
)
def test_mutations_exclusive(facade, first, second):
    entered, release, other = threading.Event(), threading.Event(), threading.Event()
    original_move, original_finish = facade.session.move, facade.session.finish

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_move(**kwargs) if first == "env.move" else original_finish()

    def next_operation(*args, **kwargs):
        other.set()
        return original_finish() if second == "env.finish" else original_move(**kwargs)

    if first == second:
        calls = iter([blocked, next_operation])
        facade.session.move = lambda **kw: next(calls)(**kw)
    else:
        setattr(facade.session, first.split(".")[1], blocked)
        setattr(facade.session, second.split(".")[1], next_operation)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(dispatch, facade, first)
        assert entered.wait(2)
        b = pool.submit(dispatch, facade, second)
        try:
            assert not other.wait(0.05)
        finally:
            release.set()
        a.result(2)
        b.result(2)
    assert other.is_set()


class RPC(RpcClient):
    def __init__(self, meta=None):
        super().__init__()
        self.meta = META.copy() if meta is None else meta
        self.calls = []
        self.error = None
        self.closes = 0

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, args, kwargs, timeout_s))
        if self.error:
            raise self.error
        if method == "env.get_env_meta":
            return self.meta
        return (
            {}
            if method == "env.move"
            else {"video": ".microduck/video.mp4", "bytes": 5}
        )

    def close(self):
        self.closes += 1


def test_client_no_reset_defaults_timeouts_and_idempotent_close():
    from robots.microduck.env_client import MicroduckEnvClient

    baseline = BaseEnvClient._TIMEOUT_S.copy()
    rpc = RPC()
    client = MicroduckEnvClient(rpc, run_id=RUN_ID)
    assert client.last_obs == {}
    assert rpc.calls == [("env.get_env_meta", (), None, 30.0)]
    assert client.move() == {}
    assert rpc.calls[-1] == (
        "env.move",
        (),
        {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0},
        40.0,
    )
    assert client.finish() == {"video": ".microduck/video.mp4", "bytes": 5}
    assert rpc.calls[-1] == ("env.finish", (), None, 340.0)
    client.close()
    client.close()
    assert rpc.closes == 1
    client._TIMEOUT_S["default"] = 1
    assert BaseEnvClient._TIMEOUT_S == baseline
    assert MicroduckEnvClient(RPC(), run_id=RUN_ID)._TIMEOUT_S["default"] == 30


@pytest.mark.parametrize(
    "meta",
    [
        {**META, "run_id": str(uuid4())},
        {**META, "extra": 1},
        {**META, "recording_fps": 30},
        {**META, "protocol": 2},
    ],
)
def test_client_rejects_nonexact_metadata(meta):
    from robots.microduck.env_client import MicroduckEnvClient

    with pytest.raises(AssertionError, match="env_meta mismatch"):
        MicroduckEnvClient(RPC(meta), run_id=RUN_ID)


def test_client_error_preserves_observation_and_never_retries():
    from robots.microduck.env_client import MicroduckEnvClient

    rpc = RPC()
    client = MicroduckEnvClient(rpc, run_id=RUN_ID)
    previous = client.last_obs
    rpc.error = RpcError("env.move", "partial move")
    with pytest.raises(RpcError, match="partial move"):
        client.move(vx=-0.4)
    assert client.last_obs is previous
    assert len(rpc.calls) == 2
    with pytest.raises(RpcError):
        client.finish()
    assert len(rpc.calls) == 3


@pytest.mark.parametrize("run_id", ["", "not-a-uuid", None])
def test_invalid_run_id_rejected_before_rpc_or_session(tmp_path, run_id):
    from robots.microduck.env_client import MicroduckEnvClient
    from robots.microduck.env_server import MicroduckEnvFacade

    rpc = RPC()
    with pytest.raises(ValueError, match="UUID"):
        MicroduckEnvClient(rpc, run_id=run_id)
    assert rpc.calls == []
    with pytest.raises(ValueError, match="UUID"):
        MicroduckEnvFacade(Session(tmp_path), run_id=run_id)


@pytest.mark.parametrize(
    "artifact",
    [
        "trajectory.bin",
        "trajectory.json",
        "trajectory.json.partial",
        "video.mp4",
        ".video-old.mp4",
        "controller.log",
        "renderer.log",
    ],
)
def test_entrypoint_refuses_artifact_collisions(monkeypatch, tmp_path, artifact):
    from robots.microduck import env_server

    private = tmp_path / ".microduck"
    private.mkdir()
    (private / artifact).touch()
    monkeypatch.setenv("MICRODUCK_RUN_ID", RUN_ID)
    monkeypatch.setattr(
        env_server,
        "MicroduckSession",
        lambda *_: pytest.fail("collision reached constructor"),
    )
    with pytest.raises(FileExistsError):
        env_server.main(["--output-dir", str(tmp_path)])


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_entrypoint_rejects_unsafe_private_directory(monkeypatch, tmp_path, kind):
    from robots.microduck import env_server

    monkeypatch.setenv("MICRODUCK_RUN_ID", RUN_ID)
    private = tmp_path / ".microduck"
    if kind == "symlink":
        private.symlink_to(tmp_path / "absent")
    else:
        private.touch()
    with pytest.raises(FileExistsError):
        env_server.main(["--output-dir", str(tmp_path)])


@pytest.mark.parametrize("interrupt", [None, "start", "construct", "fail"])
def test_entrypoint_ready_before_bind_and_signal_flags(
    monkeypatch, tmp_path, interrupt
):
    from robots.microduck import env_server

    monkeypatch.setenv("MICRODUCK_RUN_ID", RUN_ID)
    (tmp_path / "supervisor.json").touch()
    (tmp_path / "container.log").touch()
    private = tmp_path / ".microduck"
    private.mkdir()
    for name in ("runtime.json", "supervisor.log", "container.log", "unrelated.txt"):
        (private / name).touch()
    handlers, events = {}, []

    def install(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(env_server.signal, "signal", install)

    class StartupSession(Session):
        def __init__(self, path):
            assert signal.SIGTERM in handlers and signal.SIGINT in handlers
            assert path == tmp_path / ".microduck"
            super().__init__(path)
            events.append("construct")
            if interrupt == "construct":
                handlers[signal.SIGTERM](signal.SIGTERM, None)

        def start(self):
            events.append("start")
            if interrupt == "fail":
                raise RuntimeError("not ready")
            if interrupt == "start":
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            events.append("ready")

        def close(self):
            super().close()
            events.append("close")

    def serve(self, **kwargs):
        assert events[-1] == "ready"
        assert kwargs == {"transport": "http", "host": "0.0.0.0", "port": 8000}
        events.append("bind")

    monkeypatch.setattr(env_server, "MicroduckSession", StartupSession)
    monkeypatch.setattr(env_server.MicroduckEnvFacade, "serve", serve)
    if interrupt == "fail":
        with pytest.raises(RuntimeError, match="not ready"):
            env_server.main(["--output-dir", str(tmp_path)])
    else:
        env_server.main(["--output-dir", str(tmp_path)])
    assert events.count("close") == 1
    assert ("bind" in events) == (interrupt is None)


def test_entrypoint_requires_run_id(monkeypatch):
    from robots.microduck import env_server

    monkeypatch.delenv("MICRODUCK_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="UUID"):
        env_server.main([])


def test_http_roundtrip_preserves_errors_and_empty_observations(facade):
    from robots.microduck.env_client import MicroduckEnvClient
    from rpent.utils.rpc import make_rpc_client
    from rpent.utils.rpc.http_rpc import HttpRpcServer

    server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    rpc = make_rpc_client(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        client = MicroduckEnvClient(rpc, run_id=RUN_ID)
        assert client.move() == {}
        facade.session.error = ValueError("controller rejected move")
        with pytest.raises(RpcError, match="controller rejected move"):
            client.move()
        assert client.last_obs == {}
        with pytest.raises(RpcError, match="unknown RPC"):
            rpc.call("env.reset")
    finally:
        rpc.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
    assert not worker.is_alive()


def test_signal_during_blocked_start_cancels_without_binding(monkeypatch, tmp_path):
    from robots.microduck import env_server

    handlers, events = {}, []
    cancelled = threading.Event()
    monkeypatch.setenv("MICRODUCK_RUN_ID", RUN_ID)
    monkeypatch.setattr(
        env_server.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn)
    )

    class BlockedSession(Session):
        def start(self):
            events.append("starting")
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            assert cancelled.wait(2), "Signal did not interrupt startup"

        def close(self):
            assert threading.current_thread() is not threading.main_thread()
            events.append("close")
            cancelled.set()

    monkeypatch.setattr(env_server, "MicroduckSession", BlockedSession)
    monkeypatch.setattr(
        env_server.MicroduckEnvFacade,
        "serve",
        lambda *a, **kw: pytest.fail("bound during interrupted startup"),
    )
    env_server.main(["--output-dir", str(tmp_path)])
    assert events == ["starting", "close"]
