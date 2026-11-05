# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0

"""Real state/dispatcher and bounded finalization without Docker."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from rpent.dashboard.events import NullDashboardEventSink


class RecordingEnv:
    def __init__(self, root):
        self.root = root
        self.events = []
        self.move_error = None
        self.finish_error = None
        self.close_error = None
        self.metadata = {"video": ".microduck/video.mp4", "bytes": 5}

    def move(self, **kwargs):
        self.events.append(("move", kwargs))
        if self.move_error:
            raise RuntimeError(self.move_error)
        return {}

    def finish(self):
        self.events.append("finish")
        if self.finish_error:
            raise RuntimeError(self.finish_error)
        return self.metadata

    def close(self):
        self.events.append("close")
        if self.close_error:
            raise RuntimeError(self.close_error)


def make_toolkit(tmp_path, env=None):
    from robots.microduck.robot_spec import get_toolkit
    from rpent.utils.logging import init_output_dir

    init_output_dir(tmp_path)
    env = env or RecordingEnv(tmp_path)
    (tmp_path / ".microduck").mkdir(exist_ok=True)
    (tmp_path / ".microduck/video.mp4").write_bytes(b"video")
    config = SimpleNamespace(
        output_dir=tmp_path,
        prompt_vars={"memory_dir": str(tmp_path / ".microduck-memory")},
    )
    toolkit = get_toolkit(
        runtime_kwargs={"env": env, "output_dir": tmp_path, "run_id": "owned"},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    return toolkit, env


def test_initial_and_action_state_empty_defaults_and_common_tools(tmp_path):
    toolkit, env = make_toolkit(tmp_path)
    assert toolkit.state.latest_step == 0
    assert toolkit.state.latest_record().state == {}
    assert {t.name for t in toolkit.list_tools()} == {
        "robot_move",
        "finish",
        "read_text_file",
        "write_text_file",
        "list_dir",
    }
    for args in ({"duration_s": 5}, {"vx": -0.4}):
        result = toolkit.execute_tool("robot_move", args)
        assert result.data == {} and result.images == []
        assert toolkit.state.latest_record().state == {}
        assert toolkit.state.latest_record().result == {}
    assert env.events == [
        ("move", {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 5.0}),
        ("move", {"vx": -0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0}),
    ]
    toolkit.close()
    toolkit.close()
    assert env.events[-2:] == ["finish", "close"]
    assert (tmp_path / "microduck.mp4").read_bytes() == b"video"
    manifest = json.loads((tmp_path / "states.json").read_text())
    assert manifest["run_artifacts"] == ["microduck.mp4"]
    assert (tmp_path / ".microduck/video.mp4").read_bytes() == b"video"


@pytest.mark.parametrize(
    "args",
    [
        {"vx": True},
        {"vx": 0.40001},
        {"vx": -0.40001},
        {"vx": float("inf")},
        {"vx": -float("inf")},
        {"vy": -0.101},
        {"vyaw": 0.501},
        {"duration_s": 0},
        {"duration_s": 31},
        {"vx": float("nan")},
        {"vy": float("inf")},
    ],
)
def test_invalid_motion_never_dispatched(tmp_path, args):
    toolkit, env = make_toolkit(tmp_path)
    assert toolkit.execute_tool("robot_move", args).is_error
    assert env.events == []
    toolkit.close()


@pytest.mark.parametrize("vx", [0.4, -0.4, 0.15, -0.15])
def test_valid_velocity_boundaries_and_smaller_commands(tmp_path, vx):
    toolkit, env = make_toolkit(tmp_path)
    result = toolkit.execute_tool("robot_move", {"vx": vx})
    assert result.data == {} and not result.is_error
    assert env.events == [
        ("move", {"vx": vx, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0})
    ]
    toolkit.close()


def test_controller_errors_visible_and_recorded_no_retry(tmp_path):
    toolkit, env = make_toolkit(tmp_path)
    env.move_error = "controller disconnected after partial move"
    result = toolkit.execute_tool("robot_move", {})
    assert result.data["error"] == env.move_error
    assert toolkit.state.latest_record().result["error"] == env.move_error
    assert len(env.events) == 1
    toolkit.close()


def test_finished_metadata_from_real_rpc_facade_registers_video(tmp_path):
    from uuid import uuid4

    from robots.microduck.env_server import MicroduckEnvFacade

    toolkit, env = make_toolkit(tmp_path)
    session = SimpleNamespace(
        output_dir=tmp_path / ".microduck",
        finish=lambda: tmp_path / ".microduck/video.mp4",
    )
    facade = MicroduckEnvFacade(session, run_id=str(uuid4()))
    env.finish = facade.finish
    toolkit.close()
    assert (tmp_path / "microduck.mp4").read_bytes() == b"video"


@pytest.mark.parametrize(
    "failure",
    [
        "wrong",
        "missing",
        "partial",
        "bool",
        "zero",
        "symlink",
        "directory",
        "save_none",
        "finish",
    ],
)
def test_invalid_finalization_is_visible_preserves_staging_closes_once(
    tmp_path, monkeypatch, failure
):
    toolkit, env = make_toolkit(tmp_path)
    video = tmp_path / ".microduck/video.mp4"
    if failure == "wrong":
        env.metadata["video"] = "../video.mp4"
    elif failure == "missing":
        video.unlink()
    elif failure == "partial":
        env.metadata["bytes"] = 6
    elif failure == "bool":
        env.metadata["bytes"] = True
    elif failure == "zero":
        env.metadata["bytes"] = 0
    elif failure == "symlink":
        video.unlink()
        target = tmp_path.parent / (tmp_path.name + "-outside.mp4")
        target.write_bytes(b"video")
        video.symlink_to(target)
    elif failure == "directory":
        video.unlink()
        video.mkdir()
    elif failure == "save_none":
        monkeypatch.setattr(toolkit.state, "save", lambda *a, **kw: None)
    else:
        env.finish_error = "session/controller failed"
    env.close_error = "secondary RPC close failure"
    with pytest.raises((ValueError, RuntimeError, OSError)):
        toolkit.close()
    toolkit.close()
    assert env.events == ["finish", "close"]
    error = json.loads((tmp_path / ".microduck/finalization-error.json").read_text())
    assert "secondary RPC close failure" in str(error)
    if failure == "finish":
        assert "session/controller failed" in str(error)
    assert not (tmp_path / "microduck.mp4").exists()


def test_close_drains_active_motion_before_finishing(tmp_path):
    toolkit, env = make_toolkit(tmp_path)
    started, release = threading.Event(), threading.Event()

    def move(**kw):
        started.set()
        assert release.wait(5)
        env.events.append("move returned")

    env.move = move
    with ThreadPoolExecutor(max_workers=2) as pool:
        moving = pool.submit(toolkit.execute_tool, "robot_move", {})
        assert started.wait(5)
        closing = pool.submit(toolkit.close)
        assert "finish" not in env.events
        release.set()
        assert moving.result(timeout=5).data == {}
        closing.result(timeout=5)
    assert env.events == ["move returned", "finish", "close"]
