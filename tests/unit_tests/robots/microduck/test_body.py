# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Body adapter tests with the original boundaries replaced by CPU fakes."""

import importlib
import json
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def body_module(monkeypatch):
    module = importlib.import_module("robots.microduck.body")
    events = []

    class World:
        def __init__(self, scene, count):
            assert count == 1
            self.lock = threading.Lock()
            self.model = types.SimpleNamespace(
                opt=types.SimpleNamespace(timestep=0.005)
            )
            self.data = types.SimpleNamespace(time=0.0, qpos=np.zeros(7))
            self.bodies = []

        def step(self, times=1):
            # Real World.step owns the lock; an adapter outer lock deadlocks.
            with self.lock:
                assert times == 4
                self.data.time = round(self.data.time + 0.02, 8)
                self.data.qpos[0] += 1
                events.append(("step", threading.get_ident()))

    class Body:
        def __init__(self, world, index):
            assert index == 0
            self.world, self.trunk = world, 0
            self.released, self.torque_on = False, True

        def place(self, pose, trunk_z, offset_y):
            assert (pose, trunk_z, offset_y) == ({"joint": 1.0}, 0.12, 0)
            self.world.data.qpos[:] = [0, 0, trunk_z, 1, 0, 0, 0]
            events.append("SIT")

        def set_torque(self, enabled):
            pytest.fail("adapter must not enable or release held SIT")

    class Server:
        def __init__(self, address, handler):
            assert address == ("127.0.0.1", 0)
            self.server_address = ("127.0.0.1", 43210)
            self.stopped = threading.Event()

        def serve_forever(self):
            self.stopped.wait()

        def shutdown(self):
            self.stopped.set()

        def server_close(self):
            events.append("server-close")

    def get_state(model, data, state, signature):
        assert signature == 7
        assert not world_lock_available(data)
        state[:] = [data.time, data.qpos[0]]
        events.append(("snapshot", threading.get_ident(), data.time))

    worlds = []
    original_init = World.__init__

    def world_init(self, *args):
        original_init(self, *args)
        worlds.append(self)

    World.__init__ = world_init

    def world_lock_available(data):
        lock = next(world.lock for world in worlds if world.data is data)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
        return acquired

    monkeypatch.setitem(
        sys.modules, "mjlab_microduck", types.ModuleType("mjlab_microduck")
    )
    monkeypatch.setitem(
        sys.modules, "mjlab_microduck.sim", types.ModuleType("mjlab_microduck.sim")
    )
    monkeypatch.setitem(
        sys.modules,
        "mjlab_microduck.sim.body_server",
        types.SimpleNamespace(
            World=World,
            Body=Body,
            Server=Server,
            Handler=object,
            pose_table=lambda scene, key: (
                ({"joint": 1.0}, 0.12) if key == "SIT" else None
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        types.SimpleNamespace(
            mjtState=types.SimpleNamespace(mjSTATE_INTEGRATION=7),
            mj_stateSize=lambda model, sig: 2,
            mj_getState=get_state,
            mj_forward=lambda model, data: events.append("forward"),
            mju_quat2Mat=lambda result, quat: result.__setitem__(
                slice(None), np.eye(3).ravel()
            ),
        ),
    )
    return module, events


def wait_until(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.002)


def test_sit_timestep_atomic_diagnostics_and_cleanup(body_module, tmp_path):
    module, events = body_module
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    try:
        assert body.start() == 43210
        assert events[:2] == ["SIT", "forward"]
        sample = body.sample()
        assert set(sample) == {
            "trunk_xyz",
            "quat_wxyz",
            "gravity_xyz",
            "sim_time_s",
            "monotonic_time_s",
        }
        assert sample["trunk_xyz"][2] == 0.12
        assert sample["quat_wxyz"] == [1, 0, 0, 0]
        assert sample["gravity_xyz"] == [0, 0, -1]
        assert sample["monotonic_time_s"] > 0
    finally:
        body.close()
        body.close()
    assert "server-close" in events


def test_25hz_owned_snapshots_initial_thread_and_off_boundary_finish(
    body_module, tmp_path
):
    module, events = body_module
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    try:
        body.begin_recording()
        wait_until(
            lambda: (
                len([e for e in events if isinstance(e, tuple) and e[0] == "snapshot"])
                >= 2
            )
        )
        # At the intermediate .02 boundary, finish must wait for the next .04 sample.
        wait_until(lambda: round(body.sample()["sim_time_s"] * 50) % 2 == 1)
        requested = body.sample()["sim_time_s"]
        path = body.finish()
        assert path.name == "trajectory.json"
        meta = json.loads(path.read_text())
        rows = np.fromfile(tmp_path / "trajectory.bin", dtype="<f8").reshape(-1, 2)
        assert rows[-1, 0] > requested
        assert np.diff(rows[:, 0]) == pytest.approx(0.04)
        assert rows[0, 1] < rows[-1, 1]
        snaps = [e for e in events if isinstance(e, tuple) and e[0] == "snapshot"]
        assert all(e[1] != threading.get_ident() for e in snaps)
        assert len({e[1] for e in snaps}) == 1
        assert meta["scene"] == "scene.xml"
        stopped = body.sample()["sim_time_s"]
        time.sleep(0.03)
        assert body.sample()["sim_time_s"] == stopped
    finally:
        body.close()


def test_error_finish_skips_next_frame(body_module, tmp_path):
    module, events = body_module
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    try:
        body.begin_recording()
        count = len([e for e in events if isinstance(e, tuple) and e[0] == "snapshot"])
        body.finish(capture_final=False)
        assert (
            len([e for e in events if isinstance(e, tuple) and e[0] == "snapshot"])
            == count
        )
    finally:
        body.close()


def test_simulated_duration_cap_latches_failure_and_preserves_prefix(
    body_module, tmp_path, monkeypatch
):
    module, events = body_module
    monkeypatch.setattr(module, "MAX_DURATION_S", 0.08)
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    try:
        body.begin_recording()
        wait_until(lambda: not body._physics.is_alive())
        for _ in range(2):
            with pytest.raises(RuntimeError, match="simulated.*cap"):
                body.finish()
        assert str(body._failure) == (
            "Recording reached simulated-time cap (0.08 seconds)"
        )
        rows = np.fromfile(tmp_path / "trajectory.bin", dtype="<f8").reshape(-1, 2)
        assert rows[:, 0] == pytest.approx(
            [rows[0, 0], rows[0, 0] + 0.04, rows[0, 0] + 0.08]
        )
        meta = json.loads((tmp_path / "trajectory.json").read_text())
        assert meta["count"] == 3
        assert meta["sim_time_end_s"] == pytest.approx(rows[0, 0] + 0.08)
        assert body.sample()["sim_time_s"] == meta["sim_time_end_s"]
        assert meta["completion"] is False
        assert meta["failure"] == body._writer.failure
        assert meta["failure"] == (
            "Body recording failed: Recording reached simulated-time cap (0.08 seconds)"
        )
    finally:
        if body._writer.failure:
            with pytest.raises(RuntimeError, match="cleanup incomplete"):
                body.close()
        else:
            body.close()


def test_no_render_during_physics_or_live_writer(body_module, tmp_path, monkeypatch):
    module, events = body_module
    recording = importlib.import_module("robots.microduck.recording")
    monkeypatch.setattr(
        recording,
        "render_trajectory",
        lambda *args: pytest.fail("body must not render"),
    )
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    body.begin_recording()
    path = body.finish()
    assert path.exists()
    body.close()


def test_enqueue_outside_world_lock_and_failure_stops_physics(
    body_module, tmp_path, monkeypatch
):
    module, events = body_module
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    original = module.TrajectoryWriter.append
    calls = 0

    def append(writer, state, **kwargs):
        nonlocal calls
        assert body._world.lock.acquire(blocking=False)
        body._world.lock.release()
        calls += 1
        if calls == 2:
            raise RuntimeError("queue overflow")
        return original(writer, state, **kwargs)

    monkeypatch.setattr(module.TrajectoryWriter, "append", append)
    try:
        body.begin_recording()
        wait_until(lambda: body._stop.is_set())
        with pytest.raises(RuntimeError, match="queue overflow"):
            body.finish()
        stopped = body.sample()["sim_time_s"]
        time.sleep(0.03)
        assert body.sample()["sim_time_s"] == stopped
        assert not json.loads((tmp_path / "trajectory.json").read_text())["completion"]
    finally:
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            body.close()


def test_server_close_is_bounded(body_module, tmp_path, monkeypatch):
    module, events = body_module
    monkeypatch.setattr(module, "JOIN_TIMEOUT_S", 0.02)
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    release = threading.Event()
    entered = threading.Event()

    def close_server():
        entered.set()
        release.wait(0.3)

    body._server.server_close = close_server
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            body.close()
        assert entered.is_set()
        assert time.monotonic() - started < 0.2
    finally:
        release.set()


def test_physics_join_timeout_cannot_publish_success(
    body_module, tmp_path, monkeypatch
):
    module, events = body_module
    monkeypatch.setattr(module, "JOIN_TIMEOUT_S", 0.02)
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    body.begin_recording()
    entered, release = threading.Event(), threading.Event()
    original = body._world.step

    def step(times):
        entered.set()
        release.wait(1)
        original(times)

    body._world.step = step
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match="Physics thread join timeout"):
            body.finish()
        assert not (tmp_path / "trajectory.json").exists()
        with pytest.raises(RuntimeError, match="cleanup incomplete"):
            body.close()
    finally:
        release.set()
        body._physics.join(1)
    meta = json.loads((tmp_path / "trajectory.json").read_text())
    assert not meta["completion"] and "join timeout" in meta["failure"]


def test_initial_snapshot_timeout_is_terminal(body_module, tmp_path, monkeypatch):
    module, events = body_module
    monkeypatch.setattr(module, "JOIN_TIMEOUT_S", 0.02)
    body = module.MicroduckBody(Path("scene.xml"), output_dir=tmp_path)
    body.start()
    entered, release = threading.Event(), threading.Event()
    original = body._world.step

    def step(times):
        entered.set()
        release.wait(1)
        original(times)

    body._world.step = step
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match="Initial trajectory sample"):
            body.begin_recording()
    finally:
        release.set()
        body._physics.join(1)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        body.close()
    meta = json.loads((tmp_path / "trajectory.json").read_text())
    assert meta["count"] == 0 and not meta["completion"]
    assert "Initial trajectory sample" in meta["failure"]
