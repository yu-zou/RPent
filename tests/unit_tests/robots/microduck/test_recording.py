# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for owned state persistence and replay publication."""

import importlib
import json
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def recording():
    return importlib.import_module("robots.microduck.recording")


def test_owned_rows_and_frame_metadata(recording, tmp_path):
    writer = recording.TrajectoryWriter(tmp_path, state_size=2, signature=7)
    state = np.array([1.0, 2.0])
    writer.append(state, sim_time_s=3.0, monotonic_time_s=10.0)
    state[:] = [3, 4]
    writer.append(state, sim_time_s=3.04, monotonic_time_s=10.04)
    state[:] = 99
    path = writer.close()
    meta = json.loads(path.read_text())
    assert meta["count"] == 2
    assert meta["signature"] == 7
    assert meta["state_size"] == 2
    assert meta["dtype"] == "<f8"
    assert meta["fps"] == 25
    assert meta["state_span_s"] == 0.04
    assert meta["video_duration_s"] == 0.08
    assert meta["sim_time_start_s"] == 3.0
    assert meta["sim_time_end_s"] == 3.04
    assert meta["completion"] is True and meta["failure"] is None
    assert "source_ids" in meta and "scene" in meta
    assert np.fromfile(tmp_path / "trajectory.bin", dtype="<f8").tolist() == [
        1,
        2,
        3,
        4,
    ]
    assert writer.close() == path


def blocked_file(monkeypatch, tmp_path, *, fail=False):
    """Block only trajectory writes; metadata remains independently writable."""
    entered, release = threading.Event(), threading.Event()
    original = Path.open

    class Stream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            entered.set()
            assert release.wait(3)
            if fail:
                raise OSError("disk failed")
            return self.stream.write(data)

    def open_file(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        if path == tmp_path / "trajectory.bin" and args[0] == "xb":
            return Stream(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return entered, release


def test_queue_250_overflow_preserves_valid_prefix(recording, tmp_path, monkeypatch):
    entered, release = blocked_file(monkeypatch, tmp_path)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    try:
        writer.append(np.array([0.0]))
        assert entered.wait(1)
        for i in range(250):
            writer.append(np.array([float(i + 1)]))
        with pytest.raises(RuntimeError, match="queue"):
            writer.append(np.array([999.0]))
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="queue"):
        writer.close()
    meta = json.loads((tmp_path / "trajectory.json").read_text())
    assert meta["count"] == 251
    assert not meta["completion"] and "queue" in meta["failure"]
    assert (tmp_path / "trajectory.bin").stat().st_size == 251 * 8


def test_disk_failure_latches_and_labels_prefix(recording, tmp_path, monkeypatch):
    entered, release = blocked_file(monkeypatch, tmp_path, fail=True)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    writer.append(np.array([1.0]))
    assert entered.wait(1)
    release.set()
    with pytest.raises(RuntimeError, match="disk failed"):
        writer.close()
    with pytest.raises(RuntimeError):
        writer.append(np.array([2.0]))
    meta = json.loads((tmp_path / "trajectory.json").read_text())
    assert meta["count"] == 0 and not meta["completion"]


def test_blocked_drain_is_terminal(recording, tmp_path, monkeypatch):
    entered, release = blocked_file(monkeypatch, tmp_path)
    monkeypatch.setattr(recording, "DRAIN_TIMEOUT_S", 0.02)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    writer.append(np.array([1.0]))
    assert entered.wait(1)
    try:
        with pytest.raises(TimeoutError, match="drain"):
            writer.close()
        assert writer.alive
        assert not (tmp_path / "trajectory.json").exists()
        with pytest.raises(RuntimeError):
            writer.append(np.array([2.0]))
    finally:
        release.set()
        writer._thread.join(1)
    assert not json.loads((tmp_path / "trajectory.json").read_text())["completion"]


def test_success_metadata_only_published_after_worker_join(
    recording, tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    original = Path.write_text

    def write_metadata(path, text, *args, **kwargs):
        if path.name == "trajectory.json.partial":
            entered.set()
            assert release.wait(1)
        return original(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_metadata)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    writer.append(np.array([1.0]))
    result = []
    closer = threading.Thread(target=lambda: result.append(writer.close()))
    closer.start()
    try:
        assert entered.wait(1)
        assert writer.alive
        assert not (tmp_path / "trajectory.json").exists()
    finally:
        release.set()
        closer.join(1)
    assert json.loads(result[0].read_text())["completion"] is True


def test_partial_disk_write_truncates_to_valid_rows(recording, tmp_path, monkeypatch):
    original = Path.open

    class PartialStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, raw):
            self.stream.write(raw[:3])
            raise OSError("partial write")

    def open_file(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        return PartialStream(stream) if mode == "xb" else stream

    monkeypatch.setattr(Path, "open", open_file)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    writer.append(np.array([1.0]))
    with pytest.raises(RuntimeError, match="partial write"):
        writer.close()
    assert (tmp_path / "trajectory.bin").read_bytes() == b""
    assert json.loads((tmp_path / "trajectory.json").read_text())["count"] == 0


def test_metadata_timeout_never_advertises_live_success(
    recording, tmp_path, monkeypatch
):
    entered, release = threading.Event(), threading.Event()
    write_text, replace = Path.write_text, Path.replace
    monkeypatch.setattr(recording, "DRAIN_TIMEOUT_S", 0.02)

    def write_metadata(path, text, *args, **kwargs):
        if path.name == "trajectory.json.partial":
            entered.set()
            assert release.wait(1)
        return write_text(path, text, *args, **kwargs)

    publications = []

    def publish(path, target):
        if path.name == "trajectory.json.partial":
            publications.append(json.loads(path.read_text())["completion"])
        return replace(path, target)

    monkeypatch.setattr(Path, "write_text", write_metadata)
    monkeypatch.setattr(Path, "replace", publish)
    writer = recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    writer.append(np.array([1.0]))
    try:
        with pytest.raises(TimeoutError, match="drain"):
            writer.close()
        assert entered.is_set()
    finally:
        release.set()
        writer._thread.join(1)
    assert publications == [False]
    assert not json.loads((tmp_path / "trajectory.json").read_text())["completion"]


def test_existing_trajectory_refused(recording, tmp_path):
    (tmp_path / "trajectory.bin").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        recording.TrajectoryWriter(tmp_path, state_size=1, signature=7)
    assert (tmp_path / "trajectory.bin").read_bytes() == b"keep"


@pytest.fixture
def graphics(monkeypatch):
    events = []
    model = types.SimpleNamespace()

    class Renderer:
        def __init__(self, model, *, height, width):
            assert (height, width) == (480, 640)
            assert threading.current_thread() is threading.main_thread()

        def update_scene(self, data, *, camera):
            assert camera.trackbodyid == 0
            assert (camera.distance, camera.azimuth, camera.elevation) == (
                1.2,
                135,
                -20,
            )
            events.append("scene")

        def render(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            events.append("renderer-close")

    mujoco = types.SimpleNamespace(
        MjModel=types.SimpleNamespace(from_xml_path=lambda path: model),
        MjData=lambda model: object(),
        mj_stateSize=lambda model, signature: 2,
        mj_setState=lambda model, data, state, sig: events.append(tuple(state)),
        mj_forward=lambda model, data: events.append("forward"),
        mj_step=lambda *args: pytest.fail("replay must not step"),
        mj_name2id=lambda *args: 0,
        mjtObj=types.SimpleNamespace(mjOBJ_BODY=1),
        mjtCamera=types.SimpleNamespace(mjCAMERA_TRACKING=2),
        MjvCamera=types.SimpleNamespace,
        Renderer=Renderer,
    )

    class Encoder:
        def __init__(self, path):
            self.path = Path(path)

        def append_data(self, frame):
            events.append("frame")
            self.path.write_bytes(b"mp4")

        def close(self):
            events.append("encoder-close")

    def get_writer(path, **kwargs):
        assert kwargs == {"fps": 25, "codec": "libx264", "pixelformat": "yuv420p"}
        return Encoder(path)

    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    imageio = types.ModuleType("imageio")
    imageio.v2 = types.SimpleNamespace(get_writer=get_writer)
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio.v2)
    return events


def trajectory(recording, tmp_path):
    writer = recording.TrajectoryWriter(tmp_path, state_size=2, signature=7)
    writer.append(np.array([1.0, 2.0]))
    writer.append(np.array([3.0, 4.0]))
    return writer.close()


def test_replay_no_step_and_atomic_publish(recording, tmp_path, graphics):
    meta = trajectory(recording, tmp_path)
    output = tmp_path / "run.mp4"
    assert recording.render_trajectory(Path("scene.xml"), meta, output) == output
    assert output.read_bytes() == b"mp4"
    assert graphics.count("frame") == 2
    assert graphics[:3] == [(1.0, 2.0), "forward", "scene"]
    assert "encoder-close" in graphics and "renderer-close" in graphics
    with pytest.raises(FileExistsError):
        recording.render_trajectory(Path("scene.xml"), meta, output)


def test_publish_destination_is_always_absent_or_complete(
    recording, tmp_path, graphics, monkeypatch
):
    meta = trajectory(recording, tmp_path)
    output = tmp_path / "run.mp4"
    observations = []
    open_file, replace, link = Path.open, Path.replace, recording.os.link

    def observe():
        if output.exists():
            with open_file(output, "rb") as stream:
                content = stream.read()
            assert content == b"mp4"
            assert graphics[-2:] == ["encoder-close", "renderer-close"]
        else:
            content = None
        observations.append(content)

    def checked_open(path, *args, **kwargs):
        observe()
        stream = open_file(path, *args, **kwargs)
        try:
            observe()
        except BaseException:
            stream.close()
            raise
        return stream

    def checked_replace(path, target):
        observe()
        result = replace(path, target)
        observe()
        return result

    def checked_link(source, target, *args, **kwargs):
        observe()
        result = link(source, target, *args, **kwargs)
        observe()
        return result

    monkeypatch.setattr(Path, "open", checked_open)
    monkeypatch.setattr(Path, "replace", checked_replace)
    monkeypatch.setattr(recording.os, "link", checked_link)
    assert recording.render_trajectory(Path("scene.xml"), meta, output) == output
    observe()
    assert None in observations and b"mp4" in observations
    assert not list(tmp_path.glob(".*.mp4"))


def test_concurrent_destination_is_not_replaced(
    recording, tmp_path, graphics, monkeypatch
):
    meta = trajectory(recording, tmp_path)
    output = tmp_path / "run.mp4"
    renderer = sys.modules["mujoco"].Renderer
    close = renderer.close

    def concurrent_publish(self):
        close(self)
        assert not output.exists()
        output.write_bytes(b"concurrent complete video")

    monkeypatch.setattr(renderer, "close", concurrent_publish)
    with pytest.raises(FileExistsError):
        recording.render_trajectory(Path("scene.xml"), meta, output)
    assert output.read_bytes() == b"concurrent complete video"
    assert not list(tmp_path.glob(".*.mp4"))


def test_encoder_close_failure_never_publishes(
    recording, tmp_path, graphics, monkeypatch
):
    meta = trajectory(recording, tmp_path)
    imageio = sys.modules["imageio.v2"]
    original = imageio.get_writer

    def get_writer(*args, **kwargs):
        encoder = original(*args, **kwargs)

        def close():
            raise OSError("encoder close failed")

        encoder.close = close
        return encoder

    monkeypatch.setattr(imageio, "get_writer", get_writer)
    with pytest.raises(OSError, match="encoder close failed"):
        recording.render_trajectory(Path("scene.xml"), meta, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()
    assert not list(tmp_path.glob(".*.mp4"))
    assert "renderer-close" in graphics


def test_render_deadline_never_publishes(recording, tmp_path, graphics, monkeypatch):
    meta = trajectory(recording, tmp_path)
    monkeypatch.setattr(recording, "RENDER_TIMEOUT_S", 0)
    with pytest.raises(TimeoutError, match="300 seconds"):
        recording.render_trajectory(Path("scene.xml"), meta, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()
    assert "frame" not in graphics


@pytest.mark.parametrize("corruption", ["truncated", "extra", "count", "dtype", "nan"])
def test_corrupt_records_never_publish(recording, tmp_path, graphics, corruption):
    path = trajectory(recording, tmp_path)
    binary = tmp_path / "trajectory.bin"
    meta = json.loads(path.read_text())
    if corruption == "truncated":
        binary.write_bytes(binary.read_bytes()[:-1])
    elif corruption == "extra":
        binary.write_bytes(binary.read_bytes() + b"x")
    elif corruption == "count":
        meta["count"] = -1
    elif corruption == "dtype":
        meta["dtype"] = ">f8"
    else:
        np.array([float("nan"), 2, 3, 4], dtype="<f8").tofile(binary)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        recording.render_trajectory(Path("scene.xml"), path, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()
