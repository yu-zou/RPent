# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Strict G1 numeric evaluation without a simulator."""

import copy
import math

import pytest

from robots.microduck.probe import _decode_video, _evaluate, _projection, _run


def sample(x=0.0, y=0.0, z=0.10, t=0.0, yaw=0.0):
    return {
        "trunk_xyz": [x, y, z],
        "quat_wxyz": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
        "gravity_xyz": [0, 0, -1],
        "sim_time_s": t,
        "monotonic_time_s": t,
    }


@pytest.fixture
def evidence():
    health = {"healthy": True, "control_loop": {"achieved_hz": 50}}
    return {
        "sit": sample(z=0.07),
        "stand": {"sample": sample(t=1), "elapsed_s": 1},
        "motions": [
            {
                "name": "forward",
                "notification_start_s": 2,
                "notification_end_s": 7,
                "before": sample(t=2),
                "after": sample(x=0.15, t=7),
                "health_before": health.copy(),
                "health_after": health.copy(),
                "requested_after_stop": [0.0, 0.0, 0.0],
            },
            {
                "name": "backward",
                "notification_start_s": 8,
                "notification_end_s": 11,
                "before": sample(x=0.15, t=8),
                "after": sample(x=0.06, t=11),
                "health_before": health.copy(),
                "health_after": health.copy(),
                "requested_after_stop": [0.0, 0.0, 0.0],
            },
        ],
        "holds": [
            [sample(t=1 + i), sample(t=1.5 + i), sample(t=2 + i)] for i in (0, 6, 10)
        ],
        "health_states": [{"at": i / 4, "hz": 50} for i in range(8, 49)],
        "health_window": [2.0, 12.0],
        "video": {
            "width": 640,
            "height": 480,
            "fps": 25,
            "frames": 301,
            "duration_s": 301 / 25,
            "decoded": True,
            "codec": "h264",
            "pixel_format": "yuv420p",
        },
        "trajectory": {
            "count": 301,
            "completion": True,
            "failure": None,
            "monotonic_time_end_s": 12.0,
        },
        "cleanup": True,
        "source_inventory": {"dependencies": {"controller": "pin"}},
    }


def test_successful_sequence_has_individual_verdicts(evidence):
    result = _evaluate(evidence)
    assert result["numeric_pass"]
    assert all(result["verdicts"].values())
    assert result["visual_inspection"] == "pending"


def test_delayed_stop_ack_does_not_add_a_pacing_limit(evidence):
    motion = evidence["motions"][0]
    motion["after"]["monotonic_time_s"] = 7.15
    motion["after"]["sim_time_s"] = 7.15
    result = _evaluate(evidence)
    assert result["numeric_pass"]
    assert result["timing"]["forward"]["wall_s"] == pytest.approx(5.15)
    assert result["timing"]["forward"]["notification_s"] == 5


def test_short_notifications_fail_even_with_correct_stop_inclusive_duration(evidence):
    evidence["motions"][0]["notification_end_s"] = 6.5
    result = _evaluate(evidence)
    assert not result["numeric_pass"]
    assert not result["verdicts"]["forward_duration"]


@pytest.mark.parametrize(
    "fault",
    [
        "sit",
        "stand",
        "noop",
        "wrong",
        "pace",
        "gap",
        "hz",
        "health",
        "drift",
        "stop",
        "video",
        "loss",
    ],
)
def test_numeric_gate_rejects_each_failure(evidence, fault):
    e = copy.deepcopy(evidence)
    if fault == "sit":
        e["sit"]["trunk_xyz"][2] = 0.10
    elif fault == "stand":
        e["stand"]["sample"]["trunk_xyz"][2] = 0.07
    elif fault == "noop":
        e["motions"][0]["after"]["trunk_xyz"][0] = 0
    elif fault == "wrong":
        e["motions"][1]["after"]["trunk_xyz"][0] = 0.25
    elif fault == "pace":
        e["motions"][0]["after"]["sim_time_s"] = 8
    elif fault == "gap":
        e["health_states"] = e["health_states"][::8]
    elif fault == "hz":
        e["health_states"][2]["hz"] = 44.9
    elif fault == "health":
        e["motions"][0]["health_after"]["healthy"] = False
    elif fault == "drift":
        e["holds"][0][-1]["trunk_xyz"][0] = 0.051
    elif fault == "stop":
        e["motions"][0]["requested_after_stop"][0] = 0.15
    elif fault == "video":
        e["video"]["decoded"] = False
    else:
        e["trajectory"]["failure"] = "queue overflow"
    assert not _evaluate(e)["numeric_pass"]


@pytest.mark.parametrize(
    "requested",
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
        [0.01, 0, 0],
        [0, -0.01, 0],
        [0, 0, 0.01],
    ],
)
def test_stop_gate_requires_three_finite_nonboolean_zero_list_components(
    evidence, requested
):
    evidence["motions"][0]["requested_after_stop"] = requested
    result = _evaluate(evidence)
    assert not result["numeric_pass"]
    assert result["verdicts"]["forward_stop"] is False


def test_rotated_heading_projection_and_gate(evidence):
    start = sample(yaw=math.pi / 2)
    assert _projection(start, sample(y=0.12)) == pytest.approx(0.12)
    for m in evidence["motions"]:
        for key in ("before", "after"):
            old = m[key]
            old["trunk_xyz"][1], old["trunk_xyz"][0] = old["trunk_xyz"][0], 0
            old["quat_wxyz"] = start["quat_wxyz"]
    assert _evaluate(evidence)["numeric_pass"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("width", 320),
        ("height", 240),
        ("fps", 24),
        ("duration_s", 99),
        ("frames", 300),
        ("codec", "mpeg4"),
        ("pixel_format", "yuv444p"),
    ],
)
def test_rejects_wrong_video_metadata(evidence, key, value):
    evidence["video"][key] = value
    assert not _evaluate(evidence)["numeric_pass"]


def test_decoder_decodes_every_frame_and_contact_sheet(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    import numpy as np

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_meta_data(self):
            return {
                "fps": 25.0,
                "duration": 0.16,
                "codec": "h264",
                "pix_fmt": "yuv420p(progressive)",
            }

        def __iter__(self):
            return iter([np.full((480, 640, 3), i, dtype=np.uint8) for i in range(4)])

    monkeypatch.setattr(imageio, "get_reader", lambda path: Reader())
    sheets = []
    monkeypatch.setattr(imageio, "imwrite", lambda path, pixels: sheets.append(pixels))
    result = _decode_video(
        tmp_path / "video.mp4",
        {"frame_times": [[i / 25, i] for i in range(4)]},
        [0, 1, 2, 3],
        tmp_path / "sheet.png",
    )
    assert result == {
        "decoded": True,
        "frames": 4,
        "width": 640,
        "height": 480,
        "fps": 25.0,
        "duration_s": 0.16,
        "codec": "h264",
        "pixel_format": "yuv420p",
    }
    assert sheets[0].shape == (480, 2560, 3)
    assert [int(sheets[0][0, i * 640, 0]) for i in range(4)] == [0, 1, 2, 3]


def test_probe_failure_persists_gate_and_restores_signals(tmp_path, monkeypatch):
    import json

    import robots.microduck.probe as probe

    events = []

    class FailedSession:
        _info = {"dependencies": {"controller": "pin"}}
        _state = "FAILED"
        _health_states = [{"at": 12.25, "hz": 40}]
        _background_failure = RuntimeError("Received controller loop below 45Hz")

        def __init__(self, output):
            pass

        def start(self):
            raise RuntimeError("unhealthy controller")

        def close(self):
            events.append("close")

    handlers = []
    monkeypatch.setattr(probe, "MicroduckSession", FailedSession)
    monkeypatch.setattr(
        probe.signal,
        "signal",
        lambda number, handler: handlers.append((number, handler)) or "previous",
    )
    result = _run(tmp_path)
    assert not result["numeric_pass"]
    assert result["verdicts"]["execution"] is False
    assert result["evidence"]["source_inventory"] == FailedSession._info
    assert result["evidence"]["health_states"] == FailedSession._health_states
    assert result["verdicts"]["diagnostics"] is False
    assert (
        json.loads((tmp_path / "gate.json").read_text())["error"]
        == "unhealthy controller"
    )
    assert events == ["close"] and len(handlers) == 4


@pytest.mark.parametrize("unhealthy_final", [False, True])
def test_fixed_probe_sequence_uses_pre_enable_sit_and_move_boundaries(
    tmp_path, monkeypatch, evidence, unhealthy_final
):
    import json

    import robots.microduck.probe as probe

    if unhealthy_final:
        evidence["trajectory"]["monotonic_time_end_s"] = 12.5
        evidence["health_states"].append({"at": 12.25, "hz": 40})
    events = []

    class Session:
        _sit_sample = evidence["sit"]
        _stand = evidence["stand"]
        _info = evidence["source_inventory"]
        _health_states = evidence["health_states"]
        _state = "NEW"
        _background_failure = None

        def __init__(self, output):
            self.output_dir = output

        def start(self):
            events.append("start")
            self._state = "READY"

        def move(self, **kwargs):
            events.append(kwargs)
            self._last_motion = evidence["motions"][
                len([e for e in events if isinstance(e, dict)]) - 1
            ]
            return {}

        def finish(self):
            self.output_dir.mkdir()
            (self.output_dir / "trajectory.json").write_text(
                json.dumps(evidence["trajectory"])
            )
            self._state = "CLOSED"
            events.append("finish")
            return self.output_dir / "video.mp4"

        def close(self):
            events.append("close")

    holds = iter(evidence["holds"])
    monkeypatch.setattr(probe, "MicroduckSession", Session)
    monkeypatch.setattr(probe, "_hold", lambda s: next(holds))
    monkeypatch.setattr(probe, "_decode_video", lambda *a: evidence["video"])
    monkeypatch.setattr(probe.time, "monotonic", lambda: 12.0)
    monkeypatch.setattr(probe.signal, "signal", lambda *a: None)
    result = _run(tmp_path)
    assert events == [
        "start",
        {"vx": 0.4, "duration_s": 5},
        {"vx": -0.4, "duration_s": 3},
        "finish",
        "close",
    ]
    assert result["numeric_pass"] is not unhealthy_final
    assert result["evidence"]["sit"] == evidence["sit"]
