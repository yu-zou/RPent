# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0

"""Production CLI and API loop with a deterministic in-process model."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "finish_error,planner_error", [(False, False), (True, False), (True, True)]
)
def test_production_api_forward5_backward_default3_finish_video_before_stop(
    tmp_path, monkeypatch, finish_error, planner_error
):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from robots.microduck import get_robot_spec
    from rpent.cli import main as cli
    from rpent.planner.api_loop import ApiAgentLoop
    from tests.unit_tests.robots.microduck.test_toolkit import RecordingEnv

    env = RecordingEnv(tmp_path)
    if finish_error:
        env.finish_error = "controller rendering failed"
    (tmp_path / ".microduck").mkdir()
    (tmp_path / ".microduck/video.mp4").write_bytes(b"video")
    requests = []
    calls = [
        ("robot_move", {"vx": 0.4, "duration_s": 5}),
        ("robot_move", {"vx": -0.4}),
        (
            "finish",
            {
                "status": "success",
                "summary": "Requested commands executed; success unverified",
            },
        ),
    ]

    async def stream(messages, info):
        requests.append(messages)
        name, args = calls[len(requests) - 1]
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    def build(*a, **kw):
        assert kw["model"] == "provider/exact-model"
        return ApiAgentLoop(
            model=FunctionModel(stream_function=stream),
            dashboard_events=kw["dashboard_events"],
        )

    def stop():
        assert env.events[-2:] == ["finish", "close"]
        if not finish_error:
            assert (tmp_path / "microduck.mp4").read_bytes() == b"video"
            assert (
                "microduck.mp4"
                in json.loads((tmp_path / "states.json").read_text())["run_artifacts"]
            )
        env.events.append("stop")

    spec = replace(
        get_robot_spec(),
        init_runtime=lambda *a: (
            [SimpleNamespace(stop=stop)],
            {"env": env, "output_dir": tmp_path, "run_id": "owned"},
        ),
    )
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli, "build_planner", build)
    monkeypatch.setattr(
        "rpent.memory.MemoryManager.sync", lambda *a, **kw: pytest.fail("HF sync")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "microduck",
            "--task",
            "forward 5 seconds then backward",
            "--model",
            "provider/exact-model",
            "--output-dir",
            str(tmp_path),
            "--max-turns",
            "2" if planner_error else "4",
        ],
    )
    assert cli.main() == int(finish_error)
    assert env.events[:2] == [
        ("move", {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 5.0}),
        ("move", {"vx": -0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0}),
    ]
    assert env.events[-3:] == ["finish", "close", "stop"]
    transcript = json.loads((tmp_path / "transcript_microduck.json").read_text())
    assert len(requests) == (2 if planner_error else 3)
    assert (
        "controller rendering failed" in (transcript["error"] or "")
    ) is finish_error
    tools = [m for m in transcript["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tools[:2]] == ["{}", "{}"]
    manifest = json.loads((tmp_path / "states.json").read_text())
    assert len(manifest["steps"]) == 3
    assert all(s["state"] == {} and s["result"] == {} for s in manifest["steps"])
    assert all(s["artifacts"] == [] for s in manifest["steps"])
    assert all("image" not in str(m).lower() for m in tools)
