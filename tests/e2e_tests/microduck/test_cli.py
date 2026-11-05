# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Offline contracts and explicitly opted-in, owned-container CLI checks."""

import json
import math
import os
import subprocess
import sys
import time
import uuid

import pytest


def _script():
    return (
        ("robot_move", {"duration_s": 5.0}),
        ("robot_move", {"vx": -0.4}),
        ("finish", {"status": "stuck", "summary": "Open-loop sequence executed"}),
    )


def _require_timeout(available):
    if not available:
        raise RuntimeError("Opt-in Microduck CLI E2E requires pytest-timeout (600s)")


@pytest.fixture
def live(pytestconfig):
    if os.environ.get("RPENT_MICRODUCK_E2E") != "1":
        pytest.skip("Set RPENT_MICRODUCK_E2E=1 after preparing the local image")
    _require_timeout(pytestconfig.pluginmanager.hasplugin("timeout"))
    # Missing image is an explicit failure, never a skip or an implicit pull.
    from robots.microduck.container_runtime import resolve_local_image

    image = os.environ.get("MICRODUCK_IMAGE", "rpent-microduck:local")
    resolve_local_image(image)
    return image


def _run_cli(output, image, mode, monkeypatch):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from rpent.cli import main as cli
    from rpent.planner.api_loop import ApiAgentLoop

    calls = iter(_script())

    async def stream(messages, info):
        from pydantic_ai.messages import ToolReturnPart

        # This seam observes real tool returns before the next planner request;
        # it does not instrument or replace the Env/tool execution path.
        at = time.monotonic()
        responses = [
            part.tool_name
            for part in messages[-1].parts
            if isinstance(part, ToolReturnPart)
        ]
        boundary_file = output / "planner-boundaries.json"
        boundaries = (
            json.loads(boundary_file.read_text()) if boundary_file.exists() else []
        )
        name, arguments = (
            (None, {}) if mode in {"planner-failure", "parent-death"} else next(calls)
        )
        boundaries.append({"request_time_s": at, "tool": name, "responses": responses})
        boundary_file.write_text(json.dumps(boundaries))
        if mode == "parent-death":
            (output / "planner-ready").touch()
            while True:
                time.sleep(1)
        if mode == "planner-failure":
            raise RuntimeError("injected planner failure")
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(arguments))}

    def build(*args, **kwargs):
        return ApiAgentLoop(
            model=FunctionModel(stream_function=stream),
            dashboard_events=kwargs["dashboard_events"],
        )

    monkeypatch.setattr(cli, "build_planner", build)
    if mode == "finalization-failure":
        from robots.microduck.toolkit import MicroduckToolkit

        def fail_publication(self, metadata):
            raise RuntimeError("injected video publication failure")

        monkeypatch.setattr(MicroduckToolkit, "_publish_video", fail_publication)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "microduck",
            "--planner",
            "api",
            "--task",
            "move forward 5 seconds, and then move backward",
            "--microduck-image",
            image,
            "--output-dir",
            str(output),
            "--max-turns",
            "3",
        ],
    )
    return cli.main()


def _assert_removed(output):
    runtime = json.loads((output / ".microduck/runtime.json").read_text())
    remaining = subprocess.run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"label=org.rpent.microduck.run={runtime['run_id']}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert not remaining.stdout.strip(), f"Owned container remains: {runtime['name']}"


def _assert_recorded_phases(trajectory, boundaries, samples):
    """Correlate acquired rows with planner/tool boundaries, not nominal indices."""
    assert trajectory["completion"] is True
    times = trajectory["frame_times"]
    assert len(times) == trajectory["count"] == len(samples) and len(times) > 1
    assert all(math.isfinite(value) for pair in times for value in pair)
    # Sampling is regular in simulated time; acquisition can catch up after
    # host scheduling jitter without dropping any simulated frames.
    for before, after in zip(times, times[1:]):
        assert after[0] - before[0] == pytest.approx(1 / 25, rel=0, abs=1e-9)
        assert after[1] > before[1]
    assert times[-1][0] - times[0][0] == pytest.approx(
        (trajectory["count"] - 1) / 25, rel=0, abs=1e-8
    )
    # Keep the existing two-period tolerance only for nearest wall boundaries.
    tolerance = 2 / 25
    requests = [boundary["request_time_s"] for boundary in boundaries]
    assert all(math.isfinite(at) for at in requests)
    assert all(a < b for a, b in zip(requests, requests[1:]))
    if len(boundaries) == 1:
        assert boundaries[0]["tool"] is None  # planner failure before any tool
        assert boundaries[0]["responses"] == []
    else:
        assert [b["tool"] for b in boundaries] == ["robot_move", "robot_move", "finish"]
        assert [b["responses"] for b in boundaries] == [
            [],
            ["robot_move"],
            ["robot_move"],
        ]
        assert requests[1] - requests[0] >= 5
        assert requests[2] - requests[1] >= 3
    assert times[0][1] < requests[0]
    # The final decoded frame represents one 25 Hz frame period; do not
    # reuse the wider nearest-boundary allowance to shorten final settling.
    assert times[-1][1] + 1 / 25 >= requests[-1] + 0.5 - 1e-9, (
        "Missing final 0.5s settling coverage"
    )

    def nearest(at):
        index = min(range(len(times)), key=lambda i: abs(times[i][1] - at))
        assert abs(times[index][1] - at) <= tolerance
        return index

    # Match the existing probe's held-SIT height and Session standing criteria.
    assert abs(samples[0]["height"] - 0.07) <= 0.005 + 1e-12, "Missing held SIT"
    standing = [
        i
        for i, sample in enumerate(samples)
        if times[i][1] <= requests[0]
        and sample["height"] > 0.095
        and sample["gravity_z"] < -0.9
    ]
    assert standing, "Missing recorded stand-up before planner readiness"
    assert standing[0] > 0
    return sorted(
        {0, standing[0], *(nearest(at) for at in requests), nearest(requests[-1] + 0.5)}
    )


# Executed only in the already-built image, never with host MuJoCo. Model and
# state layout come from baked provenance and MuJoCo, not guessed array offsets.
_REPLAY_DIAGNOSTICS = r"""
import json
from pathlib import Path
import mujoco
import numpy as np

info = json.loads(Path('/opt/microduck/build-info.json').read_text())
model = mujoco.MjModel.from_xml_path(info['scene'])
meta = json.loads(Path('/recording/.microduck/trajectory.json').read_text())
assert meta['scene'] == info['scene']
assert meta['dtype'] == '<f8'
assert mujoco.mj_stateSize(model, meta['signature']) == meta['state_size']
raw = Path('/recording/.microduck/trajectory.bin').read_bytes()
assert len(raw) == meta['count'] * meta['state_size'] * 8
rows = np.frombuffer(raw, dtype='<f8').reshape(meta['count'], meta['state_size'])
assert np.isfinite(rows).all()
body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
assert body >= 0
joint = int(model.body_jntadr[body])
assert joint >= 0 and model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
data = mujoco.MjData(model)
samples = []
for row in rows:
    mujoco.mj_setState(model, data, row, meta['signature'])
    mujoco.mj_forward(model, data)
    samples.append({'height': float(data.xpos[body, 2]),
                    'gravity_z': float(-data.xmat[body].reshape(3, 3)[2, 2])})
print(json.dumps(samples))
"""


def _recorded_samples(output):
    from robots.microduck.container_runtime import _command, _endpoint

    runtime = json.loads((output / ".microduck/runtime.json").read_text())
    endpoint = _endpoint()
    name = f"rpent-microduck-artifact-check-{uuid.uuid4().hex}"
    try:
        result = _command(
            [
                "run",
                "--rm",
                "--pull=never",
                "--network=none",
                "--name",
                name,
                "--label",
                f"org.rpent.microduck.artifact-check={name}",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,src={output.resolve()},dst=/recording,readonly",
                "--entrypoint",
                "/opt/microduck/rl/.venv/bin/python",
                runtime["image_id"],
                "-c",
                _REPLAY_DIAGNOSTICS,
            ],
            endpoint=endpoint,
            timeout=60,
        )
        return json.loads(result)
    finally:
        # Also cover an interrupted Docker client; touch only our unique name.
        remaining = _command(
            ["ps", "-aq", "--filter", f"name=^/{name}$"], endpoint=endpoint
        )
        if remaining.strip():
            _command(["rm", "-f", name], endpoint=endpoint)


def _assert_video(output):
    import imageio.v2 as imageio
    import numpy as np

    manifest = json.loads((output / "states.json").read_text())
    assert "microduck.mp4" in manifest["run_artifacts"]
    trajectory = json.loads((output / ".microduck/trajectory.json").read_text())
    selected_indices = _assert_recorded_phases(
        trajectory,
        json.loads((output / "planner-boundaries.json").read_text()),
        _recorded_samples(output),
    )
    selected = []
    count = 0
    with imageio.get_reader(output / "microduck.mp4") as reader:
        metadata = reader.get_meta_data()
        assert metadata["fps"] == 25
        assert metadata["codec"] == "h264"
        assert metadata["pix_fmt"].split("(", 1)[0] == "yuv420p"
        for count, frame in enumerate(reader, 1):
            assert frame.shape == (480, 640, 3)
            if count - 1 in selected_indices:
                selected.append(frame)
    assert count == trajectory["count"]
    assert len(selected) == len(selected_indices)
    assert not np.array_equal(selected[0], selected[1])
    imageio.imwrite(
        output / "startup-contact-sheet.png", np.concatenate(selected, axis=1)
    )


@pytest.mark.timeout(600)
def test_real_cli_sequence(live, tmp_path, monkeypatch):
    try:
        assert _run_cli(tmp_path, live, "normal", monkeypatch) == 0
    finally:
        _assert_removed(tmp_path)
    _assert_transcript(tmp_path)
    _assert_video(tmp_path)


def _assert_transcript(output):
    transcript = json.loads((output / "transcript_microduck.json").read_text())
    assert transcript["error"] is None
    assert transcript["finish"]["_finish"] is True
    assert transcript["environment_success"] is None
    calls = [m for m in transcript["messages"] if m["role"] == "assistant"]
    actual = [c for m in calls for c in m["content"] if c["type"] == "tool_use"]
    assert [(c["name"], c["input"]) for c in actual] == list(_script())
    observations = [m for m in transcript["messages"] if m["role"] == "tool"]
    assert [m["name"] for m in observations] == [name for name, _ in _script()]
    assert [json.loads(m["content"]) for m in observations[:2]] == [{}, {}]
    assert all("image" not in m["content"].lower() for m in observations)
    manifest = json.loads((output / "states.json").read_text())
    assert len(manifest["steps"]) == 3
    assert all(step["state"] == {} for step in manifest["steps"])
    assert all(step["result"] == {} for step in manifest["steps"])
    assert all(step["artifacts"] == [] for step in manifest["steps"])


@pytest.mark.timeout(600)
@pytest.mark.parametrize("mode", ["planner-failure", "finalization-failure"])
def test_real_cli_failure_cleanup(live, tmp_path, monkeypatch, mode):
    try:
        assert _run_cli(tmp_path, live, mode, monkeypatch) == 1
    finally:
        _assert_removed(tmp_path)
    transcript = json.loads((tmp_path / "transcript_microduck.json").read_text())
    assert "injected" in transcript["error"]
    if mode == "finalization-failure":
        assert (tmp_path / ".microduck/finalization-error.json").is_file()
        assert not (tmp_path / "microduck.mp4").exists()
    else:
        _assert_video(tmp_path)


@pytest.mark.timeout(600)
def test_parent_death_removes_only_owned_container(live, tmp_path):
    code = (
        "from pathlib import Path; import sys, pytest; "
        "from tests.e2e_tests.microduck.test_cli import _run_cli; "
        "_run_cli(Path(sys.argv[1]), sys.argv[2], 'parent-death', pytest.MonkeyPatch())"
    )
    with (tmp_path / "parent.log").open("w") as log:
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), live],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 100
            while not (tmp_path / "planner-ready").exists():
                assert child.poll() is None, (
                    "CLI died before readiness; inspect parent.log"
                )
                assert time.monotonic() < deadline, (
                    "CLI did not reach planner readiness"
                )
                time.sleep(0.1)
            child.kill()
            child.wait(timeout=5)
            deadline = time.monotonic() + 40
            while True:
                try:
                    _assert_removed(tmp_path)
                    break
                except AssertionError:
                    assert time.monotonic() < deadline, (
                        "Parent-death cleanup unconfirmed"
                    )
                    time.sleep(0.2)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


def test_script_requests_explicit_forward_and_default_backward():
    assert _script() == (
        ("robot_move", {"duration_s": 5.0}),
        ("robot_move", {"vx": -0.4}),
        ("finish", {"status": "stuck", "summary": "Open-loop sequence executed"}),
    )


def test_opt_in_requires_enforced_timeout_before_runtime():
    with pytest.raises(RuntimeError, match="pytest-timeout"):
        _require_timeout(False)


@pytest.fixture
def recorded_phases():
    # Synthetic replay diagnostics, not evidence of real simulator/video behavior.
    times = [[i / 25, 100 + i / 25] for i in range(276)]
    trajectory = {"completion": True, "count": len(times), "frame_times": times}
    boundaries = [
        {"request_time_s": 102.0, "tool": "robot_move", "responses": []},
        {"request_time_s": 107.0, "tool": "robot_move", "responses": ["robot_move"]},
        {"request_time_s": 110.0, "tool": "finish", "responses": ["robot_move"]},
    ]
    samples = [
        {"height": 0.07 if i < 25 else 0.11, "gravity_z": -1.0}
        for i in range(len(times))
    ]
    return trajectory, boundaries, samples


def test_recorded_phase_oracle_accepts_complete_sequence(recorded_phases):
    selected = _assert_recorded_phases(*recorded_phases)
    assert selected[0] == 0
    assert len(selected) >= 5


def test_recorded_phase_oracle_accepts_settled_planner_failure(recorded_phases):
    trajectory, _, samples = recorded_phases
    boundaries = [{"request_time_s": 102.0, "tool": None, "responses": []}]
    assert _assert_recorded_phases(trajectory, boundaries, samples)[0] == 0


@pytest.mark.parametrize("planner_failure", [False, True])
def test_recorded_phase_oracle_accepts_wall_jitter_and_catch_up(
    recorded_phases, planner_failure
):
    trajectory, boundaries, samples = recorded_phases
    # Reproduce the live 0.094186894s acquisition gap, followed by catch-up.
    # Every simulated sample remains exactly 0.04s apart.
    trajectory["frame_times"][10][1] += 0.054186894
    trajectory["frame_times"][11][1] += 0.027093447
    if planner_failure:
        boundaries = [{"request_time_s": 102.0, "tool": None, "responses": []}]
    assert _assert_recorded_phases(trajectory, boundaries, samples)[0] == 0


def test_recorded_phase_oracle_rejects_one_missing_simulated_frame(recorded_phases):
    trajectory, _, samples = recorded_phases
    # A single missing row leaves a 0.08s simulated gap, which the old oracle
    # accepted even with otherwise complete phases and matching row counts.
    del trajectory["frame_times"][10]
    del samples[10]
    trajectory["count"] -= 1
    boundaries = [{"request_time_s": 102.0, "tool": None, "responses": []}]
    with pytest.raises(AssertionError):
        _assert_recorded_phases(trajectory, boundaries, samples)


@pytest.mark.parametrize("clock", [0, 1], ids=["simulated", "monotonic"])
@pytest.mark.parametrize("offset", [math.nan, math.inf, 0, -0.01])
def test_recorded_phase_oracle_rejects_invalid_clocks(recorded_phases, clock, offset):
    trajectory, _, samples = recorded_phases
    # For either clock, the finite values repeat or precede the previous row.
    trajectory["frame_times"][10][clock] = trajectory["frame_times"][9][clock] + offset
    boundaries = [{"request_time_s": 102.0, "tool": None, "responses": []}]
    with pytest.raises(AssertionError):
        _assert_recorded_phases(trajectory, boundaries, samples)


def test_recorded_samples_use_separate_offline_readonly_container(
    tmp_path, monkeypatch
):
    from robots.microduck import container_runtime

    (tmp_path / ".microduck").mkdir()
    image_id = "sha256:" + "a" * 64
    (tmp_path / ".microduck/runtime.json").write_text(
        json.dumps({"image_id": image_id})
    )
    commands = []

    def command(args, **kwargs):
        commands.append((args, kwargs))
        return '[{"height": 0.07, "gravity_z": -1}]' if args[0] == "run" else ""

    monkeypatch.setattr(container_runtime, "_command", command)
    monkeypatch.setattr(container_runtime, "_endpoint", lambda: "unix:///offline-test")
    assert _recorded_samples(tmp_path) == [{"height": 0.07, "gravity_z": -1}]
    argv, options = commands[0]
    assert {"--rm", "--pull=never", "--network=none"} <= set(argv)
    assert (
        argv[argv.index("--mount") + 1]
        == f"type=bind,src={tmp_path},dst=/recording,readonly"
    )
    assert argv[argv.index("--entrypoint") + 1 :] == [
        "/opt/microduck/rl/.venv/bin/python",
        image_id,
        "-c",
        _REPLAY_DIAGNOSTICS,
    ]
    name = argv[argv.index("--name") + 1]
    assert name.startswith("rpent-microduck-artifact-check-")
    assert commands[1][0] == ["ps", "-aq", "--filter", f"name=^/{name}$"]
    assert options == {"endpoint": "unix:///offline-test", "timeout": 60}


@pytest.mark.parametrize(
    "defect",
    [
        "startup-only",
        "missing-forward",
        "missing-backward",
        "no-settle",
        "short-settle",
        "no-sit",
        "no-stand",
        "late-stand",
        "truncated-states",
    ],
)
def test_recorded_phase_oracle_rejects_missing_coverage(recorded_phases, defect):
    trajectory, boundaries, samples = recorded_phases
    if defect in {"startup-only", "no-settle", "short-settle", "truncated-states"}:
        end = {
            "startup-only": 51,
            "no-settle": 251,
            "short-settle": 262,
            "truncated-states": 275,
        }[defect]
        samples = samples[:end]
        if defect != "truncated-states":
            trajectory["frame_times"] = trajectory["frame_times"][:end]
            trajectory["count"] = end
    elif defect.startswith("missing-"):
        start, end = (102, 107) if defect == "missing-forward" else (107, 110)
        keep = [
            i
            for i, (_, at) in enumerate(trajectory["frame_times"])
            if not start < at < end
        ]
        trajectory["frame_times"] = [trajectory["frame_times"][i] for i in keep]
        trajectory["count"] = len(keep)
        samples = [samples[i] for i in keep]
    else:
        for i, sample in enumerate(samples):
            if defect == "no-sit":
                sample["height"] = 0.11
            elif defect == "no-stand" or (defect == "late-stand" and i < 60):
                sample["height"] = 0.07
    with pytest.raises(AssertionError):
        _assert_recorded_phases(trajectory, boundaries, samples)


@pytest.mark.parametrize("truncated", [False, True])
def test_video_oracle_checks_decoded_frame_coverage(
    tmp_path, monkeypatch, recorded_phases, truncated
):
    import imageio.v2 as imageio
    import numpy as np

    trajectory, boundaries, samples = recorded_phases
    (tmp_path / ".microduck").mkdir()
    (tmp_path / ".microduck/trajectory.json").write_text(json.dumps(trajectory))
    (tmp_path / "planner-boundaries.json").write_text(json.dumps(boundaries))
    (tmp_path / "states.json").write_text(
        json.dumps({"run_artifacts": ["microduck.mp4"]})
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_recorded_samples", lambda output: samples
    )

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_meta_data(self):
            return {"fps": 25, "codec": "h264", "pix_fmt": "yuv420p"}

        def __iter__(self):
            yield np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(trajectory["count"] - 1 - int(truncated)):
                yield np.ones((480, 640, 3), dtype=np.uint8)

    monkeypatch.setattr(imageio, "get_reader", lambda path: Reader())
    monkeypatch.setattr(imageio, "imwrite", lambda *args: None)
    if truncated:
        with pytest.raises(AssertionError):
            _assert_video(tmp_path)
    else:
        _assert_video(tmp_path)


def test_opt_in_gate_fails_before_image_lookup_without_timeout(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("RPENT_MICRODUCK_E2E", "1")
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(hasplugin=lambda name: False)
    )
    with pytest.raises(RuntimeError, match="pytest-timeout"):
        live.__wrapped__(config)


@pytest.mark.parametrize("mode", ["normal", "planner-failure", "finalization-failure"])
def test_scripted_cli_contract_offline(tmp_path, monkeypatch, mode):
    from dataclasses import replace
    from types import SimpleNamespace

    from robots.microduck import get_robot_spec
    from rpent.cli import main as cli
    from tests.unit_tests.robots.microduck.test_toolkit import RecordingEnv

    env = RecordingEnv(tmp_path)
    (tmp_path / ".microduck").mkdir()
    (tmp_path / ".microduck/video.mp4").write_bytes(b"video")
    spec = replace(
        get_robot_spec(),
        init_runtime=lambda *args: (
            [SimpleNamespace(stop=lambda: None)],
            {"env": env, "output_dir": tmp_path, "run_id": "offline"},
        ),
    )
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: spec)
    assert _run_cli(tmp_path, "unused-offline", mode, monkeypatch) == (mode != "normal")
    if mode != "normal":
        transcript = json.loads((tmp_path / "transcript_microduck.json").read_text())
        assert "injected" in transcript["error"]
        assert env.events[-2:] == ["finish", "close"]
        if mode == "finalization-failure":
            assert (tmp_path / ".microduck/finalization-error.json").is_file()
        return
    _assert_transcript(tmp_path)
    boundaries = json.loads((tmp_path / "planner-boundaries.json").read_text())
    assert [b["tool"] for b in boundaries] == ["robot_move", "robot_move", "finish"]
    assert [b["responses"] for b in boundaries] == [[], ["robot_move"], ["robot_move"]]
    assert all(
        a["request_time_s"] < b["request_time_s"]
        for a, b in zip(boundaries, boundaries[1:])
    )
    assert env.events[:2] == [
        ("move", {"vx": 0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 5.0}),
        ("move", {"vx": -0.4, "vy": 0.0, "vyaw": 0.0, "duration_s": 3.0}),
    ]
