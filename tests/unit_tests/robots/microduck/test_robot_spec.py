# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# https://www.apache.org/licenses/LICENSE-2.0

"""Offline Microduck configuration and owned-runtime contracts."""

import argparse
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID

import pytest

from rpent.dashboard.events import NullDashboardEventSink


def options(tmp_path, **overrides):
    return argparse.Namespace(
        **{
            "task": "  Walk forward {exactly}\nthen backward.  ",
            "output_dir": tmp_path,
            "planner": "api",
            "model": "provider/exact-model",
            "memory_profile": None,
            "memory_repo": None,
            "memory_dir": None,
            "microduck_image": "rpent-microduck:local",
            "dashboard": False,
            "explore": False,
            **overrides,
        }
    )


def test_lazy_discovery_and_help_without_simulator():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rpent.robots import get_robot_spec; "
            "get_robot_spec('microduck'); "
            "assert not any(x in sys.modules for x in "
            "('mujoco', 'onnxruntime', 'robots.microduck.session', "
            "'robots.microduck.container_runtime'))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "rpent.cli.main", "--robot", "microduck", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--task" in result.stdout and "--microduck-image" in result.stdout


def test_task_defaults_image_and_verbatim_prompt(tmp_path):
    from robots.microduck import get_robot_spec

    spec = get_robot_spec()
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(["--task", "walk"])
    assert parsed.microduck_image == "rpent-microduck:local"
    assert (
        parser.parse_args(
            [
                "--task",
                "walk",
                "--microduck-image",
                "custom",
            ]
        ).microduck_image
        == "custom"
    )
    args = options(tmp_path)
    config = spec.parse_config(args)
    spec.prepare_memory(args, config)
    assert config.recipe_tag == "microduck" and config.output_dir == tmp_path
    assert config.task_desc["task"] == args.task
    assert args.model == "provider/exact-model"
    assert args.memory_profile == "local"
    assert args.memory_dir == str(tmp_path / ".microduck-memory")
    assert config.prompt_vars["memory_dir"] == args.memory_dir
    assert not (tmp_path / ".microduck-memory").exists()
    assert args.task in spec.prompts.render("user", variables=config.prompt_vars)
    system = spec.prompts.render("system", variables=config.prompt_vars)
    for text in (
        "robot_move",
        "3",
        "vx=0.4",
        "|vx| <= 0.4",
        "deadzone",
        "negative",
        "sequential",
        "feedback",
        "video",
    ):
        assert text in system
    assert spec.dashboard is spec.finalize_run is spec.run_flash is None
    assert not spec.is_real_robot and not spec.supports_exploration


def test_timestamped_output_and_fresh_memory_no_corpus(tmp_path, monkeypatch):
    from robots.microduck import get_robot_spec, get_toolkit, robot_spec

    monkeypatch.setattr(robot_spec, "get_repo_root", lambda: tmp_path)
    args = options(None)
    spec = get_robot_spec()
    config = spec.parse_config(args)
    assert config.output_dir.parent == tmp_path / "logs"
    assert config.output_dir.name.endswith("_microduck")
    spec.prepare_memory(args, config)
    monkeypatch.setattr(
        "rpent.memory.MemoryManager.sync", lambda *a, **kw: pytest.fail("HF sync")
    )
    env = SimpleNamespace(finish=lambda: {"video": "wrong"}, close=lambda: None)
    toolkit = get_toolkit(
        runtime_kwargs={"env": env},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    assert toolkit.memory.root == config.output_dir / ".microduck-memory"
    assert not toolkit.memory.root.exists()
    with pytest.raises(ValueError):
        toolkit.close()
    toolkit.memory.root.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        spec.prepare_memory(args, config)


@pytest.mark.parametrize(
    "override",
    [
        {"task": "  "},
        {"dashboard": True},
        {"explore": True},
        {"planner": "flash"},
        {"memory_profile": "hf"},
        {"memory_repo": "team/repo"},
    ],
)
@pytest.mark.parametrize(
    "hook", ["validate_args", "parse_config", "prepare_memory", "init_runtime"]
)
def test_invalid_options_rejected_at_every_hook_before_io(
    tmp_path, override, hook, monkeypatch
):
    from robots.microduck import get_robot_spec

    spec = get_robot_spec()
    args = options(tmp_path / "untouched", **override)
    config = SimpleNamespace(output_dir=args.output_dir, prompt_vars={})

    def unexpected(*a, **kw):
        pytest.fail("invalid options reached startup I/O")

    monkeypatch.setattr(
        "robots.microduck.container_runtime.spawn_container", unexpected
    )
    with pytest.raises(ValueError):
        if hook == "prepare_memory":
            spec.prepare_memory(args, config)
        elif hook == "init_runtime":
            spec.init_runtime(args, args.output_dir, NullDashboardEventSink(), None)
        else:
            getattr(spec, hook)(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("components", [set(), {"vla"}, {"env", "vla"}])
def test_unsupported_components_fail_before_spawn(tmp_path, components):
    from robots.microduck import get_robot_spec

    with pytest.raises(ValueError):
        get_robot_spec().init_runtime(
            options(tmp_path), tmp_path, NullDashboardEventSink(), components
        )


@pytest.mark.parametrize("connect_error", [False, True])
def test_owned_runtime_wait60_client_identity_and_startup_cleanup(
    tmp_path, monkeypatch, connect_error
):
    from robots.microduck import container_runtime, robot_spec

    events = []
    daemon = SimpleNamespace(stop=lambda: events.append("stop"))
    rpc = SimpleNamespace(close=lambda: events.append("close"))
    captured = {}

    def spawn(**kw):
        captured.update(kw)
        return daemon, rpc

    def wait(rpc_arg, *, daemon, timeout_s):
        assert rpc_arg is rpc and timeout_s == 60
        events.append("ready")

    def client(rpc_arg, *, run_id):
        assert rpc_arg is rpc and run_id == captured["run_id"]
        if connect_error:
            raise RuntimeError("identity mismatch")
        return "client"

    monkeypatch.setattr(container_runtime, "spawn_container", spawn)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", wait)
    monkeypatch.setattr("robots.microduck.env_client.MicroduckEnvClient", client)
    if connect_error:
        with pytest.raises(RuntimeError, match="identity mismatch"):
            robot_spec._init_runtime(
                options(tmp_path), tmp_path, NullDashboardEventSink()
            )
        assert "close" in events and "stop" in events
    else:
        daemons, kwargs = robot_spec._init_runtime(
            options(tmp_path), tmp_path, NullDashboardEventSink()
        )
        assert daemons == [daemon]
        assert kwargs == {
            "env": "client",
            "output_dir": tmp_path,
            "run_id": captured["run_id"],
        }
        UUID(captured["run_id"])
        assert captured["image"] == "rpent-microduck:local"


@pytest.mark.parametrize("failure", [RuntimeError("not ready"), KeyboardInterrupt()])
def test_readiness_failure_closes_rpc_and_stops_supervisor(
    tmp_path, monkeypatch, failure
):
    from robots.microduck import get_robot_spec

    events = []
    daemon = SimpleNamespace(stop=lambda: events.append("stop"))
    rpc = SimpleNamespace(close=lambda: events.append("close"))
    monkeypatch.setattr(
        "robots.microduck.container_runtime.spawn_container", lambda **kw: (daemon, rpc)
    )

    def wait(*a, **kw):
        raise failure

    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", wait)
    with pytest.raises(type(failure)):
        get_robot_spec().init_runtime(
            options(tmp_path), tmp_path, NullDashboardEventSink()
        )
    assert events.count("stop") == 1 and events.count("close") == 1


@pytest.mark.parametrize(
    "flags",
    [
        ["--task", " "],
        ["--task", "walk", "--dashboard"],
        ["--task", "walk", "--explore"],
        ["--task", "walk", "--planner", "flash"],
        ["--task", "walk", "--memory-profile", "hf"],
        ["--task", "walk", "--memory-repo", "team/repo"],
    ],
)
def test_cli_validation_precedes_output_memory_model_and_runtime(
    tmp_path, monkeypatch, flags
):
    from rpent.cli import main as cli

    def unexpected(*a, **kw):
        pytest.fail("invalid CLI reached I/O")

    monkeypatch.setattr(cli, "init_output_dir", unexpected)
    monkeypatch.setattr(cli, "build_planner", unexpected)
    monkeypatch.setattr("rpent.memory.MemoryManager.sync", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent", "--robot", "microduck", "--output-dir", str(tmp_path), *flags],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
