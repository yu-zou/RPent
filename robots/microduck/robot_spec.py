# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Microduck CLI hooks; simulator dependencies remain inside the owned image."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from rpent.dashboard.events import DashboardEventSink
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import stop_owned_daemons, try_spawn_server, try_wait_server
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon

if TYPE_CHECKING:
    from robots.microduck.toolkit import MicroduckToolkit


def get_robot_spec() -> RobotSpec:
    """Return the offline-safe Microduck extension descriptor."""
    from robots.microduck.prompt_bundle import system_prompt, user_prompt

    return RobotSpec(
        name="microduck",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        validate_args=_validate_args,
        prepare_memory=_prepare_memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument("--task", required=True, help="Natural-language motion task.")
    parser.add_argument("--microduck-image", default="rpent-microduck:local")
    # Shared CLI fills an unspecified profile with HF before invoking validation.
    parser.set_defaults(memory_profile="local")


def _validate_args(args: argparse.Namespace) -> None:
    """Reject unsupported options without opening files or starting services."""
    task = getattr(args, "task", None)
    if not isinstance(task, str) or not task.strip():
        raise ValueError("--task must be nonblank")
    if getattr(args, "dashboard", False):
        raise ValueError("Microduck does not support --dashboard")
    if getattr(args, "explore", False):
        raise ValueError("Microduck does not support --explore")
    if getattr(args, "planner", None) == "flash":
        raise ValueError("Microduck does not support --planner flash")
    if getattr(args, "memory_repo", None) is not None or getattr(
        args, "memory_profile", None
    ) not in (None, "local"):
        raise ValueError(
            "Microduck supports only fresh local memory, not remote sources"
        )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    _validate_args(args)
    output = getattr(args, "output_dir", None)
    if output is None:
        output = get_repo_root() / "logs" / f"{datetime.now():%Y%m%d-%H%M%S}_microduck"
    output = Path(output)
    return RunConfig(
        recipe_tag="microduck",
        output_dir=output,
        prompt_vars={
            "task": args.task,
            "memory_dir": str(output / ".microduck-memory"),
        },
        task_desc={"task": args.task},
    )


def _prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Select a fresh, empty run-local memory root without HF synchronization."""
    _validate_args(args)
    root = config.output_dir / ".microduck-memory"
    if root.exists():
        raise ValueError(f"Microduck memory must be fresh: {root}")
    args.memory_profile = "local"
    args.memory_dir = str(root)
    config.prompt_vars["memory_dir"] = str(root)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None = None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Start exactly one owned supervisor and connect after stand-up readiness."""
    _validate_args(args)
    if components is not None and components != {"env"}:
        raise ValueError("Microduck runtime supports only the complete env component")
    from robots.microduck.container_runtime import spawn_container
    from robots.microduck.env_client import MicroduckEnvClient

    run_id = str(uuid4())
    owned: dict[str, ProcessDaemon] = {}
    daemon, rpc = try_spawn_server(
        owned,
        dashboard_events,
        "env",
        lambda: spawn_container(
            image=args.microduck_image, output_dir=output_dir, run_id=run_id
        ),
    )
    try:
        env = try_wait_server(
            owned,
            dashboard_events,
            "env",
            rpc,
            daemon,
            60.0,
            post_fn=lambda: MicroduckEnvClient(rpc, run_id=run_id),
        )
    except BaseException as exc:
        # The shared wait helper stops owned daemons; this hook owns the RPC too.
        if not isinstance(exc, Exception):
            stop_owned_daemons(owned, dashboard_events)
        try:
            rpc.close()
        except Exception:
            from rpent.utils.logging import get_logger

            get_logger(__name__).exception("Microduck startup RPC cleanup failed")
        raise
    return list(owned.values()), {
        "env": env,
        "output_dir": output_dir,
        "run_id": run_id,
    }


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
) -> MicroduckToolkit:
    """Construct the empty-state toolkit using the run's local memory manager."""
    from robots.microduck.toolkit import MicroduckToolkit
    from rpent.memory import MemoryManager
    from rpent.session import EnvState

    return MicroduckToolkit(
        env=runtime_kwargs["env"],
        output_dir=config.output_dir,
        state=EnvState(config.output_dir),
        memory=MemoryManager(config.output_dir / ".microduck-memory"),
        dashboard_events=dashboard_events,
    )
