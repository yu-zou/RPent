# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Microduck container integration (simulator dependencies stay in the image)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robots.microduck.toolkit import MicroduckToolkit
    from rpent.dashboard.events import DashboardEventSink
    from rpent.robots.robot_spec import RobotSpec, RunConfig


def get_robot_spec() -> RobotSpec:
    """Load the static descriptor without importing a simulator or Docker runtime."""
    from robots.microduck.robot_spec import get_robot_spec as factory

    return factory()


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
) -> MicroduckToolkit:
    """Load the host toolkit only when a run requests it."""
    from robots.microduck.robot_spec import get_toolkit as factory

    return factory(
        runtime_kwargs=runtime_kwargs, dashboard_events=dashboard_events, config=config
    )
