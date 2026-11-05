# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Open-loop motion toolkit and owned recording publication."""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from robots.microduck.env_client import MicroduckEnvClient
from rpent.dashboard.events import DashboardEventSink
from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools.base import ToolResult, tool
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger(__name__)


class MicroduckToolkit(Toolkit):
    """Expose only bounded velocity commands plus the common file/finish tools."""

    def __init__(
        self,
        *,
        env: MicroduckEnvClient,
        output_dir: Path,
        state: EnvState,
        memory: MemoryManager,
        dashboard_events: DashboardEventSink,
    ) -> None:
        self._env = env
        self._output_dir = Path(output_dir)
        self._close_lock = threading.Lock()
        self._closed = False
        self._controller_errors: list[str] = []
        super().__init__(state=state, memory=memory, dashboard_events=dashboard_events)
        self.add_tool(self.robot_move)
        finish = self._tools["finish"]
        self.add_tool(
            replace(
                finish,
                description="End the command sequence. Report execution, not verified environment success. Video is saved automatically on close.",
            ),
            replace=True,
        )
        self.get_env_state(command={}, result={}, elapsed_s=0.0)

    @tool(name="robot_move")
    def robot_move(
        self,
        vx: float = 0.4,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = 3.0,
    ) -> ToolResult:
        """Execute one sequential bounded velocity command without sensed feedback.

        Args:
            vx: Forward velocity intent in m/s; default 0.4, absolute limit 0.4.
                Negative requests backward motion. Smaller values are valid but
                may not produce walking in the simulator's low-speed deadzone.
            vy: Lateral velocity in m/s, absolute limit 0.10.
            vyaw: Yaw velocity in rad/s, absolute limit 0.5.
            duration_s: Duration in seconds, greater than zero and at most 30.
        """
        values = {"vx": vx, "vy": vy, "vyaw": vyaw, "duration_s": duration_s}
        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite nonboolean number")
        if (
            abs(vx) > 0.4
            or abs(vy) > 0.10
            or abs(vyaw) > 0.5
            or not 0 < duration_s <= 30
        ):
            raise ValueError("Microduck motion exceeds velocity or duration bounds")
        try:
            self._env.move(**values)
        except Exception as exc:
            self._controller_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        return ToolResult(data={})

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        """Record commands and errors without fabricating observations."""
        with self.state.record_step(
            state={}, command=command, result=result, elapsed_s=elapsed_s
        ):
            pass
        return ToolResult(data={})

    def _publish_video(self, metadata: dict[str, Any]) -> None:
        if (
            not isinstance(metadata, dict)
            or metadata.get("video") != ".microduck/video.mp4"
        ):
            raise ValueError("Microduck finish must return .microduck/video.mp4")
        size = metadata.get("bytes")
        if type(size) is not int or size <= 0:
            raise ValueError(
                "Microduck video byte size must be a positive nonboolean integer"
            )
        root = self._output_dir.resolve()
        staging = (root / ".microduck").resolve()
        path = (root / metadata["video"]).resolve(strict=True)
        if (
            not staging.is_relative_to(root)
            or not path.is_relative_to(staging)
            or not path.is_file()
        ):
            raise ValueError("Microduck video must be a contained regular file")
        if path.stat().st_size != size:
            raise ValueError("Microduck video byte size mismatch")
        data = path.read_bytes()
        if len(data) != size:
            raise ValueError("Microduck video changed during publication")
        if self.state.save("microduck.mp4", data, step=None) is None:
            raise RuntimeError("Microduck video artifact registration failed")

    def close(self) -> None:
        """Drain actions, finish once, publish bytes, then always close the RPC."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            errors: list[str] = []
            primary: BaseException | None = None
            try:
                super().close()
                self._publish_video(self._env.finish())
            except BaseException as exc:
                primary = exc
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                try:
                    self._env.close()
                except BaseException as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    if primary is None:
                        primary = exc
                if errors:
                    try:
                        write_json_atomic(
                            self._output_dir / ".microduck/finalization-error.json",
                            {
                                "controller_errors": self._controller_errors,
                                "errors": errors,
                            },
                        )
                    except Exception:
                        logger.exception(
                            "Could not persist Microduck finalization errors"
                        )
            if primary is not None:
                raise primary
