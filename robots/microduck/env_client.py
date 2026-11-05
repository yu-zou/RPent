# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""Host client for one owned MicroDuck recording session."""

from typing import Any
from uuid import UUID

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.utils.rpc import RpcClient


def _metadata(run_id: str) -> dict[str, Any]:
    if not isinstance(run_id, str):
        raise ValueError("MicroDuck run_id must be a UUID string")
    try:
        normalized = UUID(run_id).hex
    except ValueError as exc:
        raise ValueError("MicroDuck run_id must be a UUID string") from exc
    return {
        "robot": "microduck",
        "protocol": 1,
        "run_id": normalized,
        "recording_fps": 25,
    }


class MicroduckEnvClient(BaseEnvClient):
    """Verify run identity without resetting the already-ready simulator."""

    def __init__(self, rpc: RpcClient, *, run_id: str) -> None:
        expected_meta = _metadata(run_id)
        self._TIMEOUT_S = {
            **BaseEnvClient._TIMEOUT_S,
            "env.move": 40.0,
            "env.finish": 340.0,
        }
        self._closed = False
        self.last_obs: dict[str, Any] = {}
        super().__init__(rpc, expected_meta=expected_meta, reset_on_connect=False)

    def move(
        self,
        vx: float = 0.4,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = 3.0,
    ) -> dict[str, Any]:
        """Execute one bounded velocity command without retrying partial moves."""
        result = self._client.call(
            "env.move",
            kwargs={"vx": vx, "vy": vy, "vyaw": vyaw, "duration_s": duration_s},
            timeout_s=self._TIMEOUT_S["env.move"],
        )
        self.last_obs = result
        return result

    def finish(self) -> dict[str, Any]:
        """Finish recording and return completed, relative video metadata."""
        return self._client.call("env.finish", timeout_s=self._TIMEOUT_S["env.finish"])

    def close(self) -> None:
        """Close the underlying RPC exactly once; do not implicitly render."""
        if not self._closed:
            self._closed = True
            self._client.close()
