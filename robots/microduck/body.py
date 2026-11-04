# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Original MicroDuck body server with independent state recording."""

import threading
import time
from pathlib import Path
from typing import Any

from robots.microduck.recording import TrajectoryWriter
from rpent.utils.logging import get_logger

MAX_DURATION_S = 300.0
JOIN_TIMEOUT_S = 2.0
logger = get_logger(__name__)


class MicroduckBody:
    """Own one original SIT body, loopback server, and physics worker.

    Args:
        scene: Pinned original scene XML.
        output_dir: Private directory for recorded integration states.
    """

    def __init__(self, scene: Path, *, output_dir: Path) -> None:
        self.scene, self.output_dir = scene, output_dir
        self._world: Any = None
        self._body: Any = None
        self._server: Any = None
        self._mujoco: Any = None
        self._physics: threading.Thread | None = None
        self._serving: threading.Thread | None = None
        self._writer: TrajectoryWriter | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._recording_ready = threading.Event()
        self._begin_pending = False
        self._final_requested = False
        self._next_sample: float | None = None
        self._recording_start = 0.0
        self._failure: Exception | None = None
        self._closed = False
        self._metadata: Path | None = None

    def start(self) -> int:
        """Initialize held SIT before serving and return the loopback TCP port."""
        if self._closed or self._world is not None:
            raise RuntimeError("Body can only be started once")
        import mujoco
        from mjlab_microduck.sim.body_server import (
            Body,
            Handler,
            Server,
            World,
            pose_table,
        )

        self._mujoco = mujoco
        try:
            world = self._world = World(self.scene, 1)
            world.model.opt.timestep = 0.005
            pose, z = pose_table(self.scene, "SIT")
            if pose is None:
                raise ValueError("Scene has no SIT keyframe")
            self._body = Body(world, 0)
            self._body.place(pose, z, 0)
            world.bodies.append(self._body)
            mujoco.mj_forward(world.model, world.data)
            self._server = Server(("127.0.0.1", 0), Handler)
            self._server.body = self._body
            self._serving = threading.Thread(
                target=self._server.serve_forever,
                name="microduck-body-server",
                daemon=True,
            )
            self._serving.start()
            self._physics = threading.Thread(
                target=self._run, name="microduck-physics", daemon=True
            )
            self._physics.start()
            return int(self._server.server_address[1])
        except BaseException:
            self.close()
            raise

    def begin_recording(self) -> None:
        """Start recording with an immediate sample on the stepping thread."""
        if self._world is None or self._stop.is_set() or self._writer is not None:
            raise RuntimeError("Recording requires a running, unrecorded body")
        signature = self._mujoco.mjtState.mjSTATE_INTEGRATION
        self._writer = TrajectoryWriter(
            self.output_dir,
            state_size=self._mujoco.mj_stateSize(self._world.model, signature),
            signature=int(signature),
        )
        self._writer.scene = str(self.scene)
        with self._condition:
            self._begin_pending = True
            self._condition.notify_all()
        if not self._recording_ready.wait(JOIN_TIMEOUT_S):
            self._failure = TimeoutError("Initial trajectory sample timed out")
            self._writer._fail(str(self._failure))
            self._stop.set()
            raise self._failure
        if self._failure:
            raise RuntimeError("Initial trajectory sample failed") from self._failure

    def sample(self) -> dict[str, Any]:
        """Return atomic planner-invisible trunk and gravity diagnostics."""
        if self._world is None:
            raise RuntimeError("Body has not started")
        import numpy as np

        with self._world.lock:
            start = self._body.trunk
            qpos = self._world.data.qpos
            xyz = qpos[start : start + 3].copy()
            quat = qpos[start + 3 : start + 7].copy()
            rotation = np.empty(9)
            self._mujoco.mju_quat2Mat(rotation, quat)
            gravity = -rotation.reshape(3, 3).T[:, 2]
            return {
                "trunk_xyz": xyz.tolist(),
                "quat_wxyz": quat.tolist(),
                "gravity_xyz": gravity.tolist(),
                "sim_time_s": float(self._world.data.time),
                "monotonic_time_s": time.monotonic(),
            }

    def _capture(self) -> float:
        import numpy as np

        writer = self._writer
        assert writer is not None
        state = np.empty(writer.state_size, dtype="<f8")
        with self._world.lock:
            self._mujoco.mj_getState(
                self._world.model, self._world.data, state, writer.signature
            )
            sim = float(self._world.data.time)
            mono = time.monotonic()
        writer.append(state, sim_time_s=sim, monotonic_time_s=mono)
        return sim

    def _run(self) -> None:
        deadline = time.monotonic() + 0.02
        try:
            while not self._stop.is_set():
                with self._condition:
                    if not self._begin_pending:
                        self._condition.wait(max(0.0, deadline - time.monotonic()))
                    begin = self._begin_pending
                    self._begin_pending = False
                if self._stop.is_set():
                    break
                if begin:
                    self._recording_start = self._capture()
                    self._next_sample = self._recording_start + 0.04
                    self._recording_ready.set()
                if self._writer and self._writer.failure:
                    raise RuntimeError(self._writer.failure)
                if time.monotonic() < deadline:
                    continue
                # World.step owns its non-reentrant lock and held-pose restore.
                self._world.step(4)
                deadline += 0.02
                if self._next_sample is not None:
                    with self._world.lock:
                        sim = float(self._world.data.time)
                    if sim + 1e-8 >= self._next_sample:
                        self._capture()
                        self._next_sample += 0.04
                        if self._final_requested:
                            self._stop.set()
                        elif sim - self._recording_start + 1e-8 >= MAX_DURATION_S:
                            raise RuntimeError(
                                "Recording reached simulated-time cap "
                                f"({MAX_DURATION_S:g} seconds)"
                            )
        except Exception as exc:
            self._failure = exc
            self._stop.set()
            if self._writer:
                self._writer._fail(f"Body recording failed: {exc}")
            logger.exception("MicroDuck physics/recording stopped")
        finally:
            self._recording_ready.set()

    def finish(self, *, capture_final: bool = True) -> Path:
        """Stop physics before draining states; never create a renderer here.

        Args:
            capture_final: Wait for the next regular frame on normal completion;
                false skips that wait for error cleanup.

        Returns:
            Completed trajectory metadata, ready for separate-process replay.
        """
        if self._metadata is not None:
            return self._metadata
        if self._writer is None:
            raise RuntimeError("Recording has not begun")
        with self._condition:
            self._final_requested = True
            if not capture_final:
                self._stop.set()
            self._condition.notify_all()
        self._join_physics()
        path = self._writer.close()
        if self._failure:
            raise RuntimeError("Body recording failed") from self._failure
        self._metadata = path
        return path

    def _join_physics(self) -> None:
        if self._physics:
            self._physics.join(JOIN_TIMEOUT_S)
            if self._physics.is_alive():
                self._stop.set()
                if self._writer:
                    self._writer._fail("Physics thread join timeout")
                raise TimeoutError("Physics thread join timeout; replay forbidden")

    def close(self) -> None:
        """Bound cleanup of owned threads/server; incomplete runs are not MP4s."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        errors = []
        try:
            self._join_physics()
        except Exception as exc:
            errors.append(exc)
        if self._writer:
            try:
                self._writer.close()
            except Exception as exc:
                errors.append(exc)
        if self._server:
            # Neither shutdown nor server_close may block the owner indefinitely.
            def stop_server() -> None:
                try:
                    self._server.shutdown()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        self._server.server_close()
                    except Exception as exc:
                        errors.append(exc)

            shutdown = threading.Thread(target=stop_server, daemon=True)
            shutdown.start()
            shutdown.join(JOIN_TIMEOUT_S)
            if shutdown.is_alive():
                errors.append(TimeoutError("Body server shutdown timeout"))
        if self._serving:
            self._serving.join(JOIN_TIMEOUT_S)
            if self._serving.is_alive():
                errors.append(TimeoutError("Body server join timeout"))
        for error in errors:
            logger.warning("MicroDuck cleanup: %s", error)
        if errors:
            raise RuntimeError("MicroDuck cleanup incomplete") from errors[0]
