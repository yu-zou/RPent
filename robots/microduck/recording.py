# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Bounded state recording and main-thread, offline MuJoCo replay."""

import argparse
import json
import math
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np

DRAIN_TIMEOUT_S = 5.0
RENDER_TIMEOUT_S = 300.0
MAX_FRAMES = 7501
logger = get_logger(__name__)


class TrajectoryWriter:
    """Persist owned integration states on one bounded disk worker.

    Args:
        output_dir: Private run directory; existing trajectories are refused.
        state_size: Number of float64 values in each integration state.
        signature: MuJoCo state bitmask used to obtain each state.
        fps: Recording rate, fixed at 25 frames per simulated second.
    """

    def __init__(
        self, output_dir: Path, *, state_size: int, signature: int, fps: int = 25
    ) -> None:
        if state_size <= 0 or fps != 25:
            raise ValueError("Positive state size and 25 FPS are required")
        self.output_dir = output_dir
        self.state_size, self.signature, self.fps = state_size, int(signature), fps
        self.scene: str | None = None
        self.metadata_path = output_dir / "trajectory.json"
        self.binary_path = output_dir / "trajectory.bin"
        self._source_ids = json.loads(
            Path(__file__).with_name("dependencies.json").read_text()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            raise FileExistsError(self.metadata_path)
        self._stream = self.binary_path.open("xb", buffering=0)
        self._queue: queue.Queue = queue.Queue(maxsize=250)
        self._closed = threading.Event()
        self._failure: str | None = None
        self._guard = threading.Lock()
        self._accepted = 0
        self._count = 0
        self._published = False
        self._times: list[tuple[float, float]] = []
        self._thread = threading.Thread(
            target=self._run, name="microduck-trajectory", daemon=True
        )
        self._thread.start()

    @property
    def alive(self) -> bool:
        """Whether the disk worker still owns the trajectory stream."""
        return self._thread.is_alive()

    @property
    def failure(self) -> str | None:
        """First terminal recording failure, if any."""
        with self._guard:
            return self._failure

    def _fail(self, reason: str) -> None:
        with self._guard:
            if self._failure is None:
                self._failure = reason

    def append(
        self,
        state: "np.ndarray",
        *,
        sim_time_s: float | None = None,
        monotonic_time_s: float | None = None,
    ) -> None:
        """Enqueue an owned row without waiting for disk or queue capacity.

        Args:
            state: Integration state; copied before this method returns.
            sim_time_s: Absolute simulation time, or the regular frame time.
            monotonic_time_s: Acquisition clock time, or the current clock.

        Raises:
            RuntimeError: Recording has closed, failed, or exhausted its bounds.
            ValueError: State or timing is invalid.
        """
        import numpy as np

        if self.failure or self._closed.is_set():
            raise RuntimeError(self.failure or "Trajectory is closed")
        row = np.array(state, dtype="<f8", copy=True)
        if row.shape != (self.state_size,) or not np.isfinite(row).all():
            raise ValueError("Invalid integration state")
        sim = self._accepted / self.fps if sim_time_s is None else float(sim_time_s)
        mono = time.monotonic() if monotonic_time_s is None else float(monotonic_time_s)
        if not math.isfinite(sim) or not math.isfinite(mono):
            raise ValueError("Invalid frame time")
        if self._accepted >= MAX_FRAMES:
            self._fail("Trajectory exceeded 300 simulated seconds")
            raise RuntimeError(self.failure)
        try:
            self._queue.put_nowait((row, sim, mono))
        except queue.Full:
            self._fail("Trajectory queue overflow (250 states)")
            raise RuntimeError(self.failure) from None
        self._accepted += 1

    def _run(self) -> None:
        try:
            with self._stream as stream:
                while not self._closed.is_set() or not self._queue.empty():
                    try:
                        row, sim, mono = self._queue.get(timeout=0.01)
                    except queue.Empty:
                        continue
                    raw = row.tobytes()
                    if stream.write(raw) != len(raw):
                        raise OSError("Short trajectory write")
                    self._count += 1
                    self._times.append((sim, mono))
        except Exception as exc:
            self._fail(f"Trajectory disk failure: {exc}")
            # A failed/partial row must never be advertised as a valid prefix.
            try:
                with self.binary_path.open("r+b") as stream:
                    stream.truncate(self._count * self.state_size * 8)
            except OSError:
                logger.exception("Could not truncate failed trajectory row")
        finally:
            self._closed.set()
            try:
                failed_before_metadata = self.failure
                self._write_metadata()
                if failed_before_metadata is None and self.failure is not None:
                    # close() may time out while metadata is being written.
                    self._write_metadata()
            except Exception as exc:
                self._fail(f"Trajectory metadata failure: {exc}")
                logger.exception("Could not write trajectory metadata")

    def _write_metadata(self) -> None:
        first = self._times[0] if self._times else (None, None)
        last = self._times[-1] if self._times else (None, None)
        failure = self.failure
        metadata = {
            "signature": self.signature,
            "state_size": self.state_size,
            "dtype": "<f8",
            "count": self._count,
            "fps": self.fps,
            "scene": self.scene,
            "source_ids": self._source_ids,
            "sim_time_start_s": first[0],
            "sim_time_end_s": last[0],
            "monotonic_time_start_s": first[1],
            "monotonic_time_end_s": last[1],
            "frame_times": self._times,
            "state_span_s": max(0, self._count - 1) / self.fps,
            "video_duration_s": self._count / self.fps,
            "completion": failure is None,
            "failure": failure,
        }
        temporary = self.metadata_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        if failure is not None:
            temporary.replace(self.metadata_path)

    def close(self) -> Path:
        """Drain within five seconds and return metadata, or raise on failure.

        A timeout is terminal: a still-live worker must not be used for replay.
        A labeled valid prefix remains available after worker termination.
        """
        self._closed.set()
        self._thread.join(DRAIN_TIMEOUT_S)
        if self.alive:
            self._fail("Trajectory drain timeout")
            raise TimeoutError(self.failure)
        if self.failure:
            raise RuntimeError(self.failure)
        # Success is not visible to replay until the disk worker has exited.
        if not self._published:
            try:
                self.metadata_path.with_suffix(".json.partial").replace(
                    self.metadata_path
                )
                self._published = True
            except Exception as exc:
                self._fail(f"Trajectory metadata failure: {exc}")
                raise RuntimeError(self.failure) from exc
        return self.metadata_path


def render_trajectory(scene: Path, metadata_path: Path, output_path: Path) -> Path:
    """Replay completed states without physics steps and atomically publish MP4.

    Call only after the body's physics and trajectory worker have stopped. The
    container caller must run this function on a subprocess main thread and
    enforce a 300-second process timeout (including native encoder shutdown).

    Args:
        scene: Original scene XML.
        metadata_path: Completed trajectory metadata beside trajectory.bin.
        output_path: New MP4 destination; existing outputs are never replaced.

    Returns:
        The published MP4 path.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Replay renderer must run on the main thread")
    if output_path.exists():
        raise FileExistsError(output_path)
    os.environ["MUJOCO_GL"] = "osmesa"
    import numpy as np

    meta = json.loads(metadata_path.read_text())
    count, size = meta["count"], meta["state_size"]
    if (
        type(count) is not int
        or not 0 < count <= MAX_FRAMES
        or type(size) is not int
        or size <= 0
        or meta["dtype"] != "<f8"
        or meta["fps"] != 25
        or meta["completion"] is not True
        or meta["failure"] is not None
    ):
        raise ValueError("Invalid or incomplete trajectory metadata")
    binary = metadata_path.with_name("trajectory.bin")
    if binary.stat().st_size != count * size * 8:
        raise ValueError("Corrupt or truncated trajectory")
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    signature = meta["signature"]
    if mujoco.mj_stateSize(model, signature) != size:
        raise ValueError("Trajectory state signature/size does not match scene")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
    )
    if camera.trackbodyid < 0:
        raise ValueError("Scene has no trunk_base camera target")
    camera.distance, camera.azimuth, camera.elevation = 1.2, 135, -20
    temporary = output_path.with_name(f".{output_path.stem}-{uuid.uuid4().hex}.mp4")
    renderer = encoder = None
    deadline = time.monotonic() + RENDER_TIMEOUT_S
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        encoder = imageio.get_writer(
            str(temporary), fps=25, codec="libx264", pixelformat="yuv420p"
        )
        with binary.open("rb") as stream:
            for _ in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Offline rendering exceeded 300 seconds")
                state = np.frombuffer(stream.read(size * 8), dtype="<f8")
                if state.size != size or not np.isfinite(state).all():
                    raise ValueError("Corrupt trajectory state")
                mujoco.mj_setState(model, data, state, signature)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                encoder.append_data(renderer.render())
        encoder.close()
        encoder = None
        renderer.close()
        renderer = None
        if time.monotonic() >= deadline:
            raise TimeoutError("Offline rendering exceeded 300 seconds")
        # Same-directory hard linking publishes only the completed inode and
        # atomically refuses an existing or concurrently published destination.
        os.link(temporary, output_path)
        return output_path
    finally:
        try:
            if encoder is not None:
                encoder.close()
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            finally:
                temporary.unlink(missing_ok=True)


def main() -> None:
    """Run offline replay in a graphics-owning subprocess main thread."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_trajectory(args.scene, args.metadata, args.output)


if __name__ == "__main__":
    main()
