# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""Container-only ready-session HTTP facade with explicit video finalization."""

import argparse
import os
import signal
import threading
from pathlib import Path
from typing import Any

from robots.microduck.env_client import _metadata
from robots.microduck.session import MicroduckSession
from rpent.robots.components.env_facade_base import BaseEnvFacade


class MicroduckEnvFacade(BaseEnvFacade):
    """Expose only metadata, serialized movement and explicit finalization."""

    def __init__(self, session: MicroduckSession, *, run_id: str) -> None:
        self._meta = _metadata(run_id)
        self.session = session
        self._finished: dict[str, Any] | None = None
        self._finish_error: Exception | None = None
        self._close_lock = threading.Lock()
        self._closed = False
        super().__init__()

    def _register_rpc(self) -> None:
        # Do not advertise unsupported reset/step/camera routes from the base.
        self._rpc.update(
            {
                "env.get_env_meta": self.get_env_meta,
                "env.move": self.move,
                "env.finish": self.finish,
            }
        )
        self._readonly_methods.add("env.get_env_meta")

    def get_env_meta(self) -> dict[str, Any]:
        """Return the exact protocol and owned-run identity."""
        return self._meta.copy()

    def move(
        self,
        vx: float = 0.4,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration_s: float = 3.0,
    ) -> dict[str, Any]:
        """Delegate validation/motion to the session; return no observations."""
        self.session.move(vx=vx, vy=vy, vyaw=vyaw, duration_s=duration_s)
        return {}

    def finish(self) -> dict[str, Any]:
        """Publish only the fixed, nonempty completed artifact; cache failures."""
        if self._finish_error is not None:
            raise self._finish_error
        if self._finished is not None:
            return self._finished.copy()
        try:
            path = self.session.finish()
            expected = self.session.output_dir / "video.mp4"
            if path != expected or path.is_symlink() or not path.is_file():
                raise ValueError(
                    "Session did not return the fixed completed video path"
                )
            size = path.stat().st_size
            if size <= 0:
                raise ValueError("Completed MicroDuck video is empty")
            self._finished = {"video": ".microduck/video.mp4", "bytes": size}
        except Exception as exc:
            self._finish_error = exc
            raise
        return self._finished.copy()

    def close(self) -> None:
        """Emergency cleanup, not a promise of a rendered video."""
        # Not the dispatch lock: cancellation must reach an in-flight move.
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self.session.close()


def _check_output(output_dir: Path) -> Path:
    private = output_dir / ".microduck"
    if private.is_symlink() or (private.exists() and not private.is_dir()):
        raise FileExistsError(f"Unsafe MicroDuck session directory: {private}")
    reserved = {
        "trajectory.bin",
        "trajectory.json",
        "trajectory.json.partial",
        "video.mp4",
        "controller.log",
        "renderer.log",
    }
    if private.exists():
        for path in private.iterdir():
            if path.name in reserved or (
                path.name.startswith(".video-") and path.suffix == ".mp4"
            ):
                raise FileExistsError(
                    f"MicroDuck session artifact already exists: {path}"
                )
    return private


def main(argv: list[str] | None = None) -> None:
    """Start completely before binding; signals request non-reentrant cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    args = parser.parse_args(argv)
    run_id = os.environ.get("MICRODUCK_RUN_ID")
    _metadata(run_id)
    private = _check_output(args.output_dir)
    stop, done = threading.Event(), threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stop.set()

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    facade = None
    watcher = None
    try:
        session = MicroduckSession(private)
        facade = MicroduckEnvFacade(session, run_id=run_id)

        def watch_signal() -> None:
            while not done.wait(0.05):
                if stop.is_set():
                    facade._shutdown_event.set()
                    facade.close()
                    return

        watcher = threading.Thread(
            target=watch_signal, name="microduck-signals", daemon=True
        )
        watcher.start()
        if not stop.is_set():
            session.start()
        if not stop.is_set():
            facade.serve(transport="http", host=args.host, port=args.port)
    finally:
        done.set()
        if facade is not None:
            facade.close()
        if watcher is not None:
            watcher.join()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
