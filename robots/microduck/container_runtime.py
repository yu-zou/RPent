# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""Owned local Docker supervisor for MicroDuck."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from rpent.evaluation.result import write_json_atomic
from rpent.utils.daemon import ProcessDaemon, watch_parent_death
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcClient, make_rpc_client

logger = get_logger("microduck.container_runtime")
RUN_LABEL = "org.rpent.microduck.run"
PROTOCOL_LABEL = "org.rpent.microduck.env-protocol"
BUILD_COMMAND = (
    "docker build "
    "--build-arg http_proxy --build-arg https_proxy "
    "--build-arg all_proxy --build-arg no_proxy "
    "--build-arg HTTP_PROXY --build-arg HTTPS_PROXY "
    "--build-arg ALL_PROXY --build-arg NO_PROXY "
    "-f robots/microduck/docker/Dockerfile -t rpent-microduck:local ."
)
_RESERVED = (
    "runtime.json",
    "supervisor.log",
    "container.log",
    "trajectory.bin",
    "trajectory.json",
    "video.mp4",
    "controller.log",
    "renderer.log",
    "finalization-error.json",
    "startup-failure.json",
    "cleanup.json",
    "owner.json",
    ".runtime.json.tmp",
    ".startup-failure.json.tmp",
    ".cleanup.json.tmp",
    ".finalization-error.json.tmp",
)


class ContainerAbsent(RuntimeError):
    """Docker explicitly reports that the inspected container does not exist."""


class MicroduckSupervisorDaemon(ProcessDaemon):
    """Give the cooperative child its cleanup budget without shared API changes."""

    def stop(self, timeout: float = 30.0) -> None:
        """Delegate lifecycle and stdin ownership to the shared daemon."""
        super().stop(timeout=timeout)


def _command(
    args: list[str],
    *,
    timeout: float = 5.0,
    deadline: float | None = None,
    endpoint: str | None = None,
    shutdown: _Shutdown | None = None,
) -> str:
    """Run one CLI command, reserving kill/reap and shared watcher-close time."""
    if shutdown is not None:
        shutdown.consume_signal()
    end = time.monotonic() + timeout
    if deadline is not None:
        end = min(end, deadline - 2.0)
    if end <= time.monotonic():
        raise TimeoutError("Docker cleanup budget exhausted; outcome unconfirmed")
    cmd = ["docker"] + (["--host", endpoint] if endpoint else []) + args
    env = os.environ.copy()
    if endpoint is not None:
        # The already validated endpoint is authoritative for this operation.
        env.pop("DOCKER_CONTEXT", None)
        env.pop("DOCKER_HOST", None)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        while True:
            limit = end
            if shutdown is not None:
                shutdown.consume_signal()
                if shutdown.deadline is not None:
                    limit = min(limit, shutdown.deadline - 2.0)
            remaining = limit - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Docker command timed out; outcome unconfirmed")
            try:
                stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=0.5)
        except (OSError, subprocess.TimeoutExpired):
            logger.error("Docker CLI reap unconfirmed after bounded wait")
        raise
    finally:
        for pipe in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
            if pipe is not None:
                pipe.close()
    if proc.returncode:
        if args[:2] == ["container", "inspect"] and (
            "No such container:" in stderr or "No such object:" in stderr
        ):
            raise ContainerAbsent("Docker reports named container absent")
        # Docker/context errors can contain endpoint credentials. Keep raw stderr private.
        raise RuntimeError(
            f"Docker {args[0]} failed (exit {proc.returncode}); outcome unconfirmed"
        )
    return stdout + stderr if args[0] == "logs" else stdout


def _endpoint(
    *, deadline: float | None = None, shutdown: _Shutdown | None = None
) -> str:
    context = os.environ.get("DOCKER_CONTEXT")
    if context:
        host = json.loads(
            _command(
                ["context", "inspect", context], deadline=deadline, shutdown=shutdown
            )
        )[0]["Endpoints"]["docker"]["Host"]
    elif os.environ.get("DOCKER_HOST"):
        host = os.environ["DOCKER_HOST"]
    else:
        context = _command(
            ["context", "show"], deadline=deadline, shutdown=shutdown
        ).strip()
        host = json.loads(
            _command(
                ["context", "inspect", context], deadline=deadline, shutdown=shutdown
            )
        )[0]["Endpoints"]["docker"]["Host"]
    if (
        sys.platform != "linux"
        or not isinstance(host, str)
        or not host.startswith("unix:///")
    ):
        raise RuntimeError("MicroDuck requires a local Linux Unix Docker endpoint")
    return host


def resolve_local_image(image: str) -> str:
    """Resolve an existing protocol-one image to its immutable local ID.

    No build, pull, login or dependency installation is performed.
    """
    endpoint = _endpoint()
    try:
        record = json.loads(_command(["image", "inspect", image], endpoint=endpoint))[0]
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        raise RuntimeError(
            f"MicroDuck local image unavailable. Build separately from the repository root:\n{BUILD_COMMAND}"
        ) from exc
    if (record.get("Config", {}).get("Labels") or {}).get(PROTOCOL_LABEL) != "1":
        raise RuntimeError("MicroDuck image requires env-protocol label 1")
    image_id = record.get("Id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeError("Docker did not return an immutable image ID")
    return image_id


def _identity(run_id: str) -> tuple[str, str]:
    normalized = UUID(run_id).hex
    return normalized, f"rpent-microduck-{normalized}"


def _prepare_output(output_dir: Path, run_id: str) -> Path:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = output / ".microduck"
    if state.is_symlink():
        raise ValueError("MicroDuck artifact directory must not be a symlink")
    if (output / "microduck.mp4").exists() or (output / "microduck.mp4").is_symlink():
        raise FileExistsError("Existing microduck.mp4 would collide with this run")
    state.mkdir(exist_ok=True)
    for name in _RESERVED:
        path = state / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Existing MicroDuck artifact: {name}")
    # Actual write permission check, without relying on VIRTUAL_ENV or os.access.
    with tempfile.TemporaryFile(dir=output):
        pass
    with (state / "owner.json").open("x") as file:
        json.dump({"run_id": run_id}, file)
    return output


def spawn_container(
    *, image: str, output_dir: Path, run_id: str
) -> tuple[ProcessDaemon, RpcClient]:
    """Start one current-interpreter supervisor and await its owned port metadata."""
    run_id, name = _identity(run_id)
    image_id = resolve_local_image(image)
    output = _prepare_output(Path(output_dir), run_id)
    daemon = MicroduckSupervisorDaemon(
        "microduck",
        [
            sys.executable,
            "-m",
            "robots.microduck.container_runtime",
            "--image-id",
            image_id,
            "--output-dir",
            str(output),
            "--run-id",
            run_id,
            "--parent-watch",
        ],
        log_path=str(output / ".microduck/supervisor.log"),
    )
    try:
        daemon.start()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                raise RuntimeError(
                    "MicroDuck supervisor exited before publishing runtime metadata"
                )
            try:
                record = json.loads((output / ".microduck/runtime.json").read_text())
            except FileNotFoundError:
                time.sleep(0.05)
                continue
            expected = {"run_id": run_id, "name": name, "image_id": image_id}
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError("MicroDuck runtime metadata identity mismatch")
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("container_id", "")):
                raise RuntimeError("Invalid MicroDuck container ID")
            url = record.get("url", "")
            match = re.fullmatch(r"http://127\.0\.0\.1:([0-9]+)", url)
            if not match or not 0 < int(match[1]) <= 65535:
                raise RuntimeError("Invalid MicroDuck loopback endpoint")
            return daemon, make_rpc_client(url)
        raise TimeoutError("MicroDuck supervisor metadata timeout (15s)")
    except BaseException:
        try:
            daemon.stop()
        except Exception:
            logger.error("MicroDuck supervisor stop failed; attempting named cleanup")
        try:
            failure_path = output / ".microduck/startup-failure.json"
            if not failure_path.exists():
                write_json_atomic(
                    failure_path,
                    {
                        "status": "host startup failure; cleanup unconfirmed",
                        "run_id": run_id,
                        "name": name,
                        "image_id": image_id,
                        "label": f"{RUN_LABEL}={run_id}",
                        "manual_cleanup": _manual(name, run_id),
                    },
                )
        except Exception:
            logger.error(
                "Cannot persist MicroDuck startup failure. %s", _manual(name, run_id)
            )
        try:
            cleanup_owned_container(name, run_id, output)
        except Exception:
            logger.error("MicroDuck cleanup unconfirmed. %s", _manual(name, run_id))
        raise


def _manual(name: str, run_id: str) -> str:
    return (
        f"Cleanup unconfirmed. On the original local Unix Docker endpoint run: "
        f"docker container inspect {name} --format '{{{{json .Config.Labels}}}}'; "
        f"only if {RUN_LABEL} equals {run_id}, run: docker rm -f {name}. "
        "A timed-out create can appear later; re-inspect this exact name. "
        "No other resources may be removed."
    )


def _inspect(name: str, run_id: str, **kwargs: Any) -> dict[str, Any]:
    record = json.loads(
        _command(["container", "inspect", name], timeout=3.0, **kwargs)
    )[0]
    if (record.get("Config", {}).get("Labels") or {}).get(
        RUN_LABEL
    ) != run_id or record.get("Name") != "/" + name:
        raise RuntimeError("MicroDuck container ownership mismatch; refusing cleanup")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("Id", "")):
        raise RuntimeError("Invalid owned container ID")
    return record


def _cleanup(
    name: str,
    run_id: str,
    output_dir: Path,
    *,
    deadline: float,
    endpoint: str,
    ambiguous: bool = False,
) -> None:
    reconcile_until = min(time.monotonic() + 7.0, deadline - 12.0)
    while True:
        try:
            record = _inspect(name, run_id, endpoint=endpoint, deadline=deadline)
            break
        except ContainerAbsent:
            if not ambiguous:
                return
            if time.monotonic() >= reconcile_until:
                raise RuntimeError(_manual(name, run_id)) from None
            time.sleep(min(0.2, max(0, reconcile_until - time.monotonic())))
    cid = record["Id"]
    errors = []
    for args, timeout in (
        (["stop", "--time", "3", cid], 5.0),
        (["logs", cid], 2.0),
        (["rm", "-f", cid], 2.0),
    ):
        try:
            result = _command(
                args, timeout=timeout, deadline=deadline, endpoint=endpoint
            )
            if args[0] == "logs":
                (output_dir / ".microduck/container.log").write_text(result)
        except Exception:
            errors.append(args[0])
            logger.error(
                "MicroDuck %s failed; continuing own-resource cleanup", args[0]
            )
    if errors:
        write_json_atomic(
            output_dir / ".microduck/cleanup.json",
            {"errors": errors, "manual_cleanup": _manual(name, run_id)},
        )
    if "rm" in errors:
        raise RuntimeError(_manual(name, run_id))


def cleanup_owned_container(name: str, run_id: str, output_dir: Path) -> None:
    """Inspect exact UUID ownership, retain logs, and remove only that container."""
    normalized, expected = _identity(run_id)
    if normalized != run_id or expected != name:
        raise ValueError("MicroDuck cleanup requires exact UUID name and label")
    output_dir = Path(output_dir)
    (output_dir / ".microduck").mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 25.0
    try:
        _cleanup(
            name,
            run_id,
            output_dir,
            deadline=deadline,
            endpoint=_endpoint(deadline=deadline),
            ambiguous=(output_dir / ".microduck/startup-failure.json").exists(),
        )
    except Exception:
        write_json_atomic(
            output_dir / ".microduck/cleanup.json",
            {"status": "unconfirmed", "manual_cleanup": _manual(name, run_id)},
        )
        raise


class _Shutdown:
    def __init__(self):
        self.event = threading.Event()
        self.deadline: float | None = None
        self._lock = threading.RLock()
        self._signal_time: float | None = None

    def record_signal(self, *_: Any) -> None:
        # Python handlers can interrupt Event.wait while its Condition is locked.
        # Record plain intent only; normal control flow performs synchronization.
        now = time.monotonic()
        self._signal_time = (
            min(now, self._signal_time) if self._signal_time is not None else now
        )

    def consume_signal(self) -> None:
        if self._signal_time is not None:
            self.request()

    def request(self) -> None:
        now = time.monotonic()
        with self._lock:
            requested = (
                min(now, self._signal_time) if self._signal_time is not None else now
            )
            first = self.deadline is None
            self.deadline = (
                min(self.deadline, requested + 25.0) if not first else requested + 25.0
            )
            if first:
                self.event.set()


def _supervise(
    image_id: str, output_dir: Path, run_id: str, shutdown: _Shutdown
) -> int:
    run_id, name = _identity(run_id)
    state = output_dir / ".microduck"
    state.mkdir(parents=True, exist_ok=True)
    endpoint = None
    attempted = False
    created = False
    result = 0
    try:
        shutdown.consume_signal()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise ValueError("Supervisor requires immutable image ID")
        endpoint = _endpoint(deadline=shutdown.deadline, shutdown=shutdown)
        shutdown.consume_signal()
        if shutdown.event.is_set():
            return 0
        mount = io.StringIO()
        csv.writer(mount, lineterminator="").writerow(
            [
                "type=bind",
                f"src={output_dir}",
                "dst=/output",
            ]
        )
        attempted = True
        cid = _command(
            [
                "run",
                "-d",
                "--init",
                "--pull=never",
                "--name",
                name,
                "--label",
                f"{RUN_LABEL}={run_id}",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--publish",
                "127.0.0.1::8000",
                "--mount",
                mount.getvalue(),
                "--env",
                "HOME=/tmp",
                "--env",
                f"MICRODUCK_RUN_ID={run_id}",
                image_id,
            ],
            endpoint=endpoint,
            shutdown=shutdown,
        ).strip()
        record = _inspect(name, run_id, endpoint=endpoint, shutdown=shutdown)
        if record["Id"] != cid or record.get("Image") != image_id:
            raise RuntimeError("Created container identity mismatch")
        created = True
        ports = record["NetworkSettings"]["Ports"]["8000/tcp"]
        if len(ports) != 1 or ports[0]["HostIp"] != "127.0.0.1":
            raise RuntimeError("Docker did not publish an exclusive loopback port")
        port = int(ports[0]["HostPort"])
        if not 0 < port <= 65535:
            raise RuntimeError("Invalid Docker published port")
        write_json_atomic(
            state / "runtime.json",
            {
                "run_id": run_id,
                "container_id": cid,
                "name": name,
                "image_id": image_id,
                "url": f"http://127.0.0.1:{port}",
            },
        )
        while True:
            shutdown.consume_signal()
            if shutdown.event.is_set():
                break
            if not record["State"]["Running"]:
                raise RuntimeError("MicroDuck container exited unexpectedly")
            shutdown.event.wait(1.0)
            shutdown.consume_signal()
            if not shutdown.event.is_set():
                record = _inspect(name, run_id, endpoint=endpoint, shutdown=shutdown)
    except Exception:
        result = 1
        write_json_atomic(
            state / "startup-failure.json",
            {
                "status": "startup or monitor failure; cleanup unconfirmed",
                "run_id": run_id,
                "name": name,
                "image_id": image_id,
                "label": f"{RUN_LABEL}={run_id}",
                "manual_cleanup": _manual(name, run_id),
            },
        )
        logger.error("MicroDuck supervisor failure. %s", _manual(name, run_id))
    finally:
        shutdown.request()
        if endpoint is not None and attempted:
            try:
                _cleanup(
                    name,
                    run_id,
                    output_dir,
                    deadline=shutdown.deadline,
                    endpoint=endpoint,
                    ambiguous=not created,
                )
            except Exception:
                result = 1
                write_json_atomic(
                    state / "cleanup.json",
                    {"status": "unconfirmed", "manual_cleanup": _manual(name, run_id)},
                )
                logger.error("%s", _manual(name, run_id))
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the child supervisor; signals and parent EOF only request shutdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args(argv)
    shutdown = _Shutdown()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, shutdown.record_signal)
    watcher = watch_parent_death(shutdown.request) if args.parent_watch else None
    try:
        return _supervise(args.image_id, args.output_dir, args.run_id, shutdown)
    finally:
        if watcher is not None:
            watcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
