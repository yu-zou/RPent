# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
"""Offline ownership and bounded supervisor contracts; never invoke Docker."""

import json
import subprocess
import sys
import textwrap
import threading
from uuid import UUID

import pytest

from robots.microduck import container_runtime as runtime

RUN = "12345678123456781234567812345678"
NAME = f"rpent-microduck-{RUN}"
IMAGE = "sha256:" + "a" * 64
CID = "b" * 64


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(runtime.time, "monotonic", value.monotonic)
    monkeypatch.setattr(runtime.time, "sleep", value.sleep)
    return value


def owned(label=RUN, running=True):
    return {
        "Id": CID,
        "Name": "/" + NAME,
        "Image": IMAGE,
        "Config": {"Labels": {runtime.RUN_LABEL: label}},
        "State": {"Running": running, "ExitCode": 0},
        "NetworkSettings": {
            "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49123"}]}
        },
    }


class Docker:
    def __init__(self, record=None, failures=()):
        self.calls = []
        self.record = record or owned()
        self.failures = failures

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[0] in self.failures:
            raise RuntimeError("Docker unavailable")
        if args[:2] == ["container", "inspect"]:
            return json.dumps([self.record])
        if args[:2] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": IMAGE, "Config": {"Labels": {runtime.PROTOCOL_LABEL: "1"}}}]
            )
        if args == ["context", "show"]:
            return "default\n"
        if args[:2] == ["context", "inspect"]:
            return '[{"Endpoints":{"docker":{"Host":"unix:///var/run/docker.sock"}}}]'
        if args[0] == "run":
            return CID
        return "container log\n"


@pytest.fixture
def docker(monkeypatch):
    fake = Docker()
    monkeypatch.setattr(runtime, "_command", fake)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    return fake


def test_image_is_existing_immutable_protocol_one(docker):
    assert runtime.resolve_local_image("rpent-microduck:local") == IMAGE
    assert all(args[0] not in ("build", "pull", "run") for args, _ in docker.calls)


@pytest.mark.parametrize("label", [None, "2", 1])
def test_incompatible_protocol_rejected(docker, label):
    original = docker.__call__

    def command(args, **kwargs):
        if args[:2] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": IMAGE, "Config": {"Labels": {runtime.PROTOCOL_LABEL: label}}}]
            )
        return original(args, **kwargs)

    runtime._command = command
    with pytest.raises(RuntimeError, match="protocol"):
        runtime.resolve_local_image("tag")


def test_missing_image_has_exact_name_only_build_guidance(monkeypatch, docker):
    def fail(args, **kwargs):
        if args[:2] == ["image", "inspect"]:
            raise RuntimeError("missing")
        return docker(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", fail)
    with pytest.raises(RuntimeError) as error:
        runtime.resolve_local_image("missing")
    assert runtime.BUILD_COMMAND in str(error.value)
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        assert f"--build-arg {name}" in str(error.value)
        assert f"{name}=" not in str(error.value)


@pytest.mark.parametrize(
    "host", ["tcp://127.0.0.1:2375", "ssh://server", "npipe:////pipe/docker_engine"]
)
def test_remote_endpoint_rejected(monkeypatch, docker, host):
    monkeypatch.setenv("DOCKER_HOST", host)
    with pytest.raises(RuntimeError, match="Unix"):
        runtime.resolve_local_image("tag")
    assert not any(args[0] == "image" for args, _ in docker.calls)


def test_context_overrides_host(monkeypatch, docker):
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    original = docker.__call__

    def command(args, **kwargs):
        if args[:2] == ["context", "inspect"]:
            assert args[2] == "remote"
            return '[{"Endpoints":{"docker":{"Host":"ssh://remote"}}}]'
        return original(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", command)
    with pytest.raises(RuntimeError, match="Unix"):
        runtime.resolve_local_image("tag")


@pytest.mark.parametrize("venv", [None, "/incorrect/venv"])
def test_spawn_exact_current_interpreter_spaces_and_inherited_env(
    monkeypatch, docker, tmp_path, venv
):
    sentinel = "/current venv/bin/python"
    monkeypatch.setattr(sys, "executable", sentinel)
    if venv is None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    else:
        monkeypatch.setenv("VIRTUAL_ENV", venv)
    output = tmp_path / "output with spaces"
    output.mkdir()
    (output / "cli.json").write_text("unrelated")

    def start(self):
        write = output / ".microduck/runtime.json"
        write.write_text(
            json.dumps(
                {
                    "run_id": RUN,
                    "container_id": CID,
                    "name": NAME,
                    "image_id": IMAGE,
                    "url": "http://127.0.0.1:49123",
                }
            )
        )

    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "start", start)
    monkeypatch.setattr(runtime, "make_rpc_client", lambda url: url)
    daemon, rpc = runtime.spawn_container(image="tag", output_dir=output, run_id=RUN)
    assert daemon.cmd == [
        sentinel,
        "-m",
        "robots.microduck.container_runtime",
        "--image-id",
        IMAGE,
        "--output-dir",
        str(output.resolve()),
        "--run-id",
        RUN,
        "--parent-watch",
    ]
    assert daemon.subprocess_env.get("VIRTUAL_ENV") == venv
    assert rpc == "http://127.0.0.1:49123"
    assert (output / "cli.json").read_text() == "unrelated"


def test_supervisor_stop_default_is_pure_delegation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime.ProcessDaemon, "stop", lambda self, timeout: calls.append(timeout)
    )
    daemon = runtime.MicroduckSupervisorDaemon("test", ["sentinel"])
    daemon.stop()
    daemon.stop(7)
    assert calls == [30.0, 7]


def test_hyphenated_launch_identity_survives_real_env_handshake(
    monkeypatch, docker, tmp_path
):
    from robots.microduck.env_client import MicroduckEnvClient
    from robots.microduck.env_server import MicroduckEnvFacade
    from robots.microduck.session import MicroduckSession
    from rpent.utils.rpc.http_rpc import HttpRpcServer

    # Only Docker/process execution is substituted: identity, supervisor
    # publication, HTTP transport, facade and client comparison remain real.
    run_id = "12345678-1234-5678-1234-567812345678"
    shutdown = runtime._Shutdown()
    servers, workers = [], []

    def command(args, **kwargs):
        if args[0] == "run":
            env_id = next(
                arg.removeprefix("MICRODUCK_RUN_ID=")
                for arg in args
                if arg.startswith("MICRODUCK_RUN_ID=")
            )
            facade = MicroduckEnvFacade(
                MicroduckSession(tmp_path / ".microduck"), run_id=env_id
            )
            server = HttpRpcServer(("127.0.0.1", 0), facade._dispatch)
            worker = threading.Thread(target=server.serve_forever)
            servers.append(server)
            workers.append(worker)
            docker.record["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"] = str(
                server.server_address[1]
            )
            worker.start()
        result = docker(args, **kwargs)
        if (
            args[:2] == ["container", "inspect"]
            and (tmp_path / ".microduck/runtime.json").exists()
        ):
            shutdown.request()
        return result

    def start(self):
        argv = self.cmd
        assert (
            runtime._supervise(
                argv[argv.index("--image-id") + 1],
                tmp_path,
                argv[argv.index("--run-id") + 1],
                shutdown,
            )
            == 0
        )

    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "start", start)
    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "poll", lambda self: None)
    rpc = None
    try:
        _, rpc = runtime.spawn_container(
            image="tag", output_dir=tmp_path, run_id=run_id
        )
        record = json.loads((tmp_path / ".microduck/runtime.json").read_text())
        assert record["run_id"] == RUN
        assert rpc.call("env.get_env_meta") == {
            "robot": "microduck",
            "protocol": 1,
            "run_id": RUN,
            "recording_fps": 25,
        }
        client = MicroduckEnvClient(rpc, run_id=run_id)
        assert client.last_obs == {}
        with pytest.raises(AssertionError, match="env_meta mismatch"):
            MicroduckEnvClient(rpc, run_id="12345678-1234-5678-1234-567812345679")
    finally:
        if rpc is not None:
            rpc.close()
        for server in servers:
            server.shutdown()
            server.server_close()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()


@pytest.mark.parametrize(
    "filename",
    [
        "runtime.json",
        "supervisor.log",
        "container.log",
        "trajectory.bin",
        "trajectory.json",
        "video.mp4",
        "controller.log",
        "renderer.log",
        "finalization-error.json",
    ],
)
def test_output_collisions_preserved(monkeypatch, docker, tmp_path, filename):
    root = tmp_path / ".microduck"
    root.mkdir()
    artifact = root / filename
    artifact.write_text("old")
    with pytest.raises((FileExistsError, ValueError)):
        runtime.spawn_container(image="tag", output_dir=tmp_path, run_id=RUN)
    assert artifact.read_text() == "old"


def test_root_video_collision(docker, tmp_path):
    (tmp_path / "microduck.mp4").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        runtime.spawn_container(image="tag", output_dir=tmp_path, run_id=RUN)


@pytest.mark.parametrize(
    "filename",
    [
        ".runtime.json.tmp",
        ".startup-failure.json.tmp",
        ".cleanup.json.tmp",
        ".finalization-error.json.tmp",
    ],
)
@pytest.mark.parametrize("kind", ["file", "symlink", "dangling_symlink"])
def test_atomic_scratch_collision_rejected_before_supervisor_spawn(
    monkeypatch, docker, tmp_path, filename, kind
):
    output = tmp_path / "output"
    state = output / ".microduck"
    state.mkdir(parents=True)
    target = tmp_path / "external-sentinel"
    if kind != "dangling_symlink":
        target.write_bytes(b"external sentinel\n")
    scratch = state / filename
    if kind == "file":
        scratch.write_bytes(b"prior scratch\n")
    else:
        scratch.symlink_to(target)
    before = scratch.lstat()
    started = []

    def start(self):
        started.append(True)
        raise RuntimeError("collision reached supervisor spawn")

    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "start", start)
    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "stop", lambda self: None)
    monkeypatch.setattr(runtime, "cleanup_owned_container", lambda *args: None)
    try:
        with pytest.raises(FileExistsError):
            runtime.spawn_container(image="tag", output_dir=output, run_id=RUN)
    finally:
        assert started == [], "scratch collision must reject before supervisor start"
        after = scratch.lstat()
        assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        if kind == "file":
            assert scratch.read_bytes() == b"prior scratch\n"
        else:
            assert scratch.is_symlink()
            assert scratch.readlink() == target
        if kind == "dangling_symlink":
            assert not target.exists()
        else:
            assert target.read_bytes() == b"external sentinel\n"


@pytest.mark.parametrize("failure", ["exit", "timeout", "stale", "rpc", "start"])
def test_spawn_failure_always_stops_and_attempts_named_cleanup(
    monkeypatch, docker, tmp_path, clock, failure
):
    stopped, cleaned = [], []

    def start(self):
        if failure == "start":
            raise OSError("cannot start")
        if failure in ("stale", "rpc"):
            (tmp_path / ".microduck/runtime.json").write_text(
                json.dumps(
                    {
                        "run_id": RUN if failure == "rpc" else UUID(int=1).hex,
                        "container_id": CID,
                        "name": NAME,
                        "image_id": IMAGE,
                        "url": "http://127.0.0.1:49123",
                    }
                )
            )

    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "start", start)
    monkeypatch.setattr(
        runtime.MicroduckSupervisorDaemon,
        "poll",
        lambda self: 1 if failure == "exit" else None,
    )
    monkeypatch.setattr(
        runtime.MicroduckSupervisorDaemon, "stop", lambda self: stopped.append(True)
    )
    monkeypatch.setattr(
        runtime,
        "cleanup_owned_container",
        lambda name, run_id, output_dir: cleaned.append((name, run_id)),
    )
    monkeypatch.setattr(
        runtime,
        "make_rpc_client",
        lambda url: (_ for _ in ()).throw(RuntimeError("rpc")),
    )
    with pytest.raises((RuntimeError, OSError, TimeoutError)):
        runtime.spawn_container(image="tag", output_dir=tmp_path, run_id=RUN)
    assert stopped == [True]
    assert cleaned == [(NAME, RUN)]
    assert clock.now <= 15.1
    metadata = json.loads((tmp_path / ".microduck/startup-failure.json").read_text())
    assert metadata["name"] == NAME
    assert metadata["label"] == f"{runtime.RUN_LABEL}={RUN}"
    assert "unconfirmed" in metadata["status"]


def test_cleanup_label_checked_logs_retained_remove_despite_errors(
    monkeypatch, docker, tmp_path
):
    docker.failures = ("stop", "logs")
    runtime.cleanup_owned_container(NAME, RUN, tmp_path)
    assert [args[0] for args, _ in docker.calls][-4:] == [
        "container",
        "stop",
        "logs",
        "rm",
    ]
    assert docker.calls[-1][0] == ["rm", "-f", CID]


def test_cleanup_refuses_foreign_label(docker, tmp_path):
    docker.record = owned(label=UUID(int=1).hex)
    with pytest.raises(RuntimeError, match="ownership"):
        runtime.cleanup_owned_container(NAME, RUN, tmp_path)
    assert not any(args[0] in ("stop", "logs", "rm") for args, _ in docker.calls)


def test_cleanup_repeated_absence_is_idempotent(monkeypatch, docker, tmp_path):
    runtime.cleanup_owned_container(NAME, RUN, tmp_path)

    def absent(args, **kwargs):
        if args[:2] == ["container", "inspect"]:
            raise runtime.ContainerAbsent("absent")
        return docker(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", absent)
    runtime.cleanup_owned_container(NAME, RUN, tmp_path)
    assert len([1 for args, _ in docker.calls if args[0] == "rm"]) == 1


def test_run_argv_and_inspected_port_metadata(monkeypatch, docker, tmp_path):
    shutdown = runtime._Shutdown()
    original = docker.__call__

    def command(args, **kwargs):
        result = original(args, **kwargs)
        if (
            args[:2] == ["container", "inspect"]
            and (tmp_path / ".microduck/runtime.json").exists()
        ):
            shutdown.request()
        return result

    monkeypatch.setattr(runtime, "_command", command)
    assert runtime._supervise(IMAGE, tmp_path, RUN, shutdown) == 0
    run = next(args for args, _ in docker.calls if args[0] == "run")
    assert run == [
        "run",
        "-d",
        "--init",
        "--pull=never",
        "--name",
        NAME,
        "--label",
        f"{runtime.RUN_LABEL}={RUN}",
        "--user",
        f"{runtime.os.getuid()}:{runtime.os.getgid()}",
        "--publish",
        "127.0.0.1::8000",
        "--mount",
        f"type=bind,src={tmp_path},dst=/output",
        "--env",
        "HOME=/tmp",
        "--env",
        f"MICRODUCK_RUN_ID={RUN}",
        IMAGE,
    ]
    metadata = json.loads((tmp_path / ".microduck/runtime.json").read_text())
    assert metadata == {
        "run_id": RUN,
        "container_id": CID,
        "name": NAME,
        "image_id": IMAGE,
        "url": "http://127.0.0.1:49123",
    }
    assert (tmp_path / ".microduck/container.log").read_text() == "container log\n"


def test_unexpected_container_exit_is_nonzero(monkeypatch, docker, tmp_path):
    docker.record = owned(running=False)
    assert runtime._supervise(IMAGE, tmp_path, RUN, runtime._Shutdown()) == 1


def test_parent_eof_and_signals_only_request_shutdown_close_watcher(
    monkeypatch, tmp_path
):
    callbacks, closed, handlers = [], [], {}

    class Watcher:
        def close(self):
            closed.append(True)

    def watch(callback):
        callbacks.append(callback)
        callback()
        return Watcher()

    monkeypatch.setattr(runtime, "watch_parent_death", watch)
    monkeypatch.setattr(
        runtime.signal,
        "signal",
        lambda sig, callback: handlers.setdefault(sig, callback),
    )

    def supervise(image_id, output_dir, run_id, shutdown):
        assert shutdown.event.is_set()
        for handler in handlers.values():
            handler(None, None)
        return 0

    monkeypatch.setattr(runtime, "_supervise", supervise)
    assert (
        runtime.main(
            [
                "--image-id",
                IMAGE,
                "--output-dir",
                str(tmp_path),
                "--run-id",
                RUN,
                "--parent-watch",
            ]
        )
        == 0
    )
    assert len(callbacks) == 1
    assert closed == [True]


def test_shutdown_global_budget_pending_run_late_creation_and_errors(
    monkeypatch, tmp_path, clock
):
    shutdown = runtime._Shutdown()
    calls = []
    inspections = 0

    def command(args, **kwargs):
        nonlocal inspections
        calls.append((args, kwargs, clock.now))
        if args == ["context", "show"]:
            return "default"
        if args[:2] == ["context", "inspect"]:
            return '[{"Endpoints":{"docker":{"Host":"unix:///var/run/docker.sock"}}}]'
        if args[0] == "run":
            shutdown.request()
            clock.sleep(5)
            raise TimeoutError("ambiguous creation")
        if args[:2] == ["container", "inspect"]:
            inspections += 1
            clock.sleep(3)
            if inspections == 1:
                raise runtime.ContainerAbsent("not yet created")
            return json.dumps([owned()])
        clock.sleep(kwargs["timeout"])
        raise TimeoutError("cleanup failure")

    monkeypatch.setattr(runtime, "_command", command)
    assert runtime._supervise(IMAGE, tmp_path, RUN, shutdown) == 1
    assert inspections >= 2
    assert any(args[0] == "rm" for args, _, _ in calls)
    assert clock.now <= 25
    for args, kwargs, started in calls:
        if args[0] in ("stop", "logs", "rm"):
            assert started + kwargs["timeout"] <= 25
    failure = json.loads((tmp_path / ".microduck/startup-failure.json").read_text())
    assert failure["name"] == NAME
    assert f"docker container inspect {NAME}" in failure["manual_cleanup"]
    assert runtime.RUN_LABEL in failure["manual_cleanup"]
    assert "prune" not in failure["manual_cleanup"]


def test_unavailable_docker_never_claims_absence(monkeypatch, tmp_path, clock):
    def unavailable(args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(runtime, "_command", unavailable)
    assert runtime._supervise(IMAGE, tmp_path, RUN, runtime._Shutdown()) == 1
    failure = (tmp_path / ".microduck/startup-failure.json").read_text()
    assert "unconfirmed" in failure.lower()
    assert NAME in failure


def test_cli_timeout_kill_and_reap_are_bounded(monkeypatch, clock):
    calls = []

    class Process:
        returncode = None

        def communicate(self, timeout):
            calls.append(("communicate", timeout))
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            calls.append(("kill",))

        def wait(self, timeout):
            calls.append(("wait", timeout))
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

    def popen(argv, **kwargs):
        assert isinstance(argv, list)
        assert "shell" not in kwargs
        return Process()

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    with pytest.raises(TimeoutError):
        runtime._command(["run", "image"], timeout=5, deadline=10)
    assert ("kill",) in calls
    assert all(call[1] <= 0.5 for call in calls if call[0] == "wait")
    assert clock.now <= 6


def test_docker_logs_preserve_container_stderr(monkeypatch):
    class Process:
        returncode = 0

        def communicate(self, timeout):
            return "stdout\n", "stderr\n"

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: Process())
    assert runtime._command(["logs", CID]) == "stdout\nstderr\n"


def test_start_error_is_not_masked_by_stop_error(monkeypatch, docker, tmp_path):
    cleaned = []

    def fail_start(self):
        raise OSError("original startup error")

    def fail_stop(self):
        raise RuntimeError("stop error")

    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "start", fail_start)
    monkeypatch.setattr(runtime.MicroduckSupervisorDaemon, "stop", fail_stop)
    monkeypatch.setattr(
        runtime, "cleanup_owned_container", lambda *args: cleaned.append(args)
    )
    with pytest.raises(OSError, match="original startup error"):
        runtime.spawn_container(image="tag", output_dir=tmp_path, run_id=RUN)
    assert len(cleaned) == 1


def test_public_cleanup_reconciles_ambiguous_creation(monkeypatch, tmp_path, clock):
    state = tmp_path / ".microduck"
    state.mkdir()
    (state / "startup-failure.json").write_text('{"status":"ambiguous"}')
    fake = Docker()
    count = 0

    def command(args, **kwargs):
        nonlocal count
        if args[:2] == ["container", "inspect"]:
            count += 1
            if count == 1:
                raise runtime.ContainerAbsent("not yet")
        return fake(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", command)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    runtime.cleanup_owned_container(NAME, RUN, tmp_path)
    assert count >= 2
    assert any(args[0] == "rm" for args, _ in fake.calls)


@pytest.mark.parametrize(
    "record",
    [
        owned(label="foreign"),
        {**owned(), "Image": "sha256:" + "c" * 64},
        {
            **owned(),
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8000"}]}
            },
        },
    ],
)
def test_supervisor_rejects_foreign_identity_and_nonloopback_metadata(
    monkeypatch, docker, tmp_path, record
):
    docker.record = record
    assert runtime._supervise(IMAGE, tmp_path, RUN, runtime._Shutdown()) == 1
    assert not (tmp_path / ".microduck/runtime.json").exists()
    if record["Config"]["Labels"][runtime.RUN_LABEL] != RUN:
        assert not any(args[0] in ("stop", "logs", "rm") for args, _ in docker.calls)


def test_host_overrides_selected_context(monkeypatch, docker):
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    runtime.resolve_local_image("tag")
    assert not any(args[0] == "context" for args, _ in docker.calls)
    assert docker.calls[-1][1]["endpoint"] == "unix:///run/user/1000/docker.sock"


@pytest.mark.parametrize("path_kind", ["symlink", "read_only"])
def test_output_must_be_owned_writable_directory(
    monkeypatch, docker, tmp_path, path_kind
):
    if path_kind == "symlink":
        target = tmp_path / "other"
        target.mkdir()
        (tmp_path / ".microduck").symlink_to(target)
    else:

        def denied(**kwargs):
            raise PermissionError("read only")

        monkeypatch.setattr(runtime.tempfile, "TemporaryFile", denied)
    with pytest.raises((ValueError, PermissionError)):
        runtime.spawn_container(image="tag", output_dir=tmp_path, run_id=RUN)


def test_command_timeout_obeys_existing_shutdown_deadline(monkeypatch, clock):
    shutdown = runtime._Shutdown()
    shutdown.request()
    clock.sleep(23)
    calls = []

    class Process:
        def communicate(self, timeout):
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            calls.append("kill")

        def wait(self, timeout):
            assert timeout <= 0.5
            clock.sleep(timeout)

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(TimeoutError):
        runtime._command(["run", "image"], timeout=5, shutdown=shutdown)
    assert calls == ["kill"]
    assert clock.now <= 25


def test_global_budget_real_command_runner_fake_daemon(monkeypatch, tmp_path, clock):
    shutdown = runtime._Shutdown()
    calls = []
    inspections = 0
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

    class Process:
        returncode = 0

        def __init__(self, args):
            self.args = args

        def communicate(self, timeout):
            nonlocal inspections
            if self.args[0] == "run":
                shutdown.request()
            if self.args[:2] == ["container", "inspect"]:
                inspections += 1
                if inspections == 1:
                    self.returncode = 1
                    return "", "Error: No such container: " + NAME
                return json.dumps([owned()]), ""
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            calls.append(("kill", clock.now))

        def wait(self, timeout):
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

    def popen(argv, **kwargs):
        assert argv[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
        calls.append((argv[3], clock.now))
        return Process(argv[3:])

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    assert runtime._supervise(IMAGE, tmp_path, RUN, shutdown) == 1
    assert inspections >= 2
    assert any(op == "rm" for op, _ in calls)
    assert 5 <= clock.now <= 25


def test_context_resolution_shutdown_counts_pending_commands(
    monkeypatch, tmp_path, clock
):
    shutdown = runtime._Shutdown()
    calls = []
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

    def command(args, **kwargs):
        calls.append((args, kwargs))
        if args == ["context", "show"]:
            shutdown.request()
            clock.sleep(5)
            return "default"
        assert kwargs.get("shutdown") is shutdown
        return '[{"Endpoints":{"docker":{"Host":"unix:///var/run/docker.sock"}}}]'

    monkeypatch.setattr(runtime, "_command", command)
    assert runtime._supervise(IMAGE, tmp_path, RUN, shutdown) == 0
    assert not any(args[0] == "run" for args, _ in calls)
    assert clock.now <= 25


def test_real_shared_parent_eof_watch_is_closed(monkeypatch, tmp_path):
    from rpent.utils.daemon import ParentDeathWatcher

    read_fd, write_fd = runtime.os.pipe()
    handles = []

    def watch(callback):
        handle = ParentDeathWatcher(callback, fd=read_fd)
        handles.append(handle)
        runtime.os.close(write_fd)
        return handle

    def supervise(image_id, output_dir, run_id, shutdown):
        assert shutdown.event.wait(2)
        assert shutdown.deadline is not None
        return 0

    monkeypatch.setattr(runtime, "watch_parent_death", watch)
    monkeypatch.setattr(runtime.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runtime, "_supervise", supervise)
    try:
        assert (
            runtime.main(
                [
                    "--image-id",
                    IMAGE,
                    "--output-dir",
                    str(tmp_path),
                    "--run-id",
                    RUN,
                    "--parent-watch",
                ]
            )
            == 0
        )
        assert not handles[0]._thread.is_alive()
    finally:
        runtime.os.close(read_fd)


@pytest.mark.parametrize("image_id", ["tag:mutable", "", "sha256:short"])
def test_image_inspection_refuses_mutable_or_malformed_ids(
    monkeypatch, docker, image_id
):
    original = docker.__call__

    def command(args, **kwargs):
        if args[:2] == ["image", "inspect"]:
            return json.dumps(
                [{"Id": image_id, "Config": {"Labels": {runtime.PROTOCOL_LABEL: "1"}}}]
            )
        return original(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", command)
    with pytest.raises(RuntimeError, match="immutable"):
        runtime.resolve_local_image("tag")


@pytest.mark.parametrize("labels", [None, {}])
def test_missing_image_labels_fail_with_protocol_error(monkeypatch, docker, labels):
    original = docker.__call__

    def command(args, **kwargs):
        if args[:2] == ["image", "inspect"]:
            return json.dumps([{"Id": IMAGE, "Config": {"Labels": labels}}])
        return original(args, **kwargs)

    monkeypatch.setattr(runtime, "_command", command)
    with pytest.raises(RuntimeError, match="protocol"):
        runtime.resolve_local_image("tag")


def test_global_budget_reserves_shared_watcher_close(monkeypatch, clock):
    shutdown = runtime._Shutdown()
    shutdown.request()
    clock.sleep(22)

    class Process:
        def communicate(self, timeout):
            clock.sleep(timeout)
            raise subprocess.TimeoutExpired("docker", timeout)

        def kill(self):
            pass

        def wait(self, timeout):
            clock.sleep(timeout)

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(TimeoutError):
        runtime._command(["logs", CID], shutdown=shutdown)
    # ParentDeathWatcher.close() has a one-second bounded join.
    clock.sleep(1)
    assert clock.now <= 25


def test_pinned_endpoint_cannot_be_overridden_by_inherited_context(monkeypatch):
    monkeypatch.setenv("DOCKER_CONTEXT", "selected-local")
    monkeypatch.setenv("DOCKER_HOST", "ssh://ignored-remote")
    captured = []

    class Process:
        returncode = 0

        def communicate(self, timeout):
            return "[]", ""

    def popen(argv, **kwargs):
        captured.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(runtime.subprocess, "Popen", popen)
    runtime._command(["image", "inspect", IMAGE], endpoint="unix:///run/docker.sock")
    argv, kwargs = captured[0]
    assert argv[:3] == ["docker", "--host", "unix:///run/docker.sock"]
    assert "DOCKER_CONTEXT" not in kwargs["env"]
    assert "DOCKER_HOST" not in kwargs["env"]
    expected = dict(runtime.os.environ)
    expected.pop("DOCKER_CONTEXT")
    expected.pop("DOCKER_HOST")
    assert kwargs["env"] == expected
    assert runtime.os.environ["DOCKER_CONTEXT"] == "selected-local"


def test_reentrant_shutdown_signal_cannot_deadlock_or_extend_deadline(monkeypatch):
    shutdown = runtime._Shutdown()
    interrupted = False
    errors = []

    def interrupted_clock():
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            shutdown.request()  # A Python signal can interrupt request itself.
        return 10.0

    monkeypatch.setattr(runtime.time, "monotonic", interrupted_clock)

    def request():
        try:
            shutdown.request()
        except Exception as exc:
            errors.append(exc)

    thread = runtime.threading.Thread(target=request, daemon=True)
    thread.start()
    thread.join(timeout=0.2)
    blocked = thread.is_alive()
    if blocked:
        # Unstick the original non-reentrant implementation after proving RED.
        shutdown._lock.release()
        thread.join(timeout=0.5)
    assert not blocked, "shutdown request deadlocked when interrupted by a signal"
    assert not errors
    assert shutdown.event.is_set()
    assert shutdown.deadline == 35.0
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 100.0)
    shutdown.request()
    assert shutdown.deadline == 35.0


@pytest.mark.parametrize("first", ["signal", "eof"])
def test_signal_and_eof_preserve_first_request_deadline(monkeypatch, first):
    shutdown = runtime._Shutdown()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.0)
    if first == "signal":
        shutdown.record_signal(None, None)
    else:
        shutdown.request()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 12.0)
    shutdown.record_signal(None, None)
    shutdown.request()
    shutdown.consume_signal()
    assert shutdown.deadline == 35.0
    assert shutdown.event.is_set()


def test_signal_handler_records_intent_without_thread_synchronization(
    monkeypatch, tmp_path
):
    handlers = {}

    def forbidden(*args, **kwargs):
        pytest.fail("signal handler entered threading synchronization")

    class ForbiddenLock:
        acquire = forbidden
        __enter__ = forbidden

    monkeypatch.setattr(
        runtime.signal,
        "signal",
        lambda sig, callback: handlers.setdefault(sig, callback),
    )

    def supervise(image_id, output_dir, run_id, shutdown):
        # Normal control flow may synchronize; the Python handlers must not.
        monkeypatch.setattr(shutdown, "request", forbidden)
        monkeypatch.setattr(shutdown.event, "set", forbidden)
        monkeypatch.setattr(shutdown, "_lock", ForbiddenLock())
        for sig in (runtime.signal.SIGTERM, runtime.signal.SIGINT):
            handlers[sig](sig, None)
        return 0

    monkeypatch.setattr(runtime, "_supervise", supervise)
    assert (
        runtime.main(
            [
                "--image-id",
                IMAGE,
                "--output-dir",
                str(tmp_path),
                "--run-id",
                RUN,
            ]
        )
        == 0
    )


@pytest.mark.parametrize("parent_eof", [False, True], ids=["signal", "signal-eof"])
def test_signal_inside_event_condition_lock_exits_with_owned_cleanup(
    tmp_path, parent_eof
):
    # A stuck child is killed/reaped by subprocess.run's outer timeout, not by
    # pytest-timeout (which is not a prerequisite of these offline tests).
    script = textwrap.dedent("""
        import json
        import os
        import runpy
        import signal
        import sys
        import threading
        import time
        from pathlib import Path

        from robots.microduck import container_runtime as runtime
        from rpent.utils.daemon import ParentDeathWatcher

        fixtures = runpy.run_path(sys.argv[1])
        fake = fixtures["Docker"]()
        runtime._command = fake
        os.environ.pop("DOCKER_CONTEXT", None)
        os.environ.pop("DOCKER_HOST", None)
        output = Path(sys.argv[2])
        overlap = sys.argv[3] == "True"
        original_shutdown = runtime._Shutdown
        eof_entered_set = threading.Event()
        watchers = []
        read_fd, write_fd = os.pipe()
        delivered = []

        class InstrumentedShutdown(original_shutdown):
            def __init__(self):
                super().__init__()
                original_wait = self.event.wait
                original_set = self.event.set

                def set_event():
                    if threading.current_thread() is not threading.main_thread():
                        eof_entered_set.set()
                    return original_set()

                def wait(timeout=None):
                    if not delivered:
                        with self.event._cond:
                            if overlap:
                                os.close(write_fd)
                                assert eof_entered_set.wait(2), "EOF never reached Event.set"
                            delivered.append(time.monotonic())
                            os.kill(os.getpid(), signal.SIGTERM)
                    return original_wait(timeout)

                self.event.set = set_event
                self.event.wait = wait

        def watch(callback):
            watcher = ParentDeathWatcher(callback, fd=read_fd)
            watchers.append(watcher)
            return watcher

        runtime._Shutdown = InstrumentedShutdown
        runtime.watch_parent_death = watch
        args = ["--image-id", fixtures["IMAGE"], "--output-dir", str(output),
                "--run-id", fixtures["RUN"]]
        if overlap:
            args.append("--parent-watch")
        try:
            result = runtime.main(args)
            assert delivered, "monitor never exercised the locked signal window"
            assert time.monotonic() - delivered[0] < 25
            assert all(not watcher._thread.is_alive() for watcher in watchers)
            print(json.dumps({"result": result, "calls": [args for args, _ in fake.calls]}))
        finally:
            os.close(read_fd)
            if not overlap:
                os.close(write_fd)
    """)
    try:
        child = subprocess.run(
            [sys.executable, "-c", script, __file__, str(tmp_path), str(parent_eof)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("supervisor deadlocked after SIGTERM inside Event condition lock")
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result["result"] == 0
    assert result["calls"][-4:] == [
        ["container", "inspect", NAME],
        ["stop", "--time", "3", CID],
        ["logs", CID],
        ["rm", "-f", CID],
    ]
    metadata = json.loads((tmp_path / ".microduck/runtime.json").read_text())
    assert metadata["container_id"] == CID
    assert (tmp_path / ".microduck/container.log").read_text() == "container log\n"
