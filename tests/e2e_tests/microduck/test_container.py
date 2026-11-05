# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Explicitly opted-in local-image smoke test; never builds or downloads."""

import os
import subprocess
from uuid import uuid4

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RPENT_MICRODUCK_E2E") != "1",
        reason="Set RPENT_MICRODUCK_E2E=1 to run the owned local Docker smoke",
    ),
    pytest.mark.timeout(480),
]


def test_real_container_moves_and_publishes_once(tmp_path, pytestconfig):
    # MD-05 owns acquisition/cleanup. Import only after explicit opt-in so
    # ordinary collection has no Docker, simulator or optional-extra dependency.
    import imageio.v2 as imageio

    from robots.microduck.container_runtime import spawn_container
    from robots.microduck.env_client import MicroduckEnvClient
    from rpent.utils.rpc import wait_for_ready

    if not pytestconfig.pluginmanager.hasplugin("timeout"):
        pytest.fail("Opt-in MicroDuck E2E requires pytest-timeout for the 480s bound")
    run_id = str(uuid4())
    daemon, rpc = spawn_container(
        output_dir=tmp_path,
        image=os.environ.get("MICRODUCK_IMAGE", "rpent-microduck:local"),
        run_id=run_id,
    )
    client = None
    try:
        wait_for_ready(rpc, daemon=daemon, timeout_s=90)
        assert rpc.call("healthz") == {"status": "ok"}
        client = MicroduckEnvClient(rpc, run_id=run_id)
        assert client.move(duration_s=5) == {}
        assert client.move(vx=-0.4) == {}
        result = client.finish()
        video = tmp_path / ".microduck/video.mp4"
        before = video.stat()
        assert result == {"video": ".microduck/video.mp4", "bytes": before.st_size}
        assert before.st_size > 0
        assert client.finish() == result
        assert video.stat().st_mtime_ns == before.st_mtime_ns
        with imageio.get_reader(video) as reader:
            assert reader.get_meta_data()["fps"] == 25
            assert reader.get_data(0).shape[:2] == (480, 640)
    finally:
        if client is not None:
            client.close()
        else:
            rpc.close()
        daemon.stop()
        # An empty listing (not an arbitrary inspect failure) proves absence.
        remaining = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=org.rpent.microduck.run={run_id.replace('-', '')}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        assert not remaining.stdout.strip(), "Owned MicroDuck container leaked"
    assert video.is_file()
