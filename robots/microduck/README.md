# Microduck image recipe

This backend runs the simulator and controller in a prepared, native-architecture
Docker image, not in the host Python environment. The host optional extra is
`microduck` (`imageio-ffmpeg>=0.5`); package import needs no simulator.

**Recipe milestone only:** the final image build and offline verification are
deferred until the body, recorder, session and Env entrypoint exist (MD-02–04).
No build, gait, rendering or live acceptance is established by this recipe.

## Build separately

From the repository root, after those sources exist:

```bash
docker build \
  --build-arg http_proxy --build-arg https_proxy --build-arg all_proxy --build-arg no_proxy \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY --build-arg NO_PROXY \
  -f robots/microduck/docker/Dockerfile -t rpent-microduck:local .
```

Docker obtains exported proxy values by name, preserving casing and bypass
lists; do not substitute values into command arguments, dump the environment,
enable shell tracing, or persist proxies in configuration or artifacts. Registry
pulls require operator-configured daemon/BuildKit access separately. A host
loopback proxy is not the same endpoint inside build containers: report failures,
do not silently rewrite addresses or use host networking. Keep raw failure logs
private; redact URLs and credentials before sharing. Runtime must never build,
pull, download, upgrade, or explicitly forward proxies to the simulator.

After a successful build:

```bash
docker image inspect rpent-microduck:local --format '{{.Id}} {{index .Config.Labels "org.rpent.microduck.env-protocol"}}'
docker run --rm --network none --entrypoint python rpent-microduck:local /opt/rpent/robots/microduck/docker/verify_image.py
```

Record the immutable image ID and protocol label `1`. Inspect image configuration
and history privately for unintended proxy persistence; do not publish sensitive
output. Offline verification does not establish walking or video quality.

## Image contracts and provenance

`dependencies.json` pins the original controller, simulator and v8 policy commit.
Only Docker build acquires them. The controller uses locked native Cargo builds;
the simulator retains its original source, editable venv and frozen lockfile,
including training dependencies (training is not executed). RPent installation
must not change MuJoCo, NumPy, SciPy, ONNX Runtime, mjlab, warp-lang or
better-actuator-models. In particular MuJoCo3.10.0, NumPy2.4.1 and ORT1.24.4
are required. Resolver or ABI failures are build failures, not permission to
weaken pins.

- Controller: `/opt/microduck/runtime/bin/robotd`.
- Simulator and venv: `/opt/microduck/rl`, `.venv/bin` on PATH.
- Baked policy release: `/opt/microduck/policies/current`, requiring manifest
  and all six protocol models; no manifest-less fallback.
- `/opt/microduck/build-info.json`: pinned provenance, safe name/version package
  inventories, policy SHA256 inventory, discovered original interface modules,
  scene path and actual ORT shared-library path. The session must set robotd's
  `ORT_DYLIB_PATH` to this recorded path; no guessed library location.
- OSMesa, `HOME=/tmp`, writable temporary directory. Runtime supplies
  `MICRODUCK_RUN_ID`, host UID:GID and output bind mount; the Env server must
  reject a missing run ID. No planner keys or task text belong in the image.

Exact upstream module paths are not yet independently source-verified here.
The build verifier discovers unique original `World`, `Body`, `pose_table`,
`Server`, `Handler` definitions from the pinned source and validates imports;
ambiguity/missing definitions fail rather than trying guessed fallback modules.
MD-02 must consume these original interfaces; final image verification imports
the RPent body and recording adapters as well. Scene compilation checks assets,
the `SIT` keyframe and `trunk_base` body without stepping physics.

Controller license/notices are retained under `runtime/notices`; original
simulator source remains in place. Pinned policy repository notices/model cards
are discovered from its build-time file listing and retained under
`policies/current/notices` (missing notice evidence fails the build). Code and model
licenses must be reviewed independently; no commercial permission is implied.
The Docker context is allowlisted and excludes host environments, credentials,
caches, outputs and `docs/references/`; narrow COPY adds a second boundary.
