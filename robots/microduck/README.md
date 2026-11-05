# Microduck: build, then run

This backend runs the simulator and controller in a prepared, native-architecture
Docker image, not in the host Python environment. The host optional extra is
`microduck` (`imageio-ffmpeg>=0.5`); package import needs no simulator.

Use a local Linux Docker Engine through a Unix socket without sudo. Remote
engines/contexts are unsupported. Use the native Docker architecture; there is
no CPU-feature check or forced platform. No GPU, privileged mode, host networking
or Docker-socket mount is required. Host UID:GID needs output bind-mount write
permission.

All host `python` commands mean the existing current venv, never system Python
or a new environment. Host Python children use `sys.executable` unchanged.
Container Python is separate. Install missing prerequisites only with operator
permission; no automatic installation.

```bash
python -c 'import sys; assert sys.prefix != sys.base_prefix, "Use the current host venv"; print(sys.executable)'
```

## Build separately

Step 1, from the repository root:

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
private; redact URLs and credentials before sharing. Do not add proxy ARG/ENV
declarations or explicit Dockerfile references; they defeat predefined-proxy
history/cache protection. Runtime never builds, pulls, downloads or upgrades.
The supervisor does not explicitly forward proxies or planner keys, but operator
Docker client defaults can inject environment variables. It does not sanitize
arbitrary Docker configuration.

After a successful build:

```bash
docker image inspect rpent-microduck:local --format '{{.Id}} {{index .Config.Labels "org.rpent.microduck.env-protocol"}}'
docker run --rm --network none --entrypoint python rpent-microduck:local /opt/rpent/robots/microduck/docker/verify_image.py
```

Record the immutable image ID and protocol label `1`. Inspect image configuration
and history privately for unintended proxy persistence; do not publish sensitive
output. Offline verification does not establish walking or video quality.

Rebuild explicitly after changing container sources or pins. A missing image
reports the same build command above; runtime never pulls it.

## Run a task

Step 2: configure `OPENAI_API_KEY` and, for a custom OpenAI-compatible endpoint,
`OPENAI_BASE_URL`. The endpoint must serve the exact `openai:deepseek-flash`
model; do not substitute another model. Use a fresh output directory:

```bash
python -m rpent.cli.main --planner api --model openai:deepseek-flash --robot microduck \
  --task "move forward 5 seconds, and then move backward" --output-dir ./logs/xxxx
```

`rpent` is valid only if its installed script belongs to this venv. `--task` must
be nonblank. `--microduck-image` overrides `rpent-microduck:local`; only a local
protocol1 image is accepted, resolved to its immutable ID. Omitted output uses
`logs/<timestamp>_microduck`. Memory is fresh, empty and local. Dashboard, Flash,
exploration, VLA and physical-robot modes are unsupported.

The intended sequence is `robot_move(duration_s=5)`, `robot_move(vx=-0.4)`, then
`finish`. Defaults are `vx=0.4, vy=0.0, vyaw=0.0, duration_s=3.0`. Bounds are
`|vx|<=0.4` m/s, `|vy|<=0.10` m/s, `|vyaw|<=0.5` rad/s and
`0<duration_s<=30` seconds; values must be finite and nonboolean. The controller
API remains `robot.move`. Commands run sequentially and partially executed
failed moves are never automatically retried. Successful observations are empty
dictionaries, not pose/camera feedback. Neither observations nor `finish` verify
displacement, balance, obstacle avoidance or task success. This is open-loop
velocity control, not direct joint control.

Smaller requested velocities remain valid but may yield negligible motion.
Velocity requests are open-loop intents, not measured speeds or a precision
motion guarantee. A model may choose different calls or fail to complete the
sequence; inspect the transcript and video rather than assuming model quality.

## Results and cleanup

Normal close finishes recording and registers root `microduck.mp4` in
`states.json` before stopping the owned container. Inspect
`transcript_microduck.json` for ordered calls, empty observations and errors.
A trunk-tracking camera with fixed viewing parameters records H.264/yuv420p
video at 640×480/25fps spanning SIT, stand-up, waiting,
movement and final settling. The queue holds 250 states; recording is capped at
300 simulated seconds and offline rendering has a 300-second deadline.
Inspect video yourself; command completion is not verified task success.

`.microduck/` retains staged video, trajectory, `runtime.json` with run/image
identity, `supervisor.log`, `container.log`, controller/renderer logs and available
failure records. `finalization-error.json` records publication/render errors;
these remain failures. Report model endpoint errors separately from simulator
failures. Never publish raw logs, credentials or environment dumps.

Each run owns one UUID-named/labeled container as host UID:GID, with an ephemeral
loopback port and `--pull=never`. Shutdown removes only that container. Parent
death triggers cleanup while the supervisor survives; forced shutdown does not
guarantee MP4. Unavailable Docker or timed-out create/cleanup means removal is
unconfirmed: follow exact name/label inspection and manual cleanup instructions
in the failure record. Never prune unrelated resources. Simultaneous host and
supervisor death requires manual inspection.

See [tests/README.md](../../tests/README.md#microduck-offline-and-opt-in-cli-checks)
for offline/opt-in checks. Deterministic tests establish integration, not model
quality or repeatable gait success. Live checks require explicit opt-in and a
manually prepared image.

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

The build verifier discovers unique original `World`, `Body`, `pose_table`,
`Server`, `Handler` definitions from the pinned source and validates imports;
ambiguity/missing definitions fail rather than trying guessed fallback modules.
Adapters consume these original interfaces; final image verification imports
the RPent body and recording adapters as well. Scene compilation checks assets,
the `SIT` keyframe and `trunk_base` body without stepping physics.

Controller license/notices are retained under `runtime/notices`; original
simulator source remains in place. Pinned policy repository notices/model cards
are discovered from its build-time file listing and retained under
`policies/current/notices` (missing notice evidence fails the build). Code and model
licenses must be reviewed independently; no commercial permission is implied.
The Docker context is allowlisted and excludes host environments, credentials,
caches, outputs and `docs/references/`; narrow COPY adds a second boundary.
