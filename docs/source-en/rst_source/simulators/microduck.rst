Microduck
=========

Run bounded velocity commands in an owned Docker simulator and save a video.
This is an open-loop interface, not a verified navigation benchmark.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      Baked walking controller; no host VLA.

   .. grid-item-card:: Planners

      API planner with tool calls.

   .. grid-item-card:: Tasks

      Sequential bounded velocity commands.

   .. grid-item-card:: Hardware

      Local Linux Docker Engine; CPU rendering with OSMesa.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Category
     - Scope
   * - Open-loop locomotion
     - Forward, backward, lateral and yaw velocity requests; no task success sensor.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Interface
     - Behavior
   * - Observation
     - Empty dictionaries; the planner receives no camera or pose feedback.
   * - Action
     - ``robot_move(vx=0.4, vy=0.0, vyaw=0.0, duration_s=3.0)``.
   * - Reward / success
     - No native task-success evaluation; ``finish`` is a planner declaration only.
   * - Task prompt
     - Required nonblank natural-language ``--task``.

Installation and Resources
--------------------------

Use the existing host venv with RPent and its ``microduck`` extra installed.
Install missing prerequisites only with operator permission. Host commands below
use that venv's ``python``, not system Python or a new environment:

.. code-block:: bash

   python -c 'import sys; assert sys.prefix != sys.base_prefix, "Use the current host venv"; print(sys.executable)'

Python children use ``sys.executable`` unchanged. Container Python is separate;
the simulator and its pinned dependencies belong inside the image, not the host.
Docker must be accessible without sudo, through a local Linux Unix socket.
Remote engines/contexts are unsupported. Use the native Docker architecture;
there is no CPU-feature check or forced platform. The host UID:GID must be able
to write the output bind mount. No GPU, privileged mode, host networking or
Docker-socket mount is required.

Step 1: build separately from the repository root. Runtime never builds, pulls,
downloads or upgrades dependencies:

.. code-block:: bash

   docker build \
     --build-arg http_proxy --build-arg https_proxy --build-arg all_proxy --build-arg no_proxy \
     --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg ALL_PROXY --build-arg NO_PROXY \
     -f robots/microduck/docker/Dockerfile -t rpent-microduck:local .

Docker reads exported proxy values by name, preserving casing and bypass lists.
Do not substitute values into arguments, enable shell tracing or dump the
environment. Do not add proxy ARG/ENV declarations or explicit Dockerfile
references: these can persist sensitive values and defeat predefined-proxy
history/cache protections. Build-step proxies do not configure registry/base-image
pulls; the operator configures Docker daemon/BuildKit access separately.
Host loopback proxy addresses do not identify the same endpoint inside build
containers. Report reachability failures rather than rewriting addresses or using
host networking. Keep raw logs private and redact URLs/credentials before sharing.
The supervisor does not explicitly forward proxies or planner credentials;
operator Docker client defaults can still inject environment variables.
It does not sanitize arbitrary Docker configuration.

Verify the prepared image without network access:

.. code-block:: bash

   docker image inspect rpent-microduck:local --format '{{.Id}} {{index .Config.Labels "org.rpent.microduck.env-protocol"}}'
   docker run --rm --network none --entrypoint python rpent-microduck:local /opt/rpent/robots/microduck/docker/verify_image.py

Retain the immutable image ID and protocol label ``1``. Offline image verification
does not prove gait quality. Review image configuration/history privately for
proxy persistence. Rebuild explicitly after changing container sources or pins.
Pinned source/policy provenance and retained notices are described in
``robots/microduck/README.md``. Review code and model licenses independently;
no commercial permission is implied.

Run a Task
----------

Step 2: configure ``OPENAI_API_KEY`` and, for a custom OpenAI-compatible endpoint,
``OPENAI_BASE_URL``. The endpoint must serve the exact ``openai:deepseek-flash``
model; do not substitute another model. Use a fresh output directory:

.. code-block:: bash

   python -m rpent.cli.main --planner api --model openai:deepseek-flash --robot microduck \
     --task "move forward 5 seconds, and then move backward" --output-dir ./logs/xxxx

The ``rpent`` spelling is valid only when its entry-point script belongs to this
venv. ``--microduck-image`` overrides ``rpent-microduck:local``; the image must
already exist locally with protocol label ``1``. Omitted output paths use
``logs/<timestamp>_microduck``. Fresh empty local memory is selected automatically.
Dashboard, Flash and exploration are unsupported.

The intended calls are forward for an explicit five seconds, backward with
``vx=-0.4`` and omitted duration (default three seconds), then ``finish``.
The default forward velocity intent is ``vx=0.4`` m/s; ``vy=vyaw=0``.
All numbers must be finite and nonboolean: ``|vx|<=0.4`` m/s,
``|vy|<=0.10`` m/s, ``|vyaw|<=0.5`` rad/s, ``0<duration_s<=30`` seconds.
Commands run sequentially; a partly executed failed move is never automatically
retried. Empty observations cannot establish displacement, obstacle avoidance,
balance or task success. There is no direct joint control or physical-robot mode.

Smaller requested velocities remain valid but may yield negligible motion.
Velocity requests are open-loop intents, not measured speeds or a precision
motion guarantee. A model may choose different calls or fail to complete the
sequence; inspect the transcript and video rather than assuming model quality.

View Results
------------

Inspect ``transcript_microduck.json`` for ordered tool calls, empty observations
and errors, and ``states.json`` for root artifact registration. Normal close
finishes recording and publishes ``microduck.mp4`` before stopping the container.
The trunk-tracking camera with fixed viewing parameters records H.264/yuv420p
video at 640×480 and 25 fps, including SIT,
stand-up, waiting, commands and final settling. Recording is capped at 300
simulated seconds (queue 250); offline rendering has a 300-second deadline.
Inspect the video yourself; planner ``finish`` is not verified task success.

Diagnostics and Checks
----------------------

``.microduck/`` retains the staged video, trajectory, ``runtime.json`` (image/run
identity), ``supervisor.log``, ``container.log``, controller/renderer logs and
failure records when available. Finalization errors are recorded in
``finalization-error.json`` and are not silently converted to success. A missing
image reports the same build command above; never let a test build or pull it.
Model endpoint errors are distinct from simulator/controller/render errors.

Each run owns one UUID-named/labeled container, runs as the host UID:GID and
publishes an ephemeral loopback port. Normal shutdown removes only that container.
Parent death triggers cooperative supervisor cleanup while the supervisor
survives; forced shutdown cannot guarantee an MP4. If Docker is unavailable or
creation/cleanup times out, removal is unconfirmed: follow the exact name/label
inspection instructions in the failure record. Do not prune or remove unrelated
containers. Simultaneous supervisor/host death requires manual inspection.

Offline checks and opt-in CLI checks are documented in ``tests/README.md``.
The deterministic planner tests exercise integration, not model quality or gait
success. They require a manually prepared image and ``pytest-timeout``; each CLI
case is bounded at 600 seconds. Without opt-in they skip; missing prerequisites
with opt-in fail explicitly.
