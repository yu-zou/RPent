# Test suite structure

RPent keeps fast, offline unit tests under `unit_tests`. Real-simulator smoke
tests, model component and policy-chain tests, and explicit real-robot
diagnostics belong under `e2e_tests`. Unit tests mirror the production module
they exercise; GPU tests are grouped by robot stack. Environment smoke tests
can run without model checkpoints.

## Directory layout

```text
tests/
├── README.md
├── e2e_tests/
│   ├── common.py         # shared assertions and runtime lifecycle helpers
│   ├── run_gpu_suite.sh  # clean-environment entry point for one robot stack
│   ├── dual_franka/      # opt-in real-robot VLA diagnostic console
│   ├── libero/
│   ├── robocasa/
│   └── robotwin/
└── unit_tests/
    ├── rpent/             # tests for the core rpent package
    │   ├── cli/
    │   ├── dashboard/
    │   ├── memory/
    │   ├── planner/
    │   ├── robots/
    │   ├── session/
    │   ├── tools/
    │   └── utils/
    └── robots/           # tests for the top-level robot extensions
        ├── libero/
        ├── robocasa/
        └── robotwin/
```

Directories that do not yet have tests do not need empty placeholders. Add
them when the first test for that module lands.

## Placement rules

- Mirror core modules under `tests/unit_tests/rpent/`. For example, tests for
  `rpent.utils.rpc` belong in `tests/unit_tests/rpent/utils/rpc/`.
- Mirror top-level robot extensions under `tests/unit_tests/robots/`. Put
  robot-specific coverage in that directory's `<robot>/` child.
- Place cross-layer tests with the primary contract owner. Registry and config
  contracts belong to `rpent/robots/`; extension toolkit and schema contracts
  belong to `robots/`.
- Keep one-off fakes in the test module that uses them. Put shared fixtures in
  the nearest `conftest.py`: use `tests/conftest.py` only for suite-wide
  fixtures and a module directory's `conftest.py` for local fixtures.
- Name files after the behavior they verify: `*_contracts.py` for stable API
  contracts, `*_loopback.py` for real local transports, `*_lifecycle.py` for
  resource ownership, and `*_smoke.py` for installation or startup checks.
- Put real-simulator smoke tests, model component tests and policy-chain tests
  under `tests/e2e_tests/<robot>/`. Keep environment smoke tests independent of
  fixtures that require model checkpoints. Reusable validation and lifecycle
  code belongs in `tests/e2e_tests/common.py`.

Unit tests must run offline on an ordinary CPU machine. They may cross a real
local boundary, such as a loopback TCP connection or child process, but must
not contact external services.

Run the complete suite with:

```bash
python -m pytest tests/unit_tests -v
```

## Microduck offline and opt-in CLI checks

Use the existing current host venv for every host Python command, including
children (which use `sys.executable` unchanged). Do not use system Python,
`uv run`, a new environment or automatic tool installation. RPent and the
`microduck` extra are prerequisites; opt-in video decoding also requires
`imageio`. Missing tools require operator permission to install in this venv.
The simulator Python environment is inside the separately prepared image.

```bash
python -c 'import sys; assert sys.prefix != sys.base_prefix, "Use the current host venv"; print(sys.executable)'
python -m pytest tests/unit_tests/robots/microduck -v
python -m pytest tests/e2e_tests/microduck/test_cli.py -v
```

The second command is offline. The third runs offline scripted-CLI contracts
and skips the four live CLI cases unless `RPENT_MICRODUCK_E2E=1`. Collection
does not import optional video/simulator dependencies. Offline tests alone do
not establish live host integration, gait success or model quality.

Prepare the image manually using the exact
proxy-aware build command in [the robot guide](../robots/microduck/README.md).
Ensure local Linux Unix-socket Docker permissions and output UID:GID write
access. Then, with `pytest-timeout` available in this venv:

```bash
RPENT_MICRODUCK_E2E=1 python -m pytest tests/e2e_tests/microduck -v
```

`MICRODUCK_IMAGE` overrides the test image (default `rpent-microduck:local`).
Opt-in without an image or timeout plugin explicitly fails; an unknown timeout
mark is not an enforced deadline. CLI cases each have a 600-second timeout;
the separate Env smoke has a 480-second timeout. Tests never build, pull or
download. The CLI cases use the real RobotSpec, supervisor, container and Env,
injecting only a deterministic planner at the existing CLI factory seam.
They check +0.4 forward5/-0.4 backward default3/finish ordering, empty observations,
registered decodable root video and startup frames, planner/publication failures,
parent death and absence of owned containers. Startup contact sheets support
manual stand-up/framing inspection; they do not prove gait success. Forced parent
death need not produce a video. Tests never prune or remove unrelated resources.

These tests make no provider requests and require no planner credentials. For
the separate named-model CLI demonstration, configure `OPENAI_API_KEY` and,
if using a custom OpenAI-compatible endpoint, `OPENAI_BASE_URL`. That endpoint
must serve the exact `openai:deepseek-flash` model. A passing deterministic
integration test does not guarantee the model will choose the intended calls.

### Current-venv documentation and lint checks

Documentation prerequisites are pinned in `docs/requirements.txt`; Ruff must be
0.15.22. Install only with operator approval. Check changed paths read-only first:

```bash
python -m ruff --version
python -m ruff check --preview -- path/to/changed.py
python -m ruff format --check -- path/to/changed.py
python -m sphinx -W --keep-going docs/source-en docs/build/html-en
python -m sphinx -W --keep-going docs/source-zh docs/build/html-zh
```

After the owner has reviewed and staged eligible new files, the hook-equivalent
full tracked-file commands below may change files. Do not run them over unrelated
work without review; inspect the diff and rerun tests afterward:

```bash
git ls-files -z -- '*.py' '*.pyi' | xargs -0 -r python -m ruff check --preview --fix --
git ls-files -z -- '*.py' '*.pyi' | xargs -0 -r python -m ruff format --
```

These are current-venv Ruff equivalents, not a claim that the pre-commit wrapper
or commit-message hook ran. The wrapper provisions separate hook environments
and needs an explicit environment-policy exception under this workflow.
Inspect rendered Chinese inline markup as well as both warning-as-error builds.

## RoboCasa environment smoke tests

RoboCasa separates simulator checks, model component checks and the full call chain:

| Tests | Coverage | Requirements |
| --- | --- | --- |
| `test_environment_smoke.py` | Three scenes, navigation RGB-D/world map and mobile-camera tracking | RoboCasa extra, assets and GPU/EGL; no model weights or planner credentials |
| `test_components.py` | Environment observations over RPC and RLDX inference | Model checkpoint, assets and GPU/EGL |
| `test_policy_chain.py` | CLI → tools → RLDX → simulator action → `finish` | Model checkpoint, assets and GPU/EGL; local scripted planner |

To check simulator installation and interfaces independently, install the
RoboCasa extra and assets, then run:

```bash
RPENT_RUN_ROBOCASA_INTEGRATION=1 \
  pytest tests/e2e_tests/robocasa/test_environment_smoke.py -v
```

This command needs neither `RLDX_MODEL_PATH` nor `RPENT_E2E_OUTPUT_DIR`. Without
`RPENT_RUN_ROBOCASA_INTEGRATION=1`, the four smoke cases are skipped.

The four cases cover `OpenDrawer`, `NavigateKitchen`, and
`PickPlaceCounterToCabinet` at seed 1, plus mobile-camera movement. Task checks
verify construction/reset, 12D actions, operation cameras, navigation RGB-D/world
map, the success predicate, and clean close. The camera check verifies pose and
image changes after eight base steps.

See the RoboCasa usage guide in
[English](../docs/source-en/rst_source/simulators/robocasa.rst#environment-smoke-tests)
or [Chinese](../docs/source-zh/rst_source/simulators/robocasa.rst#environment-smoke-tests)
for setup and test prerequisites.

## Embodied GPU E2E tests

The runner must expose one GPU and the checkpoint and simulator assets required
by the selected target. Run one GPU suite with two new output paths:

```bash
export CUDA_VISIBLE_DEVICES=<one-physical-GPU-ordinal>
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

bash tests/e2e_tests/run_gpu_suite.sh \
  <libero-pro|robocasa|robotwin> \
  /path/to/new-output-dir \
  /path/to/new-venv-root
```

For RoboCasa, the runner enables the four environment smoke cases alongside
the two component tests and one policy-chain test. The existing
[GPU CI](../.github/workflows/embodied-gpu-e2e-run.yml) uses this same runner for
all seven checks. The scripted planner needs no external planner credentials.

## Dual-Franka VLA diagnostic console

Run the real-robot diagnostic explicitly from a source checkout after installing
its robot dependencies and configuring the checkpoint, dataset normalization
statistics, cameras, reachable robot environment, and easy_handeye YAML paths under
`perception.calibration` in the robot config:

```bash
python -m tests.e2e_tests.dual_franka.dual_franka_vla \
  --robot-config /path/to/robot.yaml \
  --vla-model-path /path/to/checkpoint --vla-repo-id org/dataset
```

This console is not collected by pytest or included in the automated GPU suite.
Startup does not reset the robot. Enter `run N` for N prediction chunks, `run`
for continuous rollout, `reset` for an operator-confirmed reset, or `quit` to exit.
The instruction comes from `vla.instruction` in the robot config or `--instruction`,
not a registered task. Ctrl-C, SIGTERM, or the printed `STOP` file stops rollout;
stop checks occur between individual actions, but an in-flight RPC may still finish.
Inputs and predictions are saved as JSON/NPZ. The prediction chunk length comes
from `vla.status`; `--expected-action-steps` overrides it. The console starts only
env and VLA components, with no SAM3 dependency.
Offline regression tests remain in `unit_tests/robots/dual_franka/`.

See the deployment prerequisites in the
[English](../docs/source-en/rst_source/real_world_robots/dual_franka.rst) or
[Chinese](../docs/source-zh/rst_source/real_world_robots/dual_franka.rst) guide.
