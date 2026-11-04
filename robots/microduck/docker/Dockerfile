FROM rust:1.94.1-bookworm AS controller
RUN apt-get update && apt-get install -y --no-install-recommends \
    git pkg-config libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /src/microduck
RUN git init && git remote add origin https://github.com/pollen-robotics/microduck \
    && git fetch --depth=1 origin d13be2c1a2f4b1e0b0e499b08b92f870518b1384 \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = d13be2c1a2f4b1e0b0e499b08b92f870518b1384 \
    && cargo build --locked --release -p robotd --target-dir /src/target \
    && mkdir -p /export/bin /export/notices \
    && cp /src/target/release/robotd /export/bin/robotd \
    && cp scripts/seed-policies.sh /export/seed-policies.sh \
    && git rev-parse HEAD > /export/controller-revision \
    && find . -type f \( -iname 'license*' -o -iname 'copying*' -o -iname 'notice*' \) \
       -not -path './.git/*' -exec cp --parents '{}' /export/notices/ \; \
    && test -n "$(find /export/notices -type f -print -quit)"

FROM python:3.12-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libosmesa6 libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/local/bin/python -m pip install --no-cache-dir uv==0.9.0
COPY --from=controller /export/ /opt/microduck/runtime/
WORKDIR /opt/microduck/rl
RUN git init && git remote add origin https://github.com/pollen-robotics/microduck_rl \
    && git fetch --depth=1 origin bae39f5900b318f7e87fc485c30a31213cfa14eb \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = bae39f5900b318f7e87fc485c30a31213cfa14eb \
    && uv sync --project /opt/microduck/rl --python /usr/local/bin/python --frozen
ENV PATH="/opt/microduck/rl/.venv/bin:/opt/microduck/runtime/bin:${PATH}" \
    MUJOCO_GL=osmesa HOME=/tmp TMPDIR=/tmp
WORKDIR /opt/rpent
COPY pyproject.toml README.md LICENSE /opt/rpent/
COPY rpent/ /opt/rpent/rpent/
COPY robots/__init__.py /opt/rpent/robots/__init__.py
COPY robots/microduck/ /opt/rpent/robots/microduck/
RUN python robots/microduck/docker/verify_image.py inventory \
    && uv pip install --python /opt/microduck/rl/.venv/bin/python \
       --constraint /opt/microduck/packages-before.constraints '/opt/rpent[microduck]' \
    && uv pip check --python /opt/microduck/rl/.venv/bin/python
RUN POLICY_REPO=pollen-robotics/microduck-policies POLICY_VERSION=v8 POLICY_BASE_URL=https://huggingface.co/pollen-robotics/microduck-policies/resolve/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448 sh /opt/microduck/runtime/seed-policies.sh /opt/microduck/policies
RUN python robots/microduck/docker/verify_image.py policy-notices \
    && python robots/microduck/docker/verify_image.py prepare \
    && python robots/microduck/docker/verify_image.py
LABEL org.rpent.microduck.env-protocol="1"
ENTRYPOINT ["/opt/microduck/rl/.venv/bin/python", "-m", "robots.microduck.env_server", "--host", "0.0.0.0", "--port", "8000", "--output-dir", "/output"]
