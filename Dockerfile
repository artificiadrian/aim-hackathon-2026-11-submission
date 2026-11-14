# The image the organisers build and run for the final (and for the check of every upload). Local runs use it too.
#
# Contract: configuration comes only from the environment. ARENA_BASE_URL and OPENROUTER_BASE_URL point at the
# organisers' gateway, ARENA_API_KEY and OPENROUTER_API_KEY are tokens that only work there, ARENA_RUN_KIND is
# dev (dry run) or final. Run to completion; the container is stopped after 30 minutes.
#
# Test it locally:
#   docker build -t mcnulty . && docker run --rm --env-file .env -e ARENA_RUN_KIND=dev mcnulty --tasks t01

# Pinned by digest (the tag can move between our build and the organisers' build).
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
COPY --from=ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 /uv /usr/local/bin/uv

# The sandbox package (packages/sandbox: the helpers ws, plus sitecustomize.py copied below) goes into the system Python with its locked
# dependencies. The harness has its own /app/.venv, which does not see them, and at start src/mcnulty/agent/sandbox.py
# copies this root filesystem (without /app, uv, ...) as the sandbox's root.
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/sandbox packages/sandbox
RUN uv export --frozen --package mcnulty-sandbox --no-dev --no-emit-workspace --no-hashes -o /tmp/sandbox.txt \
    && uv pip install --system --no-cache -r /tmp/sandbox.txt \
    && uv pip install --system --no-cache --no-deps ./packages/sandbox \
    && rm -rf /tmp/sandbox.txt /usr/local/bin/pip* /usr/local/lib/python3.12/site-packages/pip* /root/.cache \
    && python -c "import pandas, openpyxl, odf, docx, pypdf, PIL, yaml, pydantic, zoneinfo, ws.files; zoneinfo.ZoneInfo('Europe/Berlin')"
COPY packages/sandbox/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# The harness runs as root: chroot needs it. Model-written code runs as uid 10001 (see src/mcnulty/agent/sandbox.py).
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["mcnulty", "run"]
CMD []
