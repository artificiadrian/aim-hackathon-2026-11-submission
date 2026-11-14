# The image the organisers build and run for the final (and for the check of every upload). Local runs use it too.
#
# Contract: configuration comes only from the environment. ARENA_BASE_URL and OPENROUTER_BASE_URL point at the
# organisers' gateway, ARENA_API_KEY and OPENROUTER_API_KEY are tokens that only work there, ARENA_RUN_KIND is
# dev (dry run) or final. Run to completion; the container is stopped after 30 minutes.
#
# Test it locally:
#   docker build -t mcnulty . && docker run --rm --env-file .env -e ARENA_RUN_KIND=dev mcnulty --tasks t01

FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv

# The sandbox libraries go into the system Python. The harness has its own /app/.venv, which does not see them, and
# at start src/mcnulty/sandbox.py copies this root filesystem (without /app, uv, ...) as the sandbox's root.
RUN uv pip install --system --no-cache \
        pandas==3.1.1 numpy==2.5.4 openpyxl==3.1.5 odfpy==1.4.1 python-docx==1.2.0 pypdf==6.20.0 \
        pillow==13.0.0 tzdata==2026.6 pyyaml==6.0.3 \
    && rm -rf /usr/local/bin/pip* /usr/local/lib/python3.12/site-packages/pip* /root/.cache \
    && python -c "import pandas, openpyxl, odf, docx, pypdf, PIL, yaml, zoneinfo; zoneinfo.ZoneInfo('Europe/Berlin')"
COPY sandbox/sitecustomize.py /usr/local/lib/python3.12/site-packages/sitecustomize.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# The harness runs as root: chroot needs it. Model-written code runs as uid 10001 (see src/mcnulty/sandbox.py).
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

ENTRYPOINT ["mcnulty", "run"]
CMD []
