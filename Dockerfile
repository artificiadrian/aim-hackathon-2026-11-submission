# The image the organisers build and run for the final (and for the check of every upload).
#
# Contract: configuration comes only from the environment. ARENA_BASE_URL and OPENROUTER_BASE_URL point at the
# organisers' gateway, ARENA_API_KEY and OPENROUTER_API_KEY are tokens that only work there, ARENA_RUN_KIND is
# dev (dry run) or final. Run to completion; the container is stopped after 30 minutes.
#
# Test it locally:
#   docker build -t harness . && docker run --rm --env-file .env -e ARENA_RUN_KIND=dev harness --tasks t01

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.11 /uv /usr/local/bin/uv
RUN useradd --create-home --uid 10001 harness && mkdir /app && chown harness:harness /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# Everything below runs as the non-root user; /app stays writable (runs/, .cache/).
USER harness
WORKDIR /app
COPY --chown=harness:harness . .
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /home/harness/.cache

ENTRYPOINT ["python", "main.py"]
CMD []
