FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn back.main:app --host 0.0.0.0 --port 8000"]
