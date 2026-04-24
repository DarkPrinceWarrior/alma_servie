FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /build/app/back

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

COPY pyproject.toml /build/pyproject.toml
COPY README.md /build/README.md
COPY alma_service /build/alma_service

COPY app/back/pyproject.toml app/back/uv.lock app/back/README.md app/back/alembic.ini ./
COPY app/back/src ./src
COPY app/back/migrations ./migrations

RUN uv sync --frozen --no-dev

ENV PATH="/build/app/back/.venv/bin:$PATH"

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn back.main:app --host 0.0.0.0 --port 8000"]
