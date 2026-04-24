FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt /build/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /build/requirements.txt && \
    pip install sqlalchemy[asyncio] asyncpg loguru pydantic-settings

COPY alma_service /build/alma_service
COPY paano /build/paano
COPY scripts /build/scripts
COPY app/back/src /build/app/back/src

ENV PYTHONPATH=/build:/build/app/back/src

CMD ["python", "-m", "back.worker.loop"]
