from __future__ import annotations

from fastapi import Request
from slowapi import Limiter


def _client_ip(request: Request) -> str:
    """Resolve real client IP behind the upstream reverse proxy.

    The deployment terminates TLS on a colleague's reverse proxy that forwards
    plain HTTP to the frontend; the frontend (Next.js) in turn proxies /api/*
    to the backend container. The closest hop FastAPI sees is the Next.js
    container, so without trusting forwarded headers every request would map
    to the same IP and a global rate-limit would lock everyone out at once.

    Resolution order: ``X-Forwarded-For`` (first hop, set by the edge), then
    ``X-Real-IP``, finally the direct peer (last fallback so the limiter never
    returns ``None``).
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    if request.client is not None:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=_client_ip)
