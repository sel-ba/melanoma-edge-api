from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple token-bucket rate limiter per client IP.

    For edge devices with limited resources, rate limiting prevents
    accidental or malicious overload.
    """

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean expired entries
        self._buckets[client_ip] = [
            t
            for t in self._buckets[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._buckets[client_ip]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Too many requests. Rate limit exceeded.")

        self._buckets[client_ip].append(now)
        return await call_next(request)
