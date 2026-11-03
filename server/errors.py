"""Errors the routes map to HTTP statuses."""

from __future__ import annotations


class SessionError(Exception):
    """A request the session cannot serve (mapped to an HTTP status by the routes)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
