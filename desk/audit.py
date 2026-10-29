"""Append-only action audit: what each agent (or the orchestrator) asked for, what the
gateway answered, and which business effects were committed.

This becomes ``result.action_audit[]`` and feeds the UI's live trace (listeners are
called on every append, e.g. to push SSE events). Entries are deep-copied in and out,
so nothing a caller does later can rewrite history. No chain-of-thought is stored:
only tool names, arguments, results and effects.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Any

Listener = Callable[[dict[str, Any]], None]


class AuditLog:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self.turn = 0
        self.listener_errors = 0

    def set_turn(self, turn_no: int) -> None:
        self.turn = turn_no

    def record(
        self,
        *,
        kind: str,
        agent: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: dict[str, Any] | None = None,
        committed_effects: list[dict[str, Any]] | None = None,
        visible_to_agent: bool = True,
        **extra: Any,
    ) -> dict[str, Any]:
        """Append one entry and return a copy of it (with its sequence number)."""
        with self._lock:
            entry: dict[str, Any] = {
                "seq": len(self._entries) + 1,
                "turn": self.turn,
                "kind": kind,
                "agent": agent,
                "tool": tool,
                "arguments": arguments or {},
                "result": result,
                "error": error,
                "committed_effects": committed_effects or [],
                "visible_to_agent": visible_to_agent,
                **extra,
            }
            entry = copy.deepcopy(entry)
            self._entries.append(entry)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(copy.deepcopy(entry))
            except Exception:  # noqa: BLE001 - a broken viewer must never break a business turn
                self.listener_errors += 1
        return copy.deepcopy(entry)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._entries)

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """Register a listener; returns a function that unsubscribes it."""
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    # Deliberately no __len__: an empty log must stay truthy so `audit or AuditLog()`
    # never silently replaces a caller's log with a new one.
