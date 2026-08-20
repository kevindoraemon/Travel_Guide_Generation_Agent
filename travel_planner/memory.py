"""Small, non-semantic memory store for the travel planner.

Ledger is append-only, views hold current state, and policy decides which
layers a graph node may read.  User/profile/skill lookups are exact by key.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_LOCK = threading.Lock()
_SESSION_KEYS = {"session_id", "channel", "temp_flags", "requested_skills"}


def _timestamp(value: str | datetime | None = None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _message_text(message: Any) -> str:
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type") or "message"
        content = message.get("content", "")
    else:
        role = getattr(message, "type", message.__class__.__name__.replace("Message", ""))
        content = getattr(message, "content", message)
    return f"{role}: {str(content)[:2000]}"


class MemoryPolicy:
    """Render only the layers needed by each consumer."""

    @staticmethod
    def render(layers: Mapping[str, Any], consumer: str) -> str:
        if consumer == "briefing":
            selected = {
                "temporary_flags": layers.get("session", {}).get("temp_flags", {}),
                "profile": layers.get("profile", {}),
                "recent_summary": layers.get("recent_summary", ""),
                "current_window": layers.get("sliding_window", []),
            }
        elif consumer == "writer":
            selected = {
                "profile": layers.get("profile", {}),
                "recent_summary": layers.get("recent_summary", ""),
                "reusable_skills": layers.get("skills", {}),
            }
        elif consumer == "tool":
            selected = {
                "session": layers.get("session", {}),
                "reusable_skills": layers.get("skills", {}),
            }
        else:
            raise ValueError(f"unknown memory consumer: {consumer}")
        return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


class MemoryStore:
    """Append events to a ledger and materialize exact-match user views."""

    def __init__(self, root: str | Path | None = None, window_size: int = 6, summary_chars: int = 1600):
        self.root = Path(root or os.environ.get("TRAVEL_MEMORY_DIR", "data/memory"))
        self.window_size = window_size
        self.summary_chars = summary_chars

    def load_layers(
        self,
        user_id: str | None,
        session: Mapping[str, Any] | None,
        messages: Iterable[Any],
    ) -> dict[str, Any]:
        if session is not None and not isinstance(session, Mapping):
            raise TypeError("session metadata must be a mapping")
        view = self.read_view(user_id) if user_id else self._empty_view("")
        requested = (session or {}).get("requested_skills", [])
        if not isinstance(requested, list):
            requested = []
        skills = {
            key: view["skills"][key]
            for key in requested
            if isinstance(key, str) and key in view["skills"]
        }
        profile = {key: item["value"] for key, item in view["profile"].items()}
        window = list(messages)[-self.window_size :]
        return {
            "session": {key: value for key, value in (session or {}).items() if key in _SESSION_KEYS},
            "profile": profile,
            "recent_summary": view.get("recent_summary", {}).get("value", ""),
            "sliding_window": [_message_text(message) for message in window],
            "skills": skills,
        }

    def save_turn(
        self,
        user_id: str | None,
        *,
        summary: str = "",
        profile_updates: Mapping[str, Any] | None = None,
        skill_results: Mapping[str, Any] | None = None,
        event_at: str | datetime | None = None,
        recorded_at: str | datetime | None = None,
    ) -> None:
        if not user_id:
            return
        self._validate_user_id(user_id)
        effective = _timestamp(event_at)
        recorded = _timestamp(recorded_at)
        events: list[dict[str, Any]] = []

        if profile_updates:
            if not isinstance(profile_updates, Mapping):
                raise TypeError("profile_updates must be a mapping")
            if any(not isinstance(field, str) or not field.strip() for field in profile_updates):
                raise ValueError("profile fields must be non-empty strings")
            events.append(self._event(user_id, "profile", dict(profile_updates), effective, recorded))
        if summary:
            events.append(self._event(user_id, "summary", summary[: self.summary_chars], effective, recorded))
        if skill_results is not None and not isinstance(skill_results, Mapping):
            raise TypeError("skill_results must be a mapping")
        for skill_key, result in (skill_results or {}).items():
            if not isinstance(skill_key, str) or not skill_key.strip():
                raise ValueError("skill keys must be non-empty strings")
            if isinstance(result, str):
                payload = {"path": result}
            elif isinstance(result, Mapping):
                payload = dict(result)
            else:
                raise TypeError(f"skill {skill_key!r} must be a path or mapping")
            if not payload.get("path"):
                raise ValueError(f"skill {skill_key!r} requires a result path")
            events.append(
                self._event(user_id, "skill", payload, effective, recorded, key=skill_key)
            )
        if events:
            self._append_and_rebuild(user_id, events)

    def read_view(self, user_id: str | None) -> dict[str, Any]:
        if not user_id:
            return self._empty_view("")
        self._validate_user_id(user_id)
        path = self._view_path(user_id)
        if not path.exists():
            return self._empty_view(user_id)
        view = json.loads(path.read_text(encoding="utf-8"))
        return view if view.get("user_id") == user_id else self._empty_view(user_id)

    def get_skill(self, user_id: str, skill_key: str) -> dict[str, Any] | None:
        """Exact lookup only; there is intentionally no fuzzy/semantic fallback."""
        return self.read_view(user_id)["skills"].get(skill_key)

    def _append_and_rebuild(self, user_id: str, events: list[dict[str, Any]]) -> None:
        ledger_path = self._ledger_path(user_id)
        view_path = self._view_path(user_id)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        view_path.parent.mkdir(parents=True, exist_ok=True)
        # ponytail: process-local lock; use a DB transaction if multi-process writers become real.
        with _LOCK:
            with ledger_path.open("a", encoding="utf-8") as stream:
                for event in events:
                    stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            view = self._reduce(user_id, self._read_ledger(ledger_path))
            temporary = view_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(view_path)

    @staticmethod
    def _reduce(user_id: str, events: Iterable[dict[str, Any]]) -> dict[str, Any]:
        view = MemoryStore._empty_view(user_id)
        for event in events:
            if event.get("user_id") != user_id:
                continue
            stamp = (event["event_at"], event["recorded_at"], event["event_id"])
            if event["kind"] == "profile":
                for field, value in event["value"].items():
                    current = view["profile"].get(field)
                    current_stamp = (
                        current["event_at"], current["recorded_at"], current["event_id"]
                    ) if current else ()
                    if stamp > current_stamp:
                        view["profile"][field] = {
                            "value": value,
                            "event_at": event["event_at"],
                            "recorded_at": event["recorded_at"],
                            "event_id": event["event_id"],
                        }
            elif event["kind"] == "summary":
                current = view.get("recent_summary")
                current_stamp = (
                    current["event_at"], current["recorded_at"], current["event_id"]
                ) if current else ()
                if stamp > current_stamp:
                    view["recent_summary"] = {
                        "value": event["value"],
                        "event_at": event["event_at"],
                        "recorded_at": event["recorded_at"],
                        "event_id": event["event_id"],
                    }
            elif event["kind"] == "skill":
                current = view["skills"].get(event["key"])
                current_stamp = (
                    current["event_at"], current["recorded_at"], current["event_id"]
                ) if current else ()
                if stamp > current_stamp:
                    view["skills"][event["key"]] = {
                        **event["value"],
                        "event_at": event["event_at"],
                        "recorded_at": event["recorded_at"],
                        "event_id": event["event_id"],
                    }
        view["built_at"] = _timestamp()
        return view

    @staticmethod
    def _read_ledger(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _event(
        user_id: str,
        kind: str,
        value: Any,
        event_at: str,
        recorded_at: str,
        *,
        key: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": uuid.uuid4().hex,
            "user_id": user_id,
            "kind": kind,
            "value": value,
            "event_at": event_at,
            "recorded_at": recorded_at,
        }
        if key is not None:
            event["key"] = key
        return event

    @staticmethod
    def _empty_view(user_id: str) -> dict[str, Any]:
        return {"user_id": user_id, "profile": {}, "recent_summary": {}, "skills": {}}

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 256:
            raise ValueError("user_id must be a non-empty string of at most 256 characters")

    @staticmethod
    def _slot(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    def _ledger_path(self, user_id: str) -> Path:
        return self.root / "ledger" / f"{self._slot(user_id)}.jsonl"

    def _view_path(self, user_id: str) -> Path:
        return self.root / "views" / f"{self._slot(user_id)}.json"


memory_store = MemoryStore()
