"""Durable, secret-free run records; conversation JSON remains backward compatible."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    id: str
    conversation_id: str
    question: str
    model: str
    embedding: str
    mode: str = "auto"
    route: str = "research"
    created_at: float = field(default_factory=time.time)
    status: str = "queued"
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    # Kept only so pre-migration run JSON remains readable; no limits are enforced.
    budget: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    pause_reason: str | None = None
    pending_question: str | None = None
    retry_at: float | None = None
    payload: dict = field(default_factory=dict)
    request_id: str | None = None
    stop_requested: bool = False
    lock: Any = field(default_factory=threading.RLock, repr=False)
    on_event: Any = field(default=None, repr=False)

    def add_event(self, stage, title, detail="", status="active"):
        with self.lock:
            for event in self.events:
                if event["status"] == "active":
                    event["status"] = (
                        status if status in {"error", "stopped", "paused"} else "complete"
                    )
            self.events.append({
                "id": len(self.events) + 1,
                "stage": stage,
                "title": title,
                "detail": detail,
                "status": status,
                "timestamp": time.time(),
            })
        if self.on_event:
            self.on_event(stage, title, detail, status)

    def as_dict(self, include_result=True):
        with self.lock:
            data = {
                key: value
                for key, value in vars(self).items()
                if key not in {"lock", "on_event"}
            }
            data["events"] = list(self.events)
            data["usage"] = dict(self.usage)
            if not include_result:
                data.pop("result", None)
            return data


class RunStorage:
    def __init__(self, home: Path):
        self.directory = home / "runs-v2"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def save(self, job):
        if not re.fullmatch(r"[a-f0-9]{12}", job.id):
            raise ValueError("Invalid run ID.")
        with self.lock:
            target = self.directory / f"{job.id}.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(job.as_dict(), ensure_ascii=False), "utf-8")
            temp.replace(target)

    def load(self):
        jobs = {}
        for path in self.directory.glob("*.json"):
            try:
                job = RunRecord(**json.loads(path.read_text("utf-8")))
                if path.stem != job.id or not re.fullmatch(r"[a-f0-9]{12}", job.id):
                    continue
                if job.status in {"running", "queued"}:
                    job.status, job.pause_reason = "paused", "restart"
                    job.error = "Server restarted. Resume to continue from saved work."
                    self.save(job)
                jobs[job.id] = job
            except (ValueError, TypeError, OSError):
                continue
        return jobs

    def delete(self, conversation_id):
        for path in self.directory.glob("*.json"):
            if json.loads(path.read_text("utf-8")).get("conversation_id") == conversation_id:
                path.unlink()
