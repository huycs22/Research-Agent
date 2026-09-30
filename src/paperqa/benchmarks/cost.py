"""Persistent request-level spending guard for DeepResearch Bench judges."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import tiktoken


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    max_output_tokens: int = 128_000


# Verified from the official OpenAI model pages on 2026-09-26.  The pinned
# benchmark selects these exact aliases.  A manifest records the verification
# date and URLs so a later experiment can deliberately refresh rather than
# silently changing historical costs.
OFFICIAL_OPENAI_PRICES = {
    "gpt-5.5": ModelPrice(5.00, 0.50, 30.00),
    "gpt-5.6-luna": ModelPrice(0.20, 0.02, 1.20),
    "gpt-5.4-mini": ModelPrice(0.75, 0.075, 4.50),
    "cloudflare/@cf/zai-org/glm-4.7-flash": ModelPrice(0.0605, 0.0605, 0.40, 4096),
}

PRICE_SOURCES = {
    "verified_at": "2026-09-26",
    "gpt-5.5": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "gpt-5.6-luna": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    "gpt-5.4-mini": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
    "cloudflare/@cf/zai-org/glm-4.7-flash": (
        "https://developers.cloudflare.com/workers-ai/models/glm-4.7-flash/"
    ),
}


class SpendingLimitReached(RuntimeError):
    """Raised before a paid request would exceed the experiment ceiling."""


def canonical_model(model: str) -> str:
    """Map OpenRouter's provider-prefixed aliases to the verified model price."""
    return model.removeprefix("openai/")


def estimate_message_tokens(messages: list[dict]) -> int:
    """Conservatively estimate Chat Completions input tokens."""
    encoder = tiktoken.get_encoding("cl100k_base")
    total = 8
    for message in messages:
        total += 8
        for value in message.values():
            if isinstance(value, str):
                total += len(encoder.encode(value, disallowed_special=()))
            elif value is not None:
                total += len(
                    encoder.encode(
                        json.dumps(value, ensure_ascii=False, sort_keys=True),
                        disallowed_special=(),
                    )
                )
    return total


def request_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
) -> float:
    model = canonical_model(model)
    try:
        price = OFFICIAL_OPENAI_PRICES[model]
    except KeyError as exc:
        raise ValueError(f"No verified price is configured for judge model {model!r}.") from exc
    cached = min(max(0, cached_input_tokens), max(0, input_tokens))
    uncached = max(0, input_tokens) - cached
    # GPT-5.5 and GPT-5.6 use long-context rates above 272K.  Charging the
    # documented 2x input and 1.5x output multiplier is conservative here.
    input_multiplier = 2.0 if input_tokens > 272_000 and model != "gpt-5.4-mini" else 1.0
    output_multiplier = 1.5 if input_tokens > 272_000 and model != "gpt-5.4-mini" else 1.0
    return (
        uncached * price.input_per_million * input_multiplier
        + cached * price.cached_input_per_million * input_multiplier
        + max(0, output_tokens) * price.output_per_million * output_multiplier
    ) / 1_000_000


class CostLedger:
    """Atomic local ledger; uncertain failures deliberately keep their reserve."""

    def __init__(self, path: Path, limit_usd: float = 10.0):
        self.path = path
        self.limit_usd = float(limit_usd)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({
                "version": 1,
                "limit_usd": self.limit_usd,
                "spent_usd": 0.0,
                "reserved_usd": 0.0,
                "entries": [],
            })

    def _read(self) -> dict:
        data = json.loads(self.path.read_text("utf-8"))
        if abs(float(data["limit_usd"]) - self.limit_usd) > 1e-9:
            raise ValueError(
                f"Existing ledger ceiling is ${data['limit_usd']:.2f}; refusing to silently change it."
            )
        return data

    def _write(self, data: dict) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        temporary.replace(self.path)

    def snapshot(self) -> dict:
        with self._lock:
            return self._read()

    def reserve(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int,
        maximum_output_tokens: int,
    ) -> str:
        priced_model = canonical_model(model)
        price = OFFICIAL_OPENAI_PRICES.get(priced_model)
        if not price:
            raise ValueError(f"No verified price is configured for judge model {model!r}.")
        maximum_output_tokens = min(maximum_output_tokens, price.max_output_tokens)
        amount = request_cost(priced_model, input_tokens, maximum_output_tokens)
        with self._lock:
            data = self._read()
            projected = float(data["spent_usd"]) + float(data["reserved_usd"]) + amount
            if projected > self.limit_usd + 1e-9:
                raise SpendingLimitReached(
                    f"The next {stage} request reserves ${amount:.4f}, which would raise "
                    f"the experiment to ${projected:.4f} above the ${self.limit_usd:.2f} ceiling."
                )
            reservation_id = uuid4().hex
            data["reserved_usd"] += amount
            data["entries"].append({
                "id": reservation_id,
                "stage": stage,
                "model": model,
                "priced_model": priced_model,
                "status": "reserved",
                "reserved_usd": amount,
                "estimated_input_tokens": input_tokens,
                "maximum_output_tokens": maximum_output_tokens,
                "created_at": time.time(),
            })
            self._write(data)
            return reservation_id

    def reconcile(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> None:
        with self._lock:
            data = self._read()
            entry = next(item for item in data["entries"] if item["id"] == reservation_id)
            if entry["status"] != "reserved":
                return
            actual = request_cost(
                entry.get("priced_model", entry["model"]),
                input_tokens,
                output_tokens,
                cached_input_tokens=cached_input_tokens,
            )
            data["reserved_usd"] = max(
                0.0, float(data["reserved_usd"]) - float(entry["reserved_usd"])
            )
            data["spent_usd"] += actual
            entry.update({
                "status": "reconciled",
                "actual_usd": actual,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "completed_at": time.time(),
            })
            self._write(data)

    def retain_uncertain(self, reservation_id: str, detail: str) -> None:
        with self._lock:
            data = self._read()
            entry = next(item for item in data["entries"] if item["id"] == reservation_id)
            if entry["status"] == "reserved":
                entry.update({
                    "status": "uncertain",
                    "detail": detail[:500],
                    "completed_at": time.time(),
                })
                self._write(data)

    def retain_open_reservations(self, detail: str, *, stage_prefix: str | None = None) -> int:
        """Mark reservations left open by a terminated worker as uncertain."""
        changed = 0
        with self._lock:
            data = self._read()
            for entry in data["entries"]:
                if entry["status"] != "reserved":
                    continue
                if stage_prefix and not str(entry.get("stage", "")).startswith(stage_prefix):
                    continue
                entry.update({
                    "status": "uncertain",
                    "detail": detail[:500],
                    "completed_at": time.time(),
                })
                changed += 1
            if changed:
                self._write(data)
        return changed
