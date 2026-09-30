"""LangChain callback that meters every Cloudflare benchmark model request."""

from __future__ import annotations

import json
import threading

from langchain_core.callbacks import BaseCallbackHandler

from .cost import CostLedger, estimate_message_tokens


class BenchmarkCostCallback(BaseCallbackHandler):
    """Reserve before submission and reconcile provider usage after completion."""

    raise_error = True

    def __init__(
        self,
        ledger: CostLedger,
        model: str,
        maximum_output_tokens: int = 4096,
        stage: str = "generation",
    ):
        self.ledger = ledger
        self.model = model
        self.maximum_output_tokens = maximum_output_tokens
        self.stage = stage
        self._reservations: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _messages_payload(messages) -> list[dict]:
        payload = []
        for message in messages:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            payload.append({
                "role": getattr(message, "type", type(message).__name__),
                "content": content,
            })
        return payload

    def on_chat_model_start(self, _serialized, messages, *, run_id, **_kwargs):
        # The agent sends one message batch per invocation.
        payload = self._messages_payload(messages[0] if messages else [])
        reservation = self.ledger.reserve(
            stage=self.stage,
            model=self.model,
            input_tokens=estimate_message_tokens(payload),
            maximum_output_tokens=self.maximum_output_tokens,
        )
        with self._lock:
            self._reservations[str(run_id)] = reservation

    def _pop(self, run_id):
        with self._lock:
            return self._reservations.pop(str(run_id), None)

    def on_llm_end(self, response, *, run_id, **_kwargs):
        reservation = self._pop(run_id)
        if not reservation:
            return
        usage = dict((getattr(response, "llm_output", None) or {}).get("token_usage") or {})
        if not usage:
            try:
                usage = dict(response.generations[0][0].message.usage_metadata or {})
            except (AttributeError, IndexError, TypeError):
                usage = {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if input_tokens is None or output_tokens is None:
            self.ledger.retain_uncertain(
                reservation, "Cloudflare response omitted token usage; reservation retained."
            )
            return
        self.ledger.reconcile(
            reservation,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
        )

    def on_llm_error(self, error, *, run_id, **_kwargs):
        reservation = self._pop(run_id)
        if reservation:
            self.ledger.retain_uncertain(
                reservation,
                f"Cloudflare request outcome uncertain: {type(error).__name__}.",
            )
