"""Resource accounting and read-only tool boundaries (not model-enforced rules)."""

from __future__ import annotations

import ast
import asyncio
import ipaddress
import logging
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger(__name__)
MAX_TRANSIENT_RETRY_DELAY = 30.0


class PauseRun(Exception):
    def __init__(self, reason: str, detail: str, retry_at: float | None = None):
        super().__init__(detail)
        self.reason, self.detail, self.retry_at = reason, detail, retry_at


def redact(value: Any) -> str:
    text = str(value)
    for name, secret in os.environ.items():
        if name.endswith(("_API_KEY", "_TOKEN", "_SECRET")) and len(secret) >= 8:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?:gsk_|sk-(?:proj-)?)[A-Za-z0-9_-]+", "[redacted]", text)
    return text[:1000]


def provider_retry_delay(exc: Exception) -> float | None:
    """Parse provider retry guidance without assuming a particular SDK."""
    detail = redact(exc)
    headers = getattr(getattr(exc, "response", None), "headers", {})
    delay = None
    try:
        delay = float(headers.get("retry-after", ""))
    except (ValueError, TypeError):
        from email.utils import parsedate_to_datetime

        try:
            delay = max(
                0,
                parsedate_to_datetime(headers.get("retry-after", "")).timestamp() - time.time(),
            )
        except (ValueError, TypeError, OverflowError):
            pass
        match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", detail, re.I)
        if match:
            delay = int(match[1] or 0) * 3600 + int(match[2] or 0) * 60 + float(match[3])
    return delay


def provider_pause(exc: Exception) -> PauseRun | None:
    if getattr(exc, "status_code", None) != 429:
        return None
    detail = redact(exc)
    category = (
        "daily quota"
        if re.search(r"tokens per day|TPD|requests per day|RPD", detail, re.I)
        else "rate limit"
    )
    delay = provider_retry_delay(exc)
    return PauseRun(
        "provider_quota",
        f"Provider {category}: {detail}",
        time.time() + delay if delay is not None else None,
    )


def provider_connection_error(exc: Exception) -> bool:
    """Recognize transport failures from the Cloudflare-compatible client."""
    import httpx
    from openai import APIConnectionError as OpenAIConnectionError

    connection_types = (
        TimeoutError,
        ConnectionError,
        OpenAIConnectionError,
        httpx.TransportError,
    )
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, connection_types):
            return True
        current = current.__cause__ or current.__context__
    return False


def provider_error_chain(exc: Exception) -> str:
    """Return a secret-redacted exception chain for local diagnostics."""
    parts = []
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        parts.append(f"{type(current).__module__}.{type(current).__name__}: {redact(current)}")
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


class Meter:
    """Track provider usage without imposing a local research budget."""

    def __init__(self, usage: dict, save: Callable[[], None]):
        self.usage, self.save = usage, save
        self.started = time.monotonic()
        self.initial_seconds = float(usage.get("seconds", 0))
        self.closed = False
        self.pending: set[asyncio.Future] = set()
        for key in (
            "tokens",
            "model_calls",
            "tool_calls",
            "estimated_tokens",
            "input_tokens",
            "output_tokens",
        ):
            usage.setdefault(key, 0)

    def tick(self):
        self.usage["seconds"] = round(self.initial_seconds + time.monotonic() - self.started, 2)

    def reserve(self, *, tokens: int = 0, model: bool = False, tool: bool = False):
        if self.closed:
            raise asyncio.CancelledError("Run is no longer active.")
        self.tick()
        additions = {"tokens": tokens, "model_calls": int(model), "tool_calls": int(tool)}
        for key, extra in additions.items():
            self.usage[key] = self.usage.get(key, 0) + extra
        self.usage["estimated_tokens"] += tokens
        self.save()

    def reconcile(self, reserved: int, usage: dict | None):
        if usage and usage.get("total_tokens") is not None:
            total = int(usage["total_tokens"])
            self.usage["tokens"] += total - reserved
            self.usage["estimated_tokens"] -= reserved
            self.usage["input_tokens"] += int(
                usage.get("input_tokens", usage.get("prompt_tokens", 0))
            )
            self.usage["output_tokens"] += int(
                usage.get("output_tokens", usage.get("completion_tokens", 0))
            )
        self.tick()
        self.save()

    async def call(
        self,
        fn,
        input_text: str,
        output_limit: int = 2048,
        *,
        attempts: int = 2,
        timeout: float = 90,
    ):
        # Estimated tokenizer plus a margin; not exact provider billing.
        import tiktoken

        reserved = (
            int(
                len(
                    tiktoken.get_encoding("cl100k_base").encode(
                        input_text, disallowed_special=()
                    )
                )
                * 1.2
            )
            + output_limit
        )
        if attempts < 1:
            raise ValueError("At least one provider attempt is required.")
        for attempt in range(attempts):
            self.reserve(tokens=reserved, model=True)
            try:
                request = asyncio.ensure_future(fn())
                self.pending.add(request)
                try:
                    result = await asyncio.wait_for(request, timeout=timeout)
                finally:
                    self.pending.discard(request)
                usage = getattr(result, "usage_metadata", None)
                if getattr(result, "result", None):
                    usage = getattr(result.result[-1], "usage_metadata", None)
                if isinstance(result, dict):
                    usage = result.get("usage") or getattr(
                        result.get("raw"), "usage_metadata", None
                    )
                self.reconcile(reserved, usage)
                return result
            except Exception as exc:
                pause = provider_pause(exc)
                if pause:
                    raise pause from exc
                connection_failure = provider_connection_error(exc)
                transient = connection_failure or getattr(exc, "status_code", 0) in {
                    500,
                    502,
                    503,
                    504,
                }
                if not transient or attempt == attempts - 1:
                    logger.warning("Provider request failed: %s", provider_error_chain(exc))
                    if connection_failure:
                        retry_count = attempts - 1
                        retry_detail = (
                            "on the first attempt"
                            if retry_count == 0
                            else f"after {retry_count} "
                            + ("retry" if retry_count == 1 else "retries")
                        )
                        raise PauseRun(
                            "provider_connection",
                            "The language-model provider could not be reached "
                            f"{retry_detail}. "
                            "Check outbound network access, then resume this run.",
                        ) from exc
                    if transient:
                        retry_count = attempts - 1
                        retry_detail = (
                            "the first attempt"
                            if retry_count == 0
                            else f"{retry_count} "
                            + ("retry" if retry_count == 1 else "retries")
                        )
                        status = getattr(exc, "status_code", "unknown")
                        raise PauseRun(
                            "provider_unavailable",
                            "The language-model provider returned a temporary "
                            f"{status} error after {retry_detail}. Resume this run later; "
                            "completed search and reading work is saved.",
                        ) from exc
                    raise
                retry_delay = provider_retry_delay(exc)
                if retry_delay is not None and retry_delay > MAX_TRANSIENT_RETRY_DELAY:
                    status = getattr(exc, "status_code", "unknown")
                    raise PauseRun(
                        "provider_unavailable",
                        "The language-model provider requested a retry delay longer than the "
                        "bounded in-process retry window. Resume this run after the provider's "
                        "retry time; completed work is saved.",
                        time.time() + retry_delay,
                    ) from exc
                await asyncio.sleep(retry_delay if retry_delay is not None else 2**attempt)

    def close(self):
        """Also cancel remote embedding tasks submitted by Chroma's executor."""
        self.closed = True
        for request in tuple(self.pending):
            request.cancel()


def calculate(expression: str) -> float:
    if len(expression) > 200:
        raise ValueError("Maximum expression length is 200 characters.")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError("Expression is too complex.")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 12:
                    raise ValueError("Exponent must be between -12 and 12.")
                value = left**right
            elif isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.Mod):
                value = left % right
            else:
                raise ValueError("Only arithmetic is allowed.")
        else:
            raise ValueError("Only arithmetic is allowed.")
        if isinstance(value, complex) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError("Calculation is outside the supported numeric range.")
        return value

    return evaluate(tree)


def public_url(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Only public HTTP(S) source URLs are allowed.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard web ports are allowed.")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() == "localhost" or parsed.hostname.lower().endswith((
            ".localhost",
            ".local",
        )):
            raise ValueError("Private-network sources are not allowed.") from None
    else:
        if not address.is_global:
            raise ValueError("Private-network sources are not allowed.")
    return url


class PublicResolver(aiohttp.resolver.ThreadedResolver):
    async def resolve(self, host, port=0, family=0):
        answers = await super().resolve(host, port, family)
        if not answers or any(
            not ipaddress.ip_address(item["host"]).is_global for item in answers
        ):
            raise ValueError("Source resolved to a private network address.")
        return answers


def readable_content_type(value: str) -> bool:
    """Accept bounded human-readable responses, including vendor XML/JSON MIME types."""
    kind = value.lower()
    return any(
        item in kind
        for item in (
            "text/",
            "application/pdf",
            "application/json",
            "application/xml",
            "+json",
            "+xml",
        )
    )


async def fetch_source(url: str, limit: int = 8_000_000) -> tuple[bytes, str, str]:
    """Validate DNS at connection time; never reconnect using unchecked DNS."""
    connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False)
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=False,
        cookie_jar=aiohttp.DummyCookieJar(),
        timeout=aiohttp.ClientTimeout(total=30),
    ) as client:
        for _ in range(4):
            public_url(url)
            async with client.get(url, allow_redirects=False) as response:
                if response.status in {301, 302, 303, 307, 308}:
                    from urllib.parse import urljoin

                    url = urljoin(url, response.headers.get("Location", ""))
                    continue
                response.raise_for_status()
                kind = response.headers.get("Content-Type", "").lower()
                if not readable_content_type(kind):
                    raise ValueError("Source is not readable text or PDF.")
                data = bytearray()
                async for part in response.content.iter_chunked(65536):
                    data.extend(part)
                    if len(data) > limit:
                        raise ValueError("Source exceeds the 8 MB reading limit.")
                return bytes(data), kind, str(response.url)
    raise ValueError("Too many redirects.")
