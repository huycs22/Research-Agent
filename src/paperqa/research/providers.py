"""Language-model provider configuration without persisting credentials."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_CLOUDFLARE_MODEL = "cloudflare/@cf/zai-org/glm-4.7-flash"
REASONING_EFFORTS = {"low", "medium", "high"}

MODEL_OPTIONS = (
    {
        "value": DEFAULT_CLOUDFLARE_MODEL,
        "label": "Cloudflare Workers AI · GLM-4.7-Flash",
        "provider": "cloudflare",
        "description": "Fast multilingual reasoning and tool calling on Workers AI.",
    },
)


def cloudflare_configured() -> bool:
    return bool(os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"))


def model_configured(model_name: str) -> bool:
    return model_name == DEFAULT_CLOUDFLARE_MODEL and cloudflare_configured()


def web_search_configured(model_name: str) -> bool:
    """Public arXiv/Crossref discovery is available to every supported model."""
    if model_name != DEFAULT_CLOUDFLARE_MODEL:
        raise ValueError("Unsupported language-model provider.")
    return True


def default_model() -> str:
    return DEFAULT_CLOUDFLARE_MODEL


def build_chat_model(model_name: str, reasoning_effort: str = "medium"):
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError("Reasoning effort must be low, medium, or high.")
    if model_name != DEFAULT_CLOUDFLARE_MODEL:
        raise ValueError("Only Cloudflare GLM-4.7-Flash is supported by the agent.")
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account or not token:
        raise ValueError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set in the server environment."
        )
    return ChatOpenAI(
        model=model_name.removeprefix("cloudflare/"),
        api_key=token,
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
        temperature=0,
        max_tokens=4096,
        max_retries=0,
        timeout=90,
        reasoning_effort=reasoning_effort,
        extra_body={"options": {"rejectIfBusy": True}},
        metadata={"provider": "cloudflare"},
    )


def is_cloudflare_model(model) -> bool:
    return getattr(model, "metadata", {}).get("provider") == "cloudflare"
