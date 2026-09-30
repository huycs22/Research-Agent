"""Final-answer syntax, citation, and completion-contract validation."""

from __future__ import annotations

import re

TOOL_MARKUP = re.compile(
    r"</?(?:tool_call|arg_key|arg_value)>|<tool_call\b|\b(?:search_library|read_passages)\s*\(",
    re.I,
)


def cited_ids(text: str) -> set[str]:
    """Return exact passage IDs cited in an answer."""
    return set(re.findall(r"src-[a-f0-9]{24}", text))


def completion_status(answer: str, contract: dict | None = None) -> dict:
    """Evaluate a deterministic terminal-marker contract without an LLM judge."""
    contract = contract or {}
    labels = {str(label).upper() for label in contract.get("labels", [])}
    marker = str(contract.get("marker", "FINAL_CHOICE"))
    label_pattern = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    matches = (
        re.findall(
            rf"{re.escape(marker)}:\s*({label_pattern})(?![A-Z0-9_-])",
            answer,
            re.I,
        )
        if labels
        else []
    )
    selected = matches[0].upper() if len(matches) == 1 else None
    format_valid = not labels or (len(matches) == 1 and selected in labels)
    abstention = str(contract.get("abstention_label", "")).upper() or None
    return {
        "available": bool(answer.strip()),
        "format_required": bool(labels),
        "format_valid": format_valid,
        "selected_label": selected,
        "abstained": bool(selected and selected == abstention),
        "fallback_abstention": False,
        "marker_count": len(matches),
    }
