"""Local semantic query routing that never calls a hosted model."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
from typing import Literal

from .library import DEFAULT_EMBEDDING, embeddings_for

logger = logging.getLogger(__name__)
MAX_LOCAL_REPLY_WORDS = 28

Intent = Literal[
    "greeting",
    "gratitude",
    "farewell",
    "identity",
    "capabilities",
    "usage",
    "wellbeing",
    "casual",
    "research",
    "unknown",
]
PromptMode = Literal["auto", "research", "chat"]
PromptRoute = Literal["research", "conversation"]


@dataclass(frozen=True)
class PromptDecision:
    route: PromptRoute
    reason: str


@dataclass(frozen=True)
class SemanticIntent:
    intent: Intent
    confidence: float
    method: Literal["semantic", "fallback"]
    scores: dict[str, float]


_PROTOTYPES: dict[Intent, tuple[str, ...]] = {
    "greeting": (
        "hello, nice to meet you",
        "good morning",
        "hey there",
        "I just wanted to say hello",
    ),
    "gratitude": (
        "thank you for your help",
        "I appreciate your assistance",
        "thanks, that was useful",
        "it was kind of you to help me",
    ),
    "farewell": (
        "goodbye for now",
        "see you later",
        "I am finished, bye",
    ),
    "identity": (
        "who are you",
        "what is your name",
        "tell me what kind of assistant you are",
    ),
    "capabilities": (
        "what are your abilities",
        "what kinds of tasks can you perform",
        "describe your features and tools",
        "what services do you provide",
        "how are you able to help me",
        "can you search papers and the public web",
        "are you able to browse the web and inspect papers",
        "what are you capable of doing",
    ),
    "usage": (
        "how do I use this application",
        "how can I upload a PDF document",
        "where do I add papers to the library",
        "help me get started with PaperQA",
    ),
    "wellbeing": (
        "how are you doing today",
        "how is it going",
        "what is up",
    ),
    "casual": (
        "tell me a funny joke",
        "let us have a casual conversation",
        "write a short poem for me",
        "what did we talk about earlier",
    ),
    "research": (
        "summarize the findings of this research paper with citations",
        "compare the methods datasets and results across these studies",
        "find supporting evidence in my scientific document library",
        "what does the academic literature report about this question",
        "explain retrieval augmented generation using primary sources",
        "search for recent research and cite the sources",
        "analyze a scientific topic using the uploaded papers",
        "read the relevant passages and answer the research question",
    ),
    "unknown": (),
}

_LOCAL_INTENTS = {
    "greeting",
    "gratitude",
    "farewell",
    "identity",
    "capabilities",
    "usage",
    "wellbeing",
}
_RESEARCH_TERMS = re.compile(
    r"\b(?:cite|citation|papers?|stud(?:y|ies)|articles?|compare|summari[sz]e|"
    r"analy[sz]e|evidence|literature|sources?|datasets?|methodology|results?|"
    r"library|retrieve|retrieval|rag|self-rag|paperqa2)\b",
    re.IGNORECASE,
)
_CAPABILITY_FRAMING = re.compile(
    r"\b(?:your\s+(?:abilities|capabilities|features|tools)|capable\s+of|"
    r"what\s+can\s+you\s+do|how\s+can\s+you\s+help|what\s+do\s+you\s+do|"
    r"(?:can\s+you|are\s+you\s+able\s+to)\s+(?:search|browse|read|compare|cite|inspect))\b",
    re.IGNORECASE,
)
_SPECIFIC_RESEARCH_TARGET = re.compile(
    r"\b(?:about|regarding|uploaded|attached|these|those|rag|self-rag|paperqa2)\b",
    re.IGNORECASE,
)
_USER_IDENTITY_QUERY = re.compile(
    r"^(?:who\s+am\s+i|what(?:'s|\s+is)\s+my\s+name|"
    r"do\s+you\s+remember\s+my\s+name|can\s+you\s+remember\s+my\s+name)[?.!\s]*$",
    re.IGNORECASE,
)
_NATURAL_CAPABILITY_QUERY = re.compile(
    r"^(?:(?:hi|hello|hey)[,!]?\s+)?"
    r"(?:(?:before|prior\s+to)\s+we\s+(?:start|begin)[,!]?\s+)?"
    r"(?:what\s+can\s+you\s+(?:do|help\s+me\s+with)|"
    r"how\s+can\s+you\s+help\s+me)[?.!\s]*$",
    re.IGNORECASE,
)
_SOCIAL_PROMPT = re.compile(
    r"^(?:(?:hi|hello|hey|hiya|greetings|yo)(?:\s+there)?|"
    r"(?:good\s+)?(?:morning|afternoon|evening)|"
    r"(?:thanks|thank\s+you|many\s+thanks)(?:[,!]?\s+(?:that\s+(?:helps|helped)|for\s+.+))?|"
    r"bye|goodbye|see\s+you|talk\s+later|how\s+are\s+you|what(?:'s|\s+is)\s+up|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|help(?:\s+me)?)"
    r"[!.?\s]*$",
    re.IGNORECASE,
)
_GREETING_PREFIX = re.compile(
    r"^(?:hi|hello|hey|hiya|greetings|yo)(?:\s+there)?(?:[!,.?]+\s*|\s+)(?P<rest>.+)$",
    re.IGNORECASE,
)
_RESEARCH_FOLLOWUP = re.compile(
    r"\b(?:cite|compare|summari[sz]e|analyse|analyze|search|retrieve|"
    r"according\s+to|what\s+does\s+the\s+paper|find\s+evidence)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right, strict=False)))


@lru_cache(maxsize=1)
def _prototype_vectors() -> dict[Intent, tuple[list[float], ...]]:
    embedding = embeddings_for(DEFAULT_EMBEDDING)
    flattened = [item for values in _PROTOTYPES.values() for item in values]
    vectors = iter(embedding.embed_documents(flattened))
    return {
        intent: tuple(next(vectors) for _ in values)
        for intent, values in _PROTOTYPES.items()
        if values
    }


@lru_cache(maxsize=512)
def semantic_intent(text: str) -> SemanticIntent:
    """Classify a query with local MiniLM embeddings and no hosted request."""
    normalized = _normalize(text)
    if not normalized:
        return SemanticIntent("unknown", 0.0, "fallback", {})
    try:
        query = embeddings_for(DEFAULT_EMBEDDING).embed_query(normalized)
        scores = {
            intent: max(_dot(query, prototype) for prototype in prototypes)
            for intent, prototypes in _prototype_vectors().items()
        }
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Semantic routing unavailable; using safe fallback: %s", exc)
        return SemanticIntent("unknown", 0.0, "fallback", {})
    intent, confidence = max(scores.items(), key=itemgetter(1))
    # Low-confidence text stays on the safe default route. Local canned replies
    # require a higher threshold because a false research classification is less
    # harmful than silently answering a genuine research request without evidence.
    threshold = 0.54 if intent in _LOCAL_INTENTS else 0.48
    if confidence < threshold:
        intent = "unknown"
    return SemanticIntent(intent, round(confidence, 4), "semantic", scores)


def local_conversation_answer(question: str) -> tuple[str, SemanticIntent] | None:
    """Return a safe canned answer for semantic social/help intents."""
    normalized = _normalize(question)
    if (
        not normalized
        or len(normalized.split()) > MAX_LOCAL_REPLY_WORDS
        or user_identity_query(normalized)
    ):
        return None
    if natural_capability_query(normalized):
        return (
            "I can search your paper library and the public web, inspect relevant passages, "
            "compare studies, calculate bounded expressions, and produce answers with "
            "traceable citations. I can also ask for clarification when a research request "
            "is ambiguous.",
            SemanticIntent("capabilities", 1.0, "fallback", {}),
        )
    decision = semantic_intent(normalized)
    if decision.intent not in _LOCAL_INTENTS:
        return None
    research_score = decision.scores.get("research", 0.0)
    capability_only = bool(_CAPABILITY_FRAMING.search(normalized)) and not bool(
        _SPECIFIC_RESEARCH_TARGET.search(normalized)
    )
    mixed_research_request = bool(_RESEARCH_TERMS.search(normalized)) and not capability_only
    if mixed_research_request or decision.confidence < research_score + 0.06:
        return None
    answers = {
        "greeting": (
            "Hello! I'm PaperQA. I can search and compare papers in your library, "
            "inspect supporting passages, and write cited research answers. What would "
            "you like to explore?"
        ),
        "gratitude": "You're welcome. What would you like to explore next?",
        "farewell": "Goodbye. Your saved research conversations will be here when you return.",
        "identity": (
            "I'm PaperQA, a scientific research assistant for exploring papers and public sources."
        ),
        "capabilities": (
            "I can search your paper library and the public web, inspect relevant passages, "
            "compare studies, calculate bounded expressions, and produce answers with "
            "traceable citations. I can also ask for clarification when a research request "
            "is ambiguous."
        ),
        "usage": (
            "Add PDF, text, Markdown, or HTML documents from the Library panel, then ask a "
            "question in Auto or Research mode. Use Chat mode when you do not want document "
            "or web retrieval."
        ),
        "wellbeing": "I'm ready to help. What would you like to research?",
    }
    return answers[decision.intent], decision


def explicit_research_request(question: str) -> bool:
    normalized = _normalize(question)
    capability_only = bool(_CAPABILITY_FRAMING.search(normalized)) and not bool(
        _SPECIFIC_RESEARCH_TARGET.search(normalized)
    )
    return bool(_RESEARCH_TERMS.search(normalized)) and not capability_only


def user_identity_query(question: str) -> bool:
    """Return whether the user is asking about their own remembered identity."""
    return bool(_USER_IDENTITY_QUERY.fullmatch(_normalize(question)))


def natural_capability_query(question: str) -> bool:
    """Recognize natural phrasings that ask generally what PaperQA can do."""
    return bool(_NATURAL_CAPABILITY_QUERY.fullmatch(_normalize(question)))


def route_prompt(
    prompt: str,
    mode: PromptMode = "auto",
    previous_route: PromptRoute | None = None,
) -> PromptDecision:
    """Route UI, API, and benchmark prompts through one local decision point."""
    if mode not in {"auto", "research", "chat"}:
        raise ValueError("Mode must be auto, research, or chat.")
    if mode == "chat":
        return PromptDecision("conversation", "Chat mode was selected")

    normalized = " ".join(prompt.split())
    greeting = _GREETING_PREFIX.fullmatch(normalized)
    social_candidates = [normalized, greeting.group("rest").strip()] if greeting else [normalized]
    if (
        any(_SOCIAL_PROMPT.fullmatch(candidate) for candidate in social_candidates)
        or natural_capability_query(normalized)
        or user_identity_query(normalized)
    ):
        return PromptDecision("conversation", "Detected a conversational prompt")
    if mode == "research":
        return PromptDecision("research", "Research mode was selected")

    semantic = semantic_intent(normalized)
    if semantic.intent in _LOCAL_INTENTS and not explicit_research_request(normalized):
        return PromptDecision(
            "conversation",
            f"Local semantic router detected {semantic.intent} ({semantic.confidence:.2f})",
        )
    if explicit_research_request(normalized) or semantic.intent == "research":
        return PromptDecision("research", "Detected a research request")
    if semantic.intent == "casual":
        return PromptDecision("conversation", "Detected a conversational prompt")
    if previous_route == "conversation" and not _RESEARCH_FOLLOWUP.search(normalized):
        return PromptDecision("conversation", "Continuing the conversation")
    return PromptDecision("research", "Detected a research request")
