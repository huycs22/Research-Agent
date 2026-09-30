"""Key-free public web discovery for the local research agent.

The agent primarily researches scientific literature, so use stable public
catalog APIs instead of scraping consumer search-result pages.  Results are
discovery leads only; ``read_web_source`` performs the separate, SSRF-safe read
that makes a source citable.
"""

from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from .safety import fetch_source, public_url

ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}


def _clean(value: object, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _query_terms(query: str) -> list[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "about",
        "can",
        "could",
        "find",
        "give",
        "information",
        "internet",
        "me",
        "please",
        "search",
        "tell",
        "web",
        "you",
        "paper",
        "papers",
        "article",
        "articles",
        "study",
        "studies",
        "source",
        "sources",
    }
    terms = [
        term for term in re.findall(r"[\w.-]+", query, re.UNICODE) if term.lower() not in stop
    ]
    return terms[:12] or [query.strip()]


def parse_arxiv(data: bytes) -> list[dict[str, str]]:
    # The response is bounded to 2 MB before parsing and ElementTree does not
    # resolve external entities. arXiv's Atom feed is the only accepted input.
    root = ET.fromstring(data)
    results = []
    for entry in root.findall("atom:entry", _ATOM):
        links = {
            link.attrib.get("type", ""): link.attrib.get("href", "")
            for link in entry.findall("atom:link", _ATOM)
        }
        url = links.get("application/pdf") or links.get("text/html")
        if not url:
            url = entry.findtext("atom:id", default="", namespaces=_ATOM)
        try:
            url = public_url(url)
        except ValueError:
            continue
        results.append({
            "title": _clean(entry.findtext("atom:title", default="", namespaces=_ATOM), 300),
            "url": url,
            "snippet": _clean(entry.findtext("atom:summary", default="", namespaces=_ATOM)),
            "published": _clean(
                entry.findtext("atom:published", default="", namespaces=_ATOM), 40
            ),
            "provider": "arXiv",
        })
    return results


def parse_crossref(data: bytes) -> list[dict[str, str]]:
    payload = json.loads(data)
    results = []
    for item in payload.get("message", {}).get("items", []):
        titles = item.get("title") or []
        containers = item.get("container-title") or []
        url = item.get("URL") or ("https://doi.org/" + item["DOI"] if item.get("DOI") else "")
        try:
            url = public_url(url)
        except ValueError:
            continue
        date = (
            item.get("published-print")
            or item.get("published-online")
            or item.get("issued")
            or {}
        )
        parts = date.get("date-parts") or []
        results.append({
            "title": _clean(titles[0] if titles else item.get("DOI"), 300),
            "url": url,
            "snippet": _clean(item.get("abstract") or (containers[0] if containers else "")),
            "published": "-".join(str(part) for part in (parts[0] if parts else [])),
            "provider": "Crossref",
        })
    return results


async def _arxiv(query: str, limit: int) -> list[dict[str, str]]:
    terms = _query_terms(query)
    expression = " AND ".join(f"all:{term}" for term in terms)
    url = (
        ARXIV_API
        + "?"
        + urlencode({
            "search_query": expression,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
    )
    data, _, _ = await fetch_source(url, limit=2_000_000)
    return parse_arxiv(data)


async def _crossref(query: str, limit: int) -> list[dict[str, str]]:
    url = (
        CROSSREF_API
        + "?"
        + urlencode({
            "query.bibliographic": query,
            "rows": limit,
            "select": "DOI,title,URL,published-print,published-online,issued,abstract,container-title",
        })
    )
    data, _, _ = await fetch_source(url, limit=2_000_000)
    return parse_crossref(data)


async def search_public_web(query: str, limit: int = 8) -> dict:
    """Search public research catalogs concurrently and return usable URLs.

    One catalog being unavailable must not erase useful results from the other.
    If both fail, surface actionable, provider-specific diagnostics to the agent.
    """
    per_provider = max(3, min(8, limit))
    responses = await asyncio.gather(
        _arxiv(query, per_provider),
        _crossref(query, per_provider),
        return_exceptions=True,
    )
    groups: list[list[dict[str, str]]] = []
    errors = []
    for provider, response in zip(("arXiv", "Crossref"), responses, strict=True):
        if isinstance(response, BaseException):
            errors.append(f"{provider}: {type(response).__name__}: {_clean(response, 250)}")
            continue
        groups.append(response)
    # Prefer a directly readable arXiv PDF when catalogs describe the same work.
    # Crossref DOI targets frequently resolve to publisher pages that reject
    # automated readers, while the arXiv record is still the original paper.
    unique: dict[str, dict[str, str]] = {}
    for group in groups:
        for item in group:
            key = re.sub(r"\W+", "", item["title"].lower()) or item["url"]
            existing = unique.get(key)
            if existing is None or (
                item["provider"] == "arXiv" and existing["provider"] != "arXiv"
            ):
                unique[key] = item

    query_terms = {term.lower() for term in _query_terms(query)}

    def overlap_count(item: dict[str, str]) -> int:
        searchable = " ".join((item.get("title", ""), item.get("snippet", "")))
        item_terms = {
            term.lower() for term in re.findall(r"[\w.-]+", searchable, re.UNICODE)
        }
        return len(query_terms & item_terms)

    # Catalog relevance ordering is advisory and occasionally returns records
    # with no lexical relationship to the request. Such records must not be
    # offered to the model as if they were valid discovery leads.
    relevant = [item for item in unique.values() if overlap_count(item)]
    if not relevant:
        detail = "; ".join(errors)
        raise RuntimeError(
            "Public search found no records matching the query."
            + (f" Provider warnings: {detail}" if detail else "")
        )

    def relevance(item: dict[str, str]):
        title_terms = {term.lower() for term in _query_terms(item["title"])}
        overlap = len(query_terms & title_terms) / max(1, len(query_terms | title_terms))
        return (
            -overlap_count(item),
            -overlap,
            item["provider"] != "arXiv",
            item["title"].lower(),
        )

    results = sorted(relevant, key=relevance)
    return {
        "results": results[:limit],
        "providers": sorted({item["provider"] for item in results[:limit]}),
        "warnings": errors,
        "instruction": (
            "These catalog records are discovery leads, not citable evidence. "
            "Use read_web_source on a result URL before citing it."
        ),
    }
