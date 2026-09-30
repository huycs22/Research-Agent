import json
from unittest.mock import AsyncMock

import pytest

from paperqa.research import web_search

ARXIV = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2401.00001</id>
    <title>  Useful Research Paper  </title>
    <summary>A grounded abstract.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <link href="https://arxiv.org/pdf/2401.00001" type="application/pdf" />
  </entry>
</feed>"""


def test_parse_arxiv_returns_readable_primary_url():
    assert web_search.parse_arxiv(ARXIV) == [
        {
            "title": "Useful Research Paper",
            "url": "https://arxiv.org/pdf/2401.00001",
            "snippet": "A grounded abstract.",
            "published": "2024-01-01T00:00:00Z",
            "provider": "arXiv",
        }
    ]


def test_parse_crossref_normalizes_record():
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1000/example",
                    "title": ["Crossref Result"],
                    "URL": "https://doi.org/10.1000/example",
                    "container-title": ["Journal"],
                    "published-online": {"date-parts": [[2025, 2, 3]]},
                }
            ]
        }
    }
    assert web_search.parse_crossref(json.dumps(payload).encode()) == [
        {
            "title": "Crossref Result",
            "url": "https://doi.org/10.1000/example",
            "snippet": "Journal",
            "published": "2025-2-3",
            "provider": "Crossref",
        }
    ]


@pytest.mark.asyncio
async def test_search_survives_one_catalog_failure(monkeypatch):
    result = [
        {
            "title": "Test query working result",
            "url": "https://doi.org/10.1000/working",
            "snippet": "Evidence lead",
            "published": "2025",
            "provider": "Crossref",
        }
    ]

    monkeypatch.setattr(
        web_search, "_arxiv", AsyncMock(side_effect=TimeoutError("catalog unavailable"))
    )
    monkeypatch.setattr(web_search, "_crossref", AsyncMock(return_value=result))
    response = await web_search.search_public_web("test query")
    assert response["results"][0]["title"] == "Test query working result"
    assert response["providers"] == ["Crossref"]
    assert "arXiv: TimeoutError" in response["warnings"][0]
    assert "read_web_source" in response["instruction"]


@pytest.mark.asyncio
async def test_search_reports_both_catalog_failures(monkeypatch):
    failed = AsyncMock(side_effect=TimeoutError("catalog unavailable"))
    monkeypatch.setattr(web_search, "_arxiv", failed)
    monkeypatch.setattr(web_search, "_crossref", failed)
    with pytest.raises(RuntimeError, match=r"arXiv: TimeoutError.*Crossref: TimeoutError"):
        await web_search.search_public_web("test query")


@pytest.mark.asyncio
async def test_search_prefers_matching_arxiv_pdf_over_duplicate_doi(monkeypatch):
    arxiv = [
        {
            "title": "Corrective Retrieval Augmented Generation",
            "url": "https://arxiv.org/pdf/2401.15884",
            "snippet": "Original preprint",
            "published": "2024",
            "provider": "arXiv",
        }
    ]

    crossref = [
        {
            "title": "Corrective Retrieval Augmented Generation",
            "url": "https://doi.org/10.2139/ssrn.5267341",
            "snippet": "Publisher record",
            "published": "2024",
            "provider": "Crossref",
        }
    ]

    monkeypatch.setattr(web_search, "_arxiv", AsyncMock(return_value=arxiv))
    monkeypatch.setattr(web_search, "_crossref", AsyncMock(return_value=crossref))
    response = await web_search.search_public_web(
        "Corrective Retrieval Augmented Generation paper"
    )
    assert response["results"] == arxiv


@pytest.mark.asyncio
async def test_search_drops_catalog_records_unrelated_to_query(monkeypatch):
    results = [
        {
            "title": "AC-aware under-frequency load shedding",
            "url": "https://arxiv.org/pdf/2501.03020",
            "snippet": "Power system frequency control.",
            "published": "2025",
            "provider": "arXiv",
        }
    ]
    monkeypatch.setattr(web_search, "_arxiv", AsyncMock(return_value=results))
    monkeypatch.setattr(web_search, "_crossref", AsyncMock(return_value=[]))

    with pytest.raises(RuntimeError, match="no records matching"):
        await web_search.search_public_web("Can you search the web and tell me about GraphRAG?")


def test_query_terms_remove_conversational_web_request_words():
    assert web_search._query_terms(
        "Can you search the web and tell me about GraphRAG?"
    ) == ["GraphRAG"]
