from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from paperqa.benchmarks.litqa import (
    DownloadClient,
    ExperimentPaths,
    PreparedLibrary,
    _pdf_candidate_links,
    acquire_one,
    align_rows,
    apply_alignment_overrides,
    choice_prompt,
    citation_metrics,
    consolidate_agent_runs,
    normalize_doi,
    normalize_text,
    parse_choice,
    read_jsonl,
    report,
    retrieval_metrics,
    validate_pdf,
    write_json,
    write_jsonl,
)
from paperqa.research.runtime import make_tools
from paperqa.research.safety import Meter


def chunk(text: str, source_id: str, document_id: str = "doc-1") -> Document:
    return Document(
        page_content=text,
        id=source_id,
        metadata={
            "source_id": source_id,
            "document_id": document_id,
            "title": "10.1/example",
            "citation": "10.1/example",
            "filename": "example.pdf",
            "doi": "10.1/example",
            "page": "1",
            "position": 0,
            "start_index": 0,
            "normalized_start": 0,
            "normalized_end": len(normalize_text(text)),
        },
    )


def test_doi_choices_and_pdf_validation() -> None:
    assert normalize_doi("https://DX.doi.org/10.1000/ABC.") == "10.1000/abc"
    validate_pdf(b"%PDF-1.7\nbody\n%%EOF")
    with pytest.raises(ValueError, match="complete PDF"):
        validate_pdf(b"<html>challenge</html>")

    row = {
        "id": "question-1",
        "question": "Which result?",
        "ideal": "correct",
        "distractors": ["wrong", "also wrong"],
    }
    prompt, gold, mapping = choice_prompt(row)
    assert choice_prompt(row) == (prompt, gold, mapping)
    assert mapping[gold] == "correct"
    assert parse_choice("Evidence.\nFINAL_CHOICE: " + gold, set(mapping)) == gold
    assert parse_choice("Choice A", set(mapping)) is None
    assert parse_choice("FINAL_CHOICE: A\nFINAL_CHOICE: B", set(mapping)) is None
    links = _pdf_candidate_links(
        b'<a href="article.pdf">paper</a><a href="MOESM1_ESM.pdf">supplement</a>',
        "https://example.org/page",
    )
    assert links == ["https://example.org/article.pdf"]


def test_jsonl_reader_preserves_unicode_line_separator(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    write_jsonl(target, [{"text": "before\u2028after"}])
    assert read_jsonl(target) == [{"text": "before\u2028after"}]


@pytest.mark.asyncio
async def test_local_pdf_reuse_and_interrupted_download(tmp_path: Path, monkeypatch) -> None:
    paths = ExperimentPaths(tmp_path, "fixture")
    local = tmp_path / "replacement.pdf"
    pdf_data = b"%PDF-1.7\nreplacement\n%%EOF"
    local.write_bytes(pdf_data)
    reused = await acquire_one(
        DownloadClient(host_delay=0),
        "10.1/local",
        paths,
        {"10.1/local": str(local)},
        None,
    )
    assert reused["status"] == "downloaded"
    assert reused["bytes"] == len(pdf_data)

    async def one_candidate(*_args):
        return [("https://example.org/interrupted.pdf", "fixture")]

    class InterruptedClient(DownloadClient):
        async def fetch(self, *_args, **_kwargs):
            raise ConnectionError("interrupted")

    monkeypatch.setattr("paperqa.benchmarks.litqa.resolve_candidates", one_candidate)
    missing = await acquire_one(
        InterruptedClient(),
        "10.1/interrupted",
        paths,
        {},
        None,
    )
    assert missing["status"] == "missing"
    assert not (paths.cache / "10.1_interrupted.download").exists()


def test_alignment_and_retrieval_metrics() -> None:
    text = "The active receptor in-\ncreased its contacts by 2.7 fold in mouse cells."
    page = {
        "page_content": text,
        "metadata": {"doi": "10.1/example", "page": "1"},
    }
    document = chunk(text, "src-" + "a" * 24)
    row = {
        "id": "q1",
        "sources": ["https://doi.org/10.1/example"],
        "key-passage": "The active receptor increased its contacts by 2.7 fold in mouse cells.",
    }
    alignment = align_rows([row], [page], [document])[0]
    assert alignment["status"] == "aligned"
    assert alignment["relevant_ids"] == [document.metadata["source_id"]]

    metrics = retrieval_metrics(["other", document.metadata["source_id"]], set(alignment["relevant_ids"]))
    assert metrics["hit@1"] is False
    assert metrics["hit@3"] is True
    assert metrics["mrr@20"] == 0.5

    overridden = apply_alignment_overrides(
        [{"question_id": "q1", "status": "unaligned", "relevant_ids": []}],
        [row],
        [document],
        {"version": 1, "overrides": {"q1": [document.metadata["source_id"]]}},
    )[0]
    assert overridden["status"] == "aligned_override"


def test_alignment_can_cross_page_boundaries() -> None:
    first_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty"
    )
    second_text = "twentyone twentytwo twentythree twentyfour twentyfive"
    first = chunk(first_text, "src-" + "b" * 24)
    second = chunk(second_text, "src-" + "c" * 24)
    second.metadata.update({"page": "2"})
    pages = [
        {"page_content": first_text, "metadata": {"doi": "10.1/example", "page": "1"}},
        {"page_content": second_text, "metadata": {"doi": "10.1/example", "page": "2"}},
    ]
    row = {
        "id": "q-pages",
        "sources": ["10.1/example"],
        "key-passage": first_text + " " + second_text,
    }
    result = align_rows([row], pages, [first, second])[0]
    assert result["status"] == "aligned"
    assert {item["page"] for item in result["spans"]} == {"1", "2"}
    assert result["gold_span_coverage"] == 1.0


def test_citation_scoring_counts_invalid_and_wrong_source_ids() -> None:
    relevant = "src-" + "1" * 24
    other = "src-" + "2" * 24
    invalid = "src-" + "3" * 24
    metrics = citation_metrics(
        f"Evidence [{relevant}] [{other}] [{invalid}]",
        [
            {"id": relevant, "document_id": "paired"},
            {"id": other, "document_id": "other"},
        ],
        {"paired"},
        {relevant},
    )
    assert metrics == {
        "coverage": True,
        "valid": 2 / 3,
        "associated_source": 1 / 3,
        "gold_passage_proxy": 1 / 3,
    }


@pytest.mark.asyncio
async def test_prepared_library_isolates_documents_and_web_tools(tmp_path: Path) -> None:
    first = chunk("alpha receptor answer", "src-" + "1" * 24, "doc-1")
    second = chunk("beta unrelated passage", "src-" + "2" * 24, "doc-2")
    library = PreparedLibrary([first, second], tmp_path / "vectors", "bm25", {"doc-1"})
    assert library.list() == [{"id": "doc-1", "name": "10.1/example"}]
    assert [item.metadata["document_id"] for item in await library.search("alpha", None, 20)] == [
        "doc-1"
    ]
    observed = []
    meter = Meter({}, lambda: None)
    tools = make_tools(
        library,
        meter,
        lambda *_: None,
        "unused",
        False,
        retrieval_observer=lambda *value: observed.append(value),
    )
    names = {tool.name for tool in tools}
    assert "search_library" in names
    assert "search_web" not in names
    assert "read_web_source" not in names


@pytest.mark.asyncio
async def test_prepared_library_read_keeps_requested_ids_before_neighbors(tmp_path: Path) -> None:
    documents = []
    for position in range(25):
        item = chunk(f"passage {position}", f"src-{position:024x}")
        item.metadata.update({"position": position, "page": "1" if position < 20 else "2"})
        documents.append(item)
    library = PreparedLibrary(documents, tmp_path / "vectors", "bm25")
    requested = [documents[index].metadata["source_id"] for index in range(1, 20, 2)]

    result = await library.read(requested, neighbors=2)

    assert [item.metadata["source_id"] for item in result[: len(requested)]] == requested
    assert set(requested) <= {item.metadata["source_id"] for item in result}
    assert all(item.metadata["page"] == "1" for item in result)


@pytest.mark.asyncio
async def test_all_three_retrievers_share_identical_chunk_ids(tmp_path: Path) -> None:
    documents = [
        chunk("alpha receptor result", "src-" + "d" * 24),
        chunk("unrelated beta control", "src-" + "e" * 24),
    ]

    class FakeVector:
        def as_retriever(self, **_kwargs):
            return RunnableLambda(lambda _query: documents)

    observed = {}
    for retriever in ("bm25", "minilm", "hybrid"):
        library = PreparedLibrary(documents, tmp_path / "vectors", retriever)
        if retriever != "bm25":
            library.vector = FakeVector()
        found = await library.search("alpha receptor", None, 20)
        observed[retriever] = {item.metadata["source_id"] for item in found}
    expected = {item.metadata["source_id"] for item in documents}
    assert observed == {
        "bm25": {documents[0].metadata["source_id"]},
        "minilm": expected,
        "hybrid": expected,
    }


def test_shared_agent_artifacts_are_isolated_from_paired_runs(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path, "fixture")
    write_json(paths.runs / "q1-bm25.json", {"question_id": "q1", "retriever": "bm25"})
    write_json(
        paths.shared_runs / "q1-bm25.json",
        {"question_id": "q1", "retriever": "bm25", "scope": "shared"},
    )

    paired = consolidate_agent_runs(paths, "paired")
    shared = consolidate_agent_runs(paths, "shared")

    assert paired[0]["scope"] == "paired"
    assert shared[0]["scope"] == "shared"
    assert paths.agent_runs.exists()
    assert paths.shared_agent_runs.exists()


def test_report_marks_partial_experiment(tmp_path: Path) -> None:
    paths = ExperimentPaths(tmp_path, "fixture")
    paths.dataset.parent.mkdir(parents=True, exist_ok=True)
    paths.dataset.write_bytes(b"PAR1fixturePAR1")
    write_jsonl(paths.chunks, [chunk("fixture", "src-a").model_dump()])
    write_json(
        paths.corpus,
        {
            "dataset_sha256": hashlib.sha256(paths.dataset.read_bytes()).hexdigest(),
            "chunk_sha256": hashlib.sha256(paths.chunks.read_bytes()).hexdigest(),
            "documents": 1,
            "parse_status": {"10.1/example": {"status": "parsed", "seconds": 0.1}},
        },
    )
    write_jsonl(
        paths.acquisition,
        [{"doi": "10.1/example", "status": "downloaded"}],
    )
    write_jsonl(
        paths.alignments,
        [{"question_id": "q1", "status": "aligned"}],
    )
    write_jsonl(
        paths.retrieval,
        [
            {
                "question_id": "q1",
                "scope": "paired",
                "retriever": "bm25",
                "seconds": 0.01,
                "metrics": retrieval_metrics(["src-a"], {"src-a"}),
            }
        ],
    )
    write_json(
        paths.runs / "q1-bm25.json",
        {
            "question_id": "q1",
            "retriever": "bm25",
            "status": "complete",
            "correct": True,
            "malformed": False,
            "abstained": False,
            "seconds": 1.0,
            "inspected_gold_passage": True,
            "citation": {
                "coverage": True,
                "valid": 1.0,
                "associated_source": 1.0,
                "gold_passage_proxy": 1.0,
            },
        },
    )
    result = report(
        argparse.Namespace(workspace=tmp_path, experiment_id="fixture", cost_ceiling=10.0)
    )
    assert result["complete"] is False
    assert result["answers"]["bm25"]["accuracy"] == 1.0
    assert json.loads(paths.results_json.read_text("utf-8"))["corpus"]["aligned"] == 1
    assert paths.results_csv.exists()
    assert paths.report_html.exists()
