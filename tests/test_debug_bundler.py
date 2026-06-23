"""
tests/test_debug_bundler.py

Tests for collect_debug_bundle() and diff_debug_bundles().
All tests run offline with no external API dependencies.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest
from haystack import Document, Pipeline
from haystack.core.component import component

from diagnostics.debug_bundler import collect_debug_bundle, diff_debug_bundles


# ---------------------------------------------------------------------------
# Minimal mock components
# ---------------------------------------------------------------------------

@component
class MockRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else []

    @component.output_types(documents=list)
    def run(self, query: str):
        return {"documents": self.docs}


@component
class MockRanker:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else []

    @component.output_types(documents=list)
    def run(self, documents: list, query: str):
        return {"documents": self.docs}


@component
class MockGenerator:
    def __init__(self, replies=None):
        self.replies = replies if replies is not None else []

    @component.output_types(replies=list)
    def run(self, prompt: list):
        return {"replies": self.replies}


def _simple_pipeline(docs=None, answer="Test answer."):
    """Builds a minimal retriever → generator pipeline."""
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=docs or [Document(content="Hello world", score=0.9)]))
    pipe.add_component("generator", MockGenerator(replies=[answer]))
    pipe.connect("retriever.documents", "generator.prompt")
    return pipe


def _reranker_pipeline(raw_docs=None, reranked_docs=None, answer="Test answer."):
    """Builds a retriever → ranker → generator pipeline."""
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=raw_docs or []))
    pipe.add_component("ranker", MockRanker(docs=reranked_docs or []))
    pipe.add_component("generator", MockGenerator(replies=[answer]))
    pipe.connect("retriever.documents", "ranker.documents")
    pipe.connect("ranker.documents", "generator.prompt")
    return pipe


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_bundle_schema_keys(tmp_path):
    """Bundle must contain all required top-level keys."""
    pipe = _simple_pipeline()
    bundle = collect_debug_bundle(
        query="What is hello world?",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "What is hello world?"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    required_keys = {"bundle_id", "created_at", "query", "pipeline", "retrieval", "reranker", "generation", "failure_type", "diagnostics", "corpus_checks"}
    assert required_keys.issubset(bundle.keys()), f"Missing keys: {required_keys - bundle.keys()}"


def test_bundle_pipeline_section_has_version(tmp_path):
    """Pipeline section must include haystack_version, components, connections, mermaid."""
    pipe = _simple_pipeline()
    bundle = collect_debug_bundle(
        query="version test",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "version test"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    pl = bundle["pipeline"]
    assert "haystack_version" in pl
    assert "components" in pl
    assert "connections" in pl
    assert "mermaid" in pl


def test_bundle_retrieval_section(tmp_path):
    """Retrieval section must list raw_top_k with expected fields."""
    doc = Document(content="Paris is the capital of France.", score=0.85, id="doc-1")
    pipe = _simple_pipeline(docs=[doc])
    bundle = collect_debug_bundle(
        query="capital of france",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "capital of france"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    assert bundle["retrieval"]["doc_count"] == 1
    top = bundle["retrieval"]["raw_top_k"][0]
    assert top["id"] == "doc-1"
    assert top["score"] == pytest.approx(0.85)
    assert "content_preview" in top
    assert "meta" in top


# ---------------------------------------------------------------------------
# Filename tests
# ---------------------------------------------------------------------------

def test_bundle_persisted_to_disk(tmp_path):
    """Bundle JSON must be written to output_dir."""
    pipe = _simple_pipeline()
    bundle = collect_debug_bundle(
        query="disk persistence check",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "disk persistence check"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    bundle_path = Path(bundle["_bundle_path"])
    assert bundle_path.exists(), "Bundle file was not written to disk."
    with open(bundle_path) as f:
        loaded = json.load(f)
    assert loaded["bundle_id"] == bundle["bundle_id"]


def test_bundle_filename_is_slug_not_uuid(tmp_path):
    """Filename must follow {query_slug}_{timestamp}.json — NOT a raw UUID."""
    pipe = _simple_pipeline()
    bundle = collect_debug_bundle(
        query="Is this a slug filename?",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "Is this a slug filename?"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    filename = Path(bundle["_bundle_path"]).name
    # Must not look like a UUID (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.json$")
    assert not uuid_pattern.match(filename), f"Filename looks like a UUID: {filename}"
    # Must start with a slug fragment from the query
    assert filename.startswith("is_this_a_slug"), f"Filename slug mismatch: {filename}"
    # bundle_id UUID must be stored INSIDE the JSON
    assert "bundle_id" in bundle
    assert re.match(r"^[0-9a-f\-]{36}$", bundle["bundle_id"])


# ---------------------------------------------------------------------------
# Corpus check tests
# ---------------------------------------------------------------------------

def test_bundle_scoped_corpus_check_no_store(tmp_path):
    """Without a document store, corpus checks must be skipped gracefully."""
    pipe = _simple_pipeline()
    bundle = collect_debug_bundle(
        query="no store test",
        pipeline=pipe,
        document_store=None,
        pipeline_inputs={"retriever": {"query": "no store test"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    assert bundle["corpus_checks"]["skipped"] is True


def test_bundle_scoped_corpus_check_with_store(tmp_path):
    """With a document store, corpus checks must be scoped to retrieved doc IDs only."""
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    doc_a = Document(content="Relevant document.", id="doc-a")
    doc_b = Document(content="Unrelated document.", id="doc-b")

    store = InMemoryDocumentStore()
    store.write_documents([doc_a, doc_b])

    pipe = _simple_pipeline(docs=[doc_a])
    bundle = collect_debug_bundle(
        query="scoped check",
        pipeline=pipe,
        document_store=store,
        pipeline_inputs={"retriever": {"query": "scoped check"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    cc = bundle["corpus_checks"]
    assert cc["skipped"] is False
    # Only doc-a was retrieved, so only doc-a should be in the scoped check
    assert cc["scoped_to_doc_ids"] == ["doc-a"]
    assert "doc-b" not in cc["scoped_to_doc_ids"]


# ---------------------------------------------------------------------------
# Failure classification via bundle
# ---------------------------------------------------------------------------

def test_bundle_failure_type_success(tmp_path):
    pipe = _simple_pipeline(docs=[Document(content="Valid content.", score=0.9)])
    bundle = collect_debug_bundle(
        query="valid query",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "valid query"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    assert bundle["failure_type"] == "SUCCESS"


def test_bundle_failure_type_no_results(tmp_path):
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[]))
    pipe.add_component("generator", MockGenerator(replies=[]))
    pipe.connect("retriever.documents", "generator.prompt")
    bundle = collect_debug_bundle(
        query="empty results query",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "empty results query"}, "generator": {"prompt": []}},
        output_dir=str(tmp_path),
    )
    assert bundle["failure_type"] == "NO_RESULTS"



def test_score_below_cutoff_detection(tmp_path):
    """SCORE_BELOW_CUTOFF fires when relevant_doc_id is not in retrieved results."""
    pipe = _simple_pipeline(docs=[Document(content="Some context.", score=0.3, id="doc-x")])
    bundle = collect_debug_bundle(
        query="score cutoff query",
        pipeline=pipe,
        pipeline_inputs={"retriever": {"query": "score cutoff query"}, "generator": {"prompt": []}},
        ranking_threshold=0.8,
        relevant_doc_id="doc-expected",  # not in retrieved docs
        output_dir=str(tmp_path),
    )
    assert bundle["failure_type"] == "SCORE_BELOW_CUTOFF"


def test_context_loss_detection(tmp_path):
    """CONTEXT_LOSS fires when relevant_doc_id is in raw_top_k but dropped by the reranker."""
    expected_doc = Document(content="The expected document.", score=0.9, id="doc-expected")
    other_doc = Document(content="A different document.", score=0.7, id="doc-other")

    # raw retriever returns both; ranker only keeps doc-other (drops doc-expected)
    pipe = _reranker_pipeline(
        raw_docs=[expected_doc, other_doc],
        reranked_docs=[other_doc],
        answer="Partial answer.",
    )
    bundle = collect_debug_bundle(
        query="context loss query",
        pipeline=pipe,
        pipeline_inputs={
            "retriever": {"query": "context loss query"},
            "ranker": {"query": "context loss query", "documents": []},
            "generator": {"prompt": []},
        },
        relevant_doc_id="doc-expected",
        output_dir=str(tmp_path),
    )
    assert bundle["failure_type"] == "CONTEXT_LOSS"


# ---------------------------------------------------------------------------
# Diff tests
# ---------------------------------------------------------------------------

def _write_bundle(tmp_path, name, raw_top_k, answer, failure_type="SUCCESS", config=None):
    """Helper to write a minimal bundle JSON for diff tests."""
    bundle = {
        "bundle_id": "test-id",
        "created_at": "2026-01-01T00:00:00+00:00",
        "query": "test query",
        "pipeline": {
            "haystack_version": "2.0.0",
            "components": config or {},
            "connections": [],
            "mermaid": "",
            "metadata": {},
        },
        "retrieval": {"retriever_component": "retriever", "doc_count": len(raw_top_k), "raw_top_k": raw_top_k},
        "reranker": {"detected": False, "component": None, "reranked_top_k": None, "note": None},
        "generation": {"component": "generator", "prompt_snapshot": None, "answer": answer},
        "failure_type": failure_type,
        "diagnostics": {},
        "corpus_checks": {"skipped": True},
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(bundle, indent=2))
    return str(path)


def test_diff_score_delta(tmp_path):
    """Diff must detect when a document's score changed between two runs."""
    bundle_a = _write_bundle(tmp_path, "a", [{"id": "doc-1", "score": 0.9, "rank": 1, "content_preview": "", "meta": {}}], "Answer A")
    bundle_b = _write_bundle(tmp_path, "b", [{"id": "doc-1", "score": 0.6, "rank": 1, "content_preview": "", "meta": {}}], "Answer B")
    diff = diff_debug_bundles(bundle_a, bundle_b)
    assert "doc-1" in diff["score_deltas"]
    assert diff["score_deltas"]["doc-1"]["delta"] == pytest.approx(-0.3, abs=1e-5)


def test_diff_appearance_disappearance(tmp_path):
    """Diff must detect documents that appeared or disappeared between runs."""
    bundle_a = _write_bundle(tmp_path, "a2", [{"id": "doc-old", "score": 0.8, "rank": 1, "content_preview": "", "meta": {}}], "Old answer")
    bundle_b = _write_bundle(tmp_path, "b2", [{"id": "doc-new", "score": 0.9, "rank": 1, "content_preview": "", "meta": {}}], "New answer")
    diff = diff_debug_bundles(bundle_a, bundle_b)
    appeared_ids = [d["id"] for d in diff["docs_appeared"]]
    disappeared_ids = [d["id"] for d in diff["docs_disappeared"]]
    assert "doc-new" in appeared_ids
    assert "doc-old" in disappeared_ids


def test_diff_failure_type_change(tmp_path):
    """Diff must detect a failure type change between two runs."""
    bundle_a = _write_bundle(tmp_path, "a3", [], "I do not know.", failure_type="GENERATOR_FAILURE")
    bundle_b = _write_bundle(tmp_path, "b3", [{"id": "doc-1", "score": 0.9, "rank": 1, "content_preview": "", "meta": {}}], "The answer is London.", failure_type="SUCCESS")
    diff = diff_debug_bundles(bundle_a, bundle_b)
    assert diff["failure_type_change"]["changed"] is True
    assert diff["failure_type_change"]["before"] == "GENERATOR_FAILURE"
    assert diff["failure_type_change"]["after"] == "SUCCESS"


def test_diff_config_changes(tmp_path):
    """Diff must detect component init_parameter changes."""
    config_a = {"retriever": {"type": "InMemoryBM25Retriever", "init_parameters": {"top_k": 5}}}
    config_b = {"retriever": {"type": "InMemoryBM25Retriever", "init_parameters": {"top_k": 10}}}
    bundle_a = _write_bundle(tmp_path, "a4", [], "Answer.", config=config_a)
    bundle_b = _write_bundle(tmp_path, "b4", [], "Answer.", config=config_b)
    diff = diff_debug_bundles(bundle_a, bundle_b)
    assert "retriever" in diff["config_changes"]
    assert diff["config_changes"]["retriever"]["before"]["top_k"] == 5
    assert diff["config_changes"]["retriever"]["after"]["top_k"] == 10


def test_diff_answer_no_change(tmp_path):
    """When answers are identical, diff must report no change."""
    bundle_a = _write_bundle(tmp_path, "a5", [], "Identical answer.")
    bundle_b = _write_bundle(tmp_path, "b5", [], "Identical answer.")
    diff = diff_debug_bundles(bundle_a, bundle_b)
    assert diff["answer_diff"] == "(no change)"
