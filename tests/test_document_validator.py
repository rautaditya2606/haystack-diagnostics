import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from diagnostics.document_validator import validate_document_store


def test_validate_document_store_all_clear():
    store = InMemoryDocumentStore()
    docs = [
        Document(
            content="This is a valid long text document that exceeds the threshold of length fifty characters.",
            meta={"category": "finance", "author": "John Doe"},
            embedding=[0.1, 0.2, 0.3],
        ),
        Document(
            content="Another completely valid document that is sufficiently long and healthy for index.",
            meta={"category": "tech", "author": "Jane Doe"},
            embedding=[0.4, 0.5, 0.6],
        ),
    ]
    store.write_documents(docs)

    report = validate_document_store(
        store,
        expected_metadata_keys=["category", "author"],
        expected_embedding_dim=3,
        short_chunk_threshold=20,
    )

    assert report["summary"]["total_documents"] == 2
    assert report["summary"]["valid_documents"] == 2
    assert report["summary"]["invalid_documents"] == 0
    assert report["summary"]["total_issues_found"] == 0

    assert report["checks"]["content_none"]["status"] == "pass"
    assert report["checks"]["empty_content"]["status"] == "pass"
    assert report["checks"]["short_chunks"]["status"] == "pass"
    assert report["checks"]["duplicate_chunks"]["status"] == "pass"
    assert report["checks"]["missing_metadata"]["status"] == "pass"
    assert report["checks"]["null_embeddings"]["status"] == "pass"
    assert report["checks"]["embedding_dimension_mismatch"]["status"] == "pass"


def test_validate_document_store_problems():
    store = InMemoryDocumentStore()
    docs = [
        # 1. Content is None
        Document(content=None, meta={"category": "a"}),
        # 2. Empty Content
        Document(content="", meta={"category": "b"}),
        # 3. Short chunk
        Document(content="short", meta={"category": "c"}),
        # 4. Duplicate (duplicate of document 3's text "short")
        Document(content="short", meta={"category": "d"}),
        # 5. Missing metadata key ("author" missing)
        Document(
            content="This is a long valid text document but author metadata key is missing.",
            meta={"category": "e"},
        ),
        # 6. Null embedding (default is None)
        Document(
            content="This has no embedding initialized.",
            meta={"category": "f"},
        ),
        # 7. Embedding dimension mismatch (expected 3, got 2)
        Document(
            content="This document has a different embedding dimension than expected.",
            meta={"category": "g", "author": "John"},
            embedding=[0.9, 0.8],
        ),
    ]
    store.write_documents(docs)

    report = validate_document_store(
        store,
        expected_metadata_keys=["category", "author"],
        expected_embedding_dim=3,
        short_chunk_threshold=20,
    )

    assert report["summary"]["total_documents"] == 7
    # All of these documents trigger at least one invalid/warning flag
    assert report["summary"]["invalid_documents"] > 0

    # Assert specific checks
    assert report["checks"]["content_none"]["count"] == 1
    assert report["checks"]["empty_content"]["count"] == 1
    assert report["checks"]["short_chunks"]["count"] == 2  # both "short" and "short"
    assert report["checks"]["duplicate_chunks"]["count"] == 1  # 1 group of duplicates
    assert len(report["checks"]["duplicate_chunks"]["duplicates"][0]["document_ids"]) == 2
    assert report["checks"]["missing_metadata"]["count"] == 6  # all except the ones with author (none of them have it)
    assert report["checks"]["null_embeddings"]["count"] == 6  # only the last has embedding
    assert report["checks"]["embedding_dimension_mismatch"]["count"] == 1


def test_validate_document_store_empty():
    store = InMemoryDocumentStore()
    report = validate_document_store(
        store,
        expected_metadata_keys=["category"],
        expected_embedding_dim=3,
        short_chunk_threshold=20
    )
    assert report["summary"]["total_documents"] == 0
    assert report["summary"]["valid_documents"] == 0
    assert report["summary"]["invalid_documents"] == 0
    assert report["summary"]["total_issues_found"] == 0
    assert report["checks"]["content_none"]["count"] == 0
    assert report["checks"]["empty_content"]["count"] == 0


def test_validate_document_store_one_document():
    store = InMemoryDocumentStore()
    store.write_documents([
        Document(
            content="This is a valid long text document that exceeds the threshold of length fifty characters.",
            meta={"category": "finance", "author": "John Doe"},
            embedding=[0.1, 0.2, 0.3],
        )
    ])
    report = validate_document_store(
        store,
        expected_metadata_keys=["category", "author"],
        expected_embedding_dim=3,
        short_chunk_threshold=20
    )
    assert report["summary"]["total_documents"] == 1
    assert report["summary"]["valid_documents"] == 1
    assert report["summary"]["invalid_documents"] == 0
    assert report["summary"]["total_issues_found"] == 0
