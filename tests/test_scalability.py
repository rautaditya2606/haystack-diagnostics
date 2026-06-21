import gc
import tracemalloc
import pytest
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from diagnostics.document_validator import validate_document_store


def test_scalability_performance():
    # Initialize store
    store = InMemoryDocumentStore()
    
    # Generate 100,000 simple synthetic documents
    docs = []
    for i in range(100000):
        docs.append(Document(
            id=f"doc_{i}",
            content=f"This is synthetic document number {i} which is sufficiently long to pass the threshold.",
            meta={"source": "perf_test"}
        ))
    
    store.write_documents(docs)
    del docs
    gc.collect()
    
    # Start tracing memory
    tracemalloc.start()
    
    report = validate_document_store(
        document_store=store,
        expected_metadata_keys=["source"],
        short_chunk_threshold=20,
        batch_size=1000,
        validate_embeddings=False  # Do not validate/load embeddings
    )
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mb = peak / (1024 * 1024)
    print(f"\n[Scalability Test] Peak memory allocated during validation: {peak_mb:.2f} MB")
    
    # Assertions
    assert report["summary"]["total_documents"] == 100000
    assert report["summary"]["invalid_documents"] == 0
    
    # Ensure memory footprint remains bounded (under 30MB)
    assert peak_mb < 30.0, f"Peak memory {peak_mb:.2f} MB exceeded threshold of 30 MB"
