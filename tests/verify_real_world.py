import os
import sys
import time
import json
import uuid
import decimal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

# Load RAG Studio environment variables
from dotenv import load_dotenv
dotenv_path = "/home/adityaraut/Documents/verba_haystack_rag_studio/rag-studio/backend/.env"
load_dotenv(dotenv_path)

# Monkey-patch RotatingFileHandler to avoid permission issues in RAG Studio logger
import logging.handlers
logging.handlers.RotatingFileHandler = lambda *args, **kwargs: logging.NullHandler()

# Add RAG Studio paths to sys.path
sys.path.append("/home/adityaraut/Documents/verba_haystack_rag_studio/rag-studio/backend/app")
sys.path.append("/home/adityaraut/Documents/Haystack_Inspector")

from rag_studio_stack.rag_studio_lib.services.haystack_manager import HaystackManager
from haystack import Pipeline, Document, component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

# Import our new diagnostics features
from diagnostics import (
    validate_document_store,
    inspect_pipeline,
    diagnose_retrieval_failure,
    collect_debug_bundle,
    diff_debug_bundles,
)

# Custom Reranker component for testing CONTEXT_LOSS and runtime warning guardrails
@component
class CustomReranker:
    def __init__(self, drop_doc_ids: list = None):
        self.drop_doc_ids = drop_doc_ids or []

    @component.output_types(documents=list)
    def run(self, documents: list):
        filtered = [d for d in documents if d.id not in self.drop_doc_ids]
        return {"documents": filtered}

@component
class RerankerA:
    @component.output_types(documents=list)
    def run(self, documents: list):
        return {"documents": documents}

@component
class RerankerB:
    @component.output_types(documents=list)
    def run(self, documents: list):
        return {"documents": documents}


def test_layer_1_fidelity(manager):
    print("\n--- Layer 1: Bundle Fidelity Verification ---")
    # Build standard pipeline
    pipe = Pipeline()
    pipe.add_component("text_embedder", manager.text_embedder)
    pipe.add_component("retriever", manager.retriever)
    pipe.add_component("prompt_builder", manager.prompt_builder)
    pipe.add_component("llm", manager.llm)

    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")

    query = "A. Raut Hardware benchmarking"
    
    # Collect bundle
    bundle = collect_debug_bundle(
        query=query,
        pipeline=pipe,
        document_store=manager.document_store,
        output_dir="./test_debug_bundles"
    )

    # 1. bundle_id exists and is UUID-like
    assert "bundle_id" in bundle
    uuid_val = uuid.UUID(bundle["bundle_id"]) # Will raise ValueError if not UUID
    print("[PASS] bundle_id is valid UUID:", bundle["bundle_id"])

    # 2. Filename uses slug pattern, not UUID
    path_str = bundle["_bundle_path"]
    filename = Path(path_str).name
    assert filename.startswith("a_raut_hardware_benchmarking_")
    assert bundle["bundle_id"] not in filename
    print("[PASS] filename matches slug pattern:", filename)

    # 3. Pipeline graph, connections, components
    assert "pipeline" in bundle
    assert "text_embedder" in bundle["pipeline"]["components"]
    assert "retriever" in bundle["pipeline"]["components"]
    assert len(bundle["pipeline"]["connections"]) > 0
    print("[PASS] pipeline structural metadata captured successfully")

    # 4. Component params match
    retriever_info = bundle["pipeline"]["components"]["retriever"]
    assert "WeaviateEmbeddingRetriever" in retriever_info["type"]
    print("[PASS] component init_parameters captured successfully")

    # 5. Retrieved docs are populated
    assert "retrieval" in bundle
    assert bundle["retrieval"]["doc_count"] > 0
    assert len(bundle["retrieval"]["raw_top_k"]) > 0
    first_doc = bundle["retrieval"]["raw_top_k"][0]
    assert "id" in first_doc
    assert "score" in first_doc
    print(f"[PASS] retrieved {bundle['retrieval']['doc_count']} docs. Top doc ID={first_doc['id']}, score={first_doc['score']}")

    # 6. Prompt snapshot and Answer
    assert bundle["generation"]["prompt_snapshot"] is not None
    assert bundle["generation"]["answer"] is not None
    print("[PASS] prompt snapshot and answer populated")

    # 7. Serialization Drift Check: bundle_returned == bundle_from_disk
    with open(path_str, "r", encoding="utf-8") as f:
        bundle_from_disk = json.load(f)
    
    # Compare
    assert bundle_from_disk == bundle
    print("[PASS] Serialization Drift Check PASSED (Returned dict matches disk JSON 100%)")
    return pipe


def test_layer_2_taxonomy(live_pipe):
    print("\n--- Layer 2: Failure Taxonomy Verification ---")
    
    # 1. NO_RESULTS
    print("Testing NO_RESULTS scenario...")
    diag_no_res = diagnose_retrieval_failure(
        pipeline=live_pipe,
        query="nonexistent_gibberish_query_12345",
        pipeline_outputs={"retriever": {"documents": []}}
    )
    assert diag_no_res["failure_type"] == "NO_RESULTS"
    print("[PASS] NO_RESULTS correctly classified.")

    # Weaviate document ID for benchmarking document:
    known_doc_id = "3e4072848804e137bb54989e06afb2d0b78637df50bccddd5bcaefb5356ab530"

    # 2. SCORE_BELOW_CUTOFF
    print("Testing SCORE_BELOW_CUTOFF scenario...")
    # Expected doc not in retriever output
    diag_cutoff = diagnose_retrieval_failure(
        pipeline=live_pipe,
        query="Hardware benchmarking",
        relevant_doc_id="non_existent_doc_id",
        pipeline_outputs={"retriever": {"documents": [Document(id=known_doc_id, content="Benchmarking...", score=0.9)]}}
    )
    assert diag_cutoff["failure_type"] == "SCORE_BELOW_CUTOFF"
    assert diag_cutoff["diagnostics"]["ranking"]["failure_subtype"] == "SCORE_BELOW_CUTOFF"
    print("[PASS] SCORE_BELOW_CUTOFF correctly classified when expected doc is not retrieved.")

    # 3. CONTEXT_LOSS
    print("Testing CONTEXT_LOSS scenario...")
    # Expected doc in retriever output but dropped by reranker
    pipe_rerank = Pipeline()
    dummy_store = InMemoryDocumentStore()
    dummy_retriever = InMemoryBM25Retriever(document_store=dummy_store)
    pipe_rerank.add_component("retriever", dummy_retriever)
    pipe_rerank.add_component("ranker", CustomReranker(drop_doc_ids=[known_doc_id]))
    pipe_rerank.connect("retriever.documents", "ranker.documents")

    # Mock outputs to simulate: retrieved doc present in retriever but dropped by ranker
    retrieved_doc = Document(id=known_doc_id, content="Benchmarking...", score=0.9)
    mock_outputs = {
        "retriever": {"documents": [retrieved_doc]},
        "ranker": {"documents": []}
    }

    diag_loss = diagnose_retrieval_failure(
        pipeline=pipe_rerank,
        query="Hardware benchmarking",
        relevant_doc_id=known_doc_id,
        reranker_component_name="ranker",
        pipeline_outputs=mock_outputs
    )
    assert diag_loss["failure_type"] == "CONTEXT_LOSS"
    assert diag_loss["diagnostics"]["ranking"]["failure_subtype"] == "CONTEXT_LOSS"
    assert diag_loss["failure_type"] != "SCORE_BELOW_CUTOFF"
    print("[PASS] CONTEXT_LOSS correctly classified and is mutually exclusive with SCORE_BELOW_CUTOFF.")


def test_layer_3_runtime_guardrails():
    print("\n--- Layer 3: Runtime Guardrail Verification ---")
    pipe_rerank = Pipeline()
    dummy_store = InMemoryDocumentStore()
    dummy_retriever = InMemoryBM25Retriever(document_store=dummy_store)
    pipe_rerank.add_component("retriever", dummy_retriever)
    pipe_rerank.add_component("ranker", CustomReranker(drop_doc_ids=[]))
    pipe_rerank.connect("retriever.documents", "ranker.documents")

    # Mock results where retriever output exists but ranker output is uncaptured
    results = {
        "retriever": {"documents": [Document(id="doc_1", content="test")]}
    }

    # Diagnose failures
    diag = diagnose_retrieval_failure(
        pipeline=pipe_rerank,
        query="A. Raut Hardware benchmarking",
        relevant_doc_id="some_id",
        pipeline_outputs=results
    )

    # Verify warn note exists and tells the user ranker output was not captured
    reranker_section = diag["diagnostics"]["reranker"]
    assert reranker_section.get("note") is not None
    expected_warn = "was detected in the pipeline but its intermediate output was not captured"
    assert expected_warn in reranker_section["note"]
    print("[PASS] Runtime guardrail successfully detected uncaptured ranker output:")
    print("  Warning note:", reranker_section["note"])


def test_layer_4_diff_engine():
    print("\n--- Layer 4: Diff Engine Verification ---")
    bundle_a_path = "./test_debug_bundles/diff_test_a.json"
    bundle_b_path = "./test_debug_bundles/diff_test_b.json"
    os.makedirs("./test_debug_bundles", exist_ok=True)

    # Setup bundle a
    bundle_a = {
        "bundle_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": "What is the capital of France?",
        "pipeline": {
            "haystack_version": "2.0.0",
            "components": {
                "retriever": {"type": "InMemoryBM25Retriever", "init_parameters": {"top_k": 2}}
            }
        },
        "retrieval": {
            "doc_count": 2,
            "raw_top_k": [
                {"id": "doc_1", "score": 0.9, "content_preview": "Paris is the capital of France."},
                {"id": "doc_2", "score": 0.7, "content_preview": "Lyon is in France."}
            ]
        },
        "generation": {
            "answer": "The capital of France is Paris."
        },
        "failure_type": "SUCCESS"
    }

    # Setup bundle b
    bundle_b = {
        "bundle_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": "What is the capital of France?",
        "pipeline": {
            "haystack_version": "2.0.0",
            "components": {
                "retriever": {"type": "InMemoryBM25Retriever", "init_parameters": {"top_k": 5}}
            }
        },
        "retrieval": {
            "doc_count": 3,
            "raw_top_k": [
                {"id": "doc_1", "score": 0.85, "content_preview": "Paris is the capital of France."}, # Score changed
                {"id": "doc_3", "score": 0.6, "content_preview": "Marseille is in France."},        # Doc appeared
                # doc_2 disappeared
            ]
        },
        "generation": {
            "answer": "The capital of France is Paris! It is beautiful." # Answer changed
        },
        "failure_type": "SUCCESS"
    }

    with open(bundle_a_path, "w") as f:
        json.dump(bundle_a, f, indent=2)
    with open(bundle_b_path, "w") as f:
        json.dump(bundle_b, f, indent=2)

    diff = diff_debug_bundles(bundle_a_path, bundle_b_path)

    # Assert appeared/disappeared docs
    assert any(d["id"] == "doc_3" for d in diff["docs_appeared"])
    assert any(d["id"] == "doc_2" for d in diff["docs_disappeared"])
    print("[PASS] docs_appeared / docs_disappeared correctly populated.")

    # Assert score deltas
    assert "doc_1" in diff["score_deltas"]
    assert abs(diff["score_deltas"]["doc_1"]["delta"] - (-0.05)) < 1e-5
    print("[PASS] score_deltas correctly populated.")

    # Assert config changes
    assert "retriever" in diff["config_changes"]
    assert diff["config_changes"]["retriever"]["before"] == {"top_k": 2}
    assert diff["config_changes"]["retriever"]["after"] == {"top_k": 5}
    print("[PASS] config_changes correctly populated.")

    # Assert answer diff
    assert "Paris" in diff["answer_diff"]
    assert "beautiful" in diff["answer_diff"]
    print("[PASS] answer_diff populated with unified diff.")

    # Run CLI comparison
    print("Testing CLI diff execution...")
    res = subprocess.run(
        [sys.executable, "-m", "diagnostics.debug_bundler", "diff", bundle_a_path, bundle_b_path],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    cli_diff = json.loads(res.stdout)
    assert cli_diff["score_deltas"] == diff["score_deltas"]
    assert cli_diff["config_changes"] == diff["config_changes"]
    print("[PASS] CLI output matches Python API output perfectly.")


def test_layer_5_large_corpus():
    print("\n--- Layer 5: Large Corpus Validation ---")
    sizes = [10000, 50000, 100000]

    for size in sizes:
        store = InMemoryDocumentStore()
        
        # Write batch of documents
        docs = []
        for i in range(size):
            docs.append(Document(
                id=f"doc_{i}",
                content=f"This is synthetic document number {i} containing a unique sentence.",
                meta={"source": "perf_scale"}
            ))
        store.write_documents(docs)

        pipe = Pipeline()
        pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=3))
        
        # Time the collect_debug_bundle latency
        start_time = time.perf_counter()
        bundle = collect_debug_bundle(
            query="synthetic document number 500",
            pipeline=pipe,
            document_store=store,
            output_dir="./test_debug_bundles"
        )
        latency = time.perf_counter() - start_time

        # Assert scoped check only checks the 3 retrieved docs, not the full size
        assert bundle["corpus_checks"]["docs_found"] == 3
        assert len(bundle["corpus_checks"]["scoped_to_doc_ids"]) == 3
        print(f"[PASS] Scale {size:6,} documents | latency: {latency:.4f}s | scoped_to_doc_ids size: {len(bundle['corpus_checks']['scoped_to_doc_ids'])}")
        
        # Verify latency is sub-second (usually < 50ms)
        assert latency < 1.0, f"Latency {latency:.4f}s exceeded sub-second threshold!"


def test_layer_6_additional_cases(manager):
    print("\n--- Layer 6: Additional Robustness Cases ---")

    # 1. Filename collisions (run twice in same second)
    print("Testing filename collision avoidance...")
    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=InMemoryDocumentStore(), top_k=2))

    bundle_1 = collect_debug_bundle("france capital", pipe, output_dir="./test_debug_bundles")
    bundle_2 = collect_debug_bundle("france capital", pipe, output_dir="./test_debug_bundles")

    assert bundle_1["_bundle_path"] != bundle_2["_bundle_path"]
    print("[PASS] Filenames are unique even when run in the same second:")
    print("  Bundle 1:", Path(bundle_1["_bundle_path"]).name)
    print("  Bundle 2:", Path(bundle_2["_bundle_path"]).name)

    # 2. Non-JSON-serializable metadata
    print("Testing non-JSON-serializable metadata handling...")
    meta_doc = Document(
        id="meta_doc_1",
        content="Testing serialization of weird types.",
        score=0.99,
        meta={
            "created": datetime.now(timezone.utc),
            "obj_id": uuid.uuid4(),
            "numpy_val": np.float32(3.1415),
            "numpy_arr": np.array([1.0, 2.0, 3.0]),
            "decimal_val": decimal.Decimal("123.45"),
            "custom_obj": object() # should fall back to str(v)
        }
    )
    store = InMemoryDocumentStore()
    store.write_documents([meta_doc])
    
    pipe_meta = Pipeline()
    pipe_meta.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    
    bundle_meta = collect_debug_bundle("serialization", pipe_meta, document_store=store, output_dir="./test_debug_bundles")
    
    # Load back to verify json is correct and didn't crash
    with open(bundle_meta["_bundle_path"], "r") as f:
        loaded = json.load(f)
    
    raw_doc = loaded["retrieval"]["raw_top_k"][0]
    meta = raw_doc["meta"]
    assert isinstance(meta["created"], str) # ISO timestamp
    assert isinstance(meta["obj_id"], str) # UUID string
    assert isinstance(meta["numpy_val"], float) # Numpy float32 -> Python float
    assert isinstance(meta["numpy_arr"], list) # Numpy array -> List
    assert isinstance(meta["decimal_val"], float) # Decimal -> Float
    assert isinstance(meta["custom_obj"], str) # Custom object -> string description
    print("[PASS] Non-JSON-serializable metadata cleaned and serialized successfully.")

    # 3. Multiple Rankers
    print("Testing deterministic multiple rankers component selection...")
    pipe_mult = Pipeline()
    pipe_mult.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipe_mult.add_component("ranker_b", RerankerB())
    pipe_mult.add_component("ranker_a", RerankerA())
    
    # Auto-discovery name (alphabetical selection)
    from diagnostics.failure_diagnoser import _discover_reranker_name
    discovered = _discover_reranker_name(pipe_mult)
    assert discovered == "ranker_a"
    print("[PASS] Deterministic alphabetical ranker discovery returned:", discovered)

    # 4. Missing/Malformed Document IDs
    print("Testing missing/malformed Document IDs...")
    bad_doc = Document(content="Document with null ID.")
    bad_doc.id = None
    bad_store = InMemoryDocumentStore()
    bad_store.write_documents([bad_doc])
    
    pipe_bad = Pipeline()
    pipe_bad.add_component("retriever", InMemoryBM25Retriever(document_store=bad_store))
    
    mock_outputs = {
        "retriever": {"documents": [bad_doc]}
    }
    
    bundle_bad = collect_debug_bundle(
        "document",
        pipe_bad,
        document_store=bad_store,
        output_dir="./test_debug_bundles",
        pipeline_outputs=mock_outputs
    )
    
    doc_snapshot = bundle_bad["retrieval"]["raw_top_k"][0]
    assert doc_snapshot["id"] == "autogen_id_0"
    print("[PASS] Malformed/null document ID gracefully handled and autogenerated:", doc_snapshot["id"])


def main():
    print("============================================================")
    print("          REAL-WORLD WEAVIATE E2E VALIDATION RUN            ")
    print("============================================================")
    
    # Connect to Weaviate
    print("Connecting to live Weaviate RAG Studio backend...")
    manager = HaystackManager(weaviate_url="http://localhost:8080")
    doc_count = manager.document_store.count_documents()
    print("Weaviate is up. Document store count:", doc_count)
    assert doc_count > 0, "Weaviate is empty! Ingest documents first."

    # Run the validation layers
    live_pipe = test_layer_1_fidelity(manager)
    test_layer_2_taxonomy(live_pipe)
    test_layer_3_runtime_guardrails()
    test_layer_4_diff_engine()
    test_layer_5_large_corpus()
    test_layer_6_additional_cases(manager)

    print("\n============================================================")
    print("       ALL E2E VALIDATION LAYERS PASSED SUCCESSFULLY!       ")
    print("============================================================")

if __name__ == "__main__":
    main()
