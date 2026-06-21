import json
import sys
import pathlib
import importlib.util
import pytest

# Load the local mcp/server.py directly to avoid collision with installed 'mcp' library package
server_path = pathlib.Path(__file__).parent.parent / "mcp" / "server.py"
spec = importlib.util.spec_from_file_location("local_mcp_server", server_path)
local_mcp_server = importlib.util.module_from_spec(spec)
sys.modules["local_mcp_server"] = local_mcp_server
spec.loader.exec_module(local_mcp_server)

validate_store = local_mcp_server.validate_store
inspect_pipeline_graph = local_mcp_server.inspect_pipeline_graph
diagnose_retrieval = local_mcp_server.diagnose_retrieval

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


def test_mcp_validate_store_inline():
    # Test validate_store with inline documents_data
    docs = [
        {"content": "Paris is the capital of France. It has many museums and monuments like the Eiffel Tower.", "meta": {"source": "wiki", "language": "en"}},
        {"content": "Berlin is the capital of Germany. It is known for its history, architecture, and nightlife.", "meta": {"source": "wiki", "language": "en"}}
    ]
    report_str = validate_store(
        store_type="in_memory",
        documents_data=docs,
        expected_metadata_keys=["source", "language"],
        short_chunk_threshold=20,
        validate_embeddings=False
    )
    report = json.loads(report_str)
    assert report["summary"]["total_documents"] == 2
    assert report["summary"]["invalid_documents"] == 0


def test_mcp_inspect_pipeline_inline():
    # Create simple pipeline and serialize it
    pb = PromptBuilder(template="Hello {{name}}")
    p = Pipeline()
    p.add_component("pb", pb)
    config_content = p.dumps()
    
    report_str = inspect_pipeline_graph(
        pipeline_config_content=config_content
    )
    report = json.loads(report_str)
    assert "pb" in report["components"]


def test_mcp_diagnose_retrieval_inline():
    # Create simple pipeline and serialize it
    # We will test diagnose_retrieval with pre-computed outputs
    pb = PromptBuilder(template="Hello {{name}}")
    p = Pipeline()
    p.add_component("pb", pb)
    
    ds = InMemoryDocumentStore()
    retriever = InMemoryBM25Retriever(document_store=ds)
    p.add_component("retriever", retriever)
    
    config_content = p.dumps()
    
    # We pass outputs that trigger NO_RESULTS
    outputs = {
        "retriever": {"documents": []},
        "generator": {"replies": ["I'm sorry, I don't know"]}
    }
    
    report_str = diagnose_retrieval(
        query="Paris",
        pipeline_config_content=config_content,
        retriever_component_name="retriever",
        pipeline_outputs=json.dumps(outputs)
    )
    report = json.loads(report_str)
    assert report["failure_type"] == "NO_RESULTS"
