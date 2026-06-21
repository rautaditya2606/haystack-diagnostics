import json
import pathlib
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from diagnostics.document_validator import validate_document_store
from diagnostics.pipeline_inspector import inspect_pipeline
from diagnostics.failure_diagnoser import diagnose_retrieval_failure

# Initialize FastMCP Server
mcp = FastMCP("Haystack Diagnostics Engine")


@mcp.tool()
def validate_store(
    store_type: str = "in_memory",
    store_data_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_index: Optional[str] = None,
    expected_metadata_keys: Optional[List[str]] = None,
    expected_embedding_dim: Optional[int] = None,
    short_chunk_threshold: int = 50,
) -> str:
    """
    Validates the health of a document store.
    
    :param store_type: Either 'in_memory' or 'qdrant'.
    :param store_data_path: Path to a JSON file containing serialized documents (required for in_memory).
    :param qdrant_url: URL to the Qdrant instance (required for qdrant).
    :param qdrant_index: Qdrant collection name/index (required for qdrant).
    :param expected_metadata_keys: List of metadata keys expected in documents.
    :param expected_embedding_dim: Expected embedding dimension size.
    :param short_chunk_threshold: Minimum character length for document text.
    """
    if store_type == "in_memory":
        if not store_data_path:
            return "Error: store_data_path is required for in_memory validation."
        
        path = pathlib.Path(store_data_path)
        if not path.exists():
            return f"Error: Document data file not found at {store_data_path}."
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Format data to Haystack Document list
            documents = []
            if isinstance(data, list):
                for doc_dict in data:
                    documents.append(Document.from_dict(doc_dict))
            else:
                return "Error: Document data must be a list of document objects."
            
            document_store = InMemoryDocumentStore()
            document_store.write_documents(documents)
        except Exception as e:
            return f"Error loading In-Memory document store: {str(e)}"
            
    elif store_type == "qdrant":
        if not qdrant_url or not qdrant_index:
            return "Error: qdrant_url and qdrant_index are required for qdrant store validation."
        
        try:
            from qdrant_haystack import QdrantDocumentStore
            document_store = QdrantDocumentStore(
                url=qdrant_url,
                index=qdrant_index
            )
        except ImportError:
            return "Error: 'qdrant-haystack' package is not installed in the environment."
        except Exception as e:
            return f"Error connecting to QdrantDocumentStore: {str(e)}"
    else:
        return f"Error: Unsupported store type '{store_type}'."

    # Execute validator
    try:
        report = validate_document_store(
            document_store=document_store,
            expected_metadata_keys=expected_metadata_keys,
            expected_embedding_dim=expected_embedding_dim,
            short_chunk_threshold=short_chunk_threshold
        )
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing document store validation: {str(e)}"


@mcp.tool()
def inspect_pipeline_graph(pipeline_config_path: str) -> str:
    """
    Inspects and serializes a Haystack Pipeline configuration.
    
    :param pipeline_config_path: Path to the YAML or JSON serialized pipeline file.
    """
    path = pathlib.Path(pipeline_config_path)
    if not path.exists():
        return f"Error: Pipeline config file not found at {pipeline_config_path}."

    try:
        if path.suffix in (".yaml", ".yml"):
            pipeline = Pipeline.load(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                pipeline = Pipeline.loads(f.read())
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

    try:
        report = inspect_pipeline(pipeline)
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing pipeline inspection: {str(e)}"


@mcp.tool()
def diagnose_retrieval(
    pipeline_config_path: str,
    query: str,
    retriever_component_name: Optional[str] = None,
    pipeline_inputs: Optional[str] = None,
    expected_answer: Optional[str] = None,
    ranking_threshold: float = 0.5,
) -> str:
    """
    Runs retrieval diagnostics on a query for a serialized pipeline.
    
    :param pipeline_config_path: Path to the serialized pipeline config file.
    :param query: Query string to run diagnostics against.
    :param retriever_component_name: Optional name of retriever.
    :param pipeline_inputs: Optional JSON string of pipeline inputs.
    :param expected_answer: Optional expected answer string.
    :param ranking_threshold: Threshold below which top document score triggers ranking failure.
    """
    path = pathlib.Path(pipeline_config_path)
    if not path.exists():
        return f"Error: Pipeline config file not found at {pipeline_config_path}."

    # Parse pipeline inputs from JSON string
    inputs_dict = None
    if pipeline_inputs:
        try:
            inputs_dict = json.loads(pipeline_inputs)
        except Exception as e:
            return f"Error parsing pipeline_inputs JSON: {str(e)}"

    # Load Pipeline
    try:
        if path.suffix in (".yaml", ".yml"):
            pipeline = Pipeline.load(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                pipeline = Pipeline.loads(f.read())
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

    # Run Diagnoser
    try:
        report = diagnose_retrieval_failure(
            pipeline=pipeline,
            query=query,
            retriever_component_name=retriever_component_name,
            pipeline_inputs=inputs_dict,
            expected_answer=expected_answer,
            ranking_threshold=ranking_threshold
        )
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing failure diagnostics: {str(e)}"


if __name__ == "__main__":
    mcp.run()
