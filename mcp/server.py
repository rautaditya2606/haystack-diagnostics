import json
import os
import pathlib
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from diagnostics.document_validator import validate_document_store
from diagnostics.pipeline_inspector import inspect_pipeline
from diagnostics.failure_diagnoser import diagnose_retrieval_failure
from diagnostics.debug_bundler import collect_debug_bundle

# Initialize FastMCP Server
mcp = FastMCP("Haystack Diagnostics Engine")


def _load_pipeline(
    pipeline_config_path: Optional[str],
    pipeline_config_content: Optional[str],
) -> Pipeline:
    """
    Loads a Haystack Pipeline from YAML/JSON content or a file path.
    Precedence: pipeline_config_content wins over pipeline_config_path if both are supplied.
    Raises ValueError if neither is provided.
    """
    if not pipeline_config_path and not pipeline_config_content:
        raise ValueError("Either pipeline_config_path or pipeline_config_content must be provided.")

    if pipeline_config_content:
        return Pipeline.loads(pipeline_config_content)

    path = pathlib.Path(pipeline_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found at {pipeline_config_path}.")
    if path.suffix in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            return Pipeline.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return Pipeline.loads(f.read())


def _load_document_store(
    store_type: str,
    documents_data: Optional[List[Dict[str, Any]]] = None,
    store_data_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_index: Optional[str] = None,
    weaviate_url: Optional[str] = None,
):
    """Builds a document store instance from the given parameters."""
    if store_type == "in_memory":
        data = None
        if documents_data is not None:
            data = documents_data
        elif store_data_path:
            path = pathlib.Path(store_data_path)
            if not path.exists():
                raise FileNotFoundError(f"Document data file not found at {store_data_path}.")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise ValueError("Either documents_data or store_data_path must be provided for in_memory.")

        if not isinstance(data, list):
            raise ValueError("Document data must be a list of document objects.")

        documents = [Document.from_dict(d) for d in data]
        store = InMemoryDocumentStore()
        store.write_documents(documents)
        return store

    elif store_type == "qdrant":
        url = qdrant_url or os.environ.get("QDRANT_URL")
        if not url or not qdrant_index:
            raise ValueError("qdrant_url (or QDRANT_URL env var) and qdrant_index are required.")
        try:
            from qdrant_haystack import QdrantDocumentStore
            return QdrantDocumentStore(url=url, index=qdrant_index)
        except ImportError:
            raise ImportError("'qdrant-haystack' package is not installed.")

    elif store_type == "weaviate":
        url = weaviate_url or os.environ.get("WEAVIATE_URL")
        if not url:
            raise ValueError("weaviate_url (or WEAVIATE_URL env var) is required.")
        try:
            from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
            return WeaviateDocumentStore(url=url)
        except ImportError:
            raise ImportError("'weaviate-haystack' package is not installed.")
    else:
        raise ValueError(f"Unsupported store type '{store_type}'.")


@mcp.tool()
def validate_store(
    store_type: str = "in_memory",
    store_data_path: Optional[str] = None,
    documents_data: Optional[List[Dict[str, Any]]] = None,
    qdrant_url: Optional[str] = None,
    qdrant_index: Optional[str] = None,
    weaviate_url: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    expected_metadata_keys: Optional[List[str]] = None,
    expected_embedding_dim: Optional[int] = None,
    short_chunk_threshold: int = 50,
    validate_embeddings: bool = True,
) -> str:
    """
    Validates the health of a document store.

    :param store_type: Either 'in_memory', 'qdrant', or 'weaviate'.
    :param store_data_path: Path to a JSON file containing serialized documents (for in_memory).
    :param documents_data: Inline list of document dicts (alternative to store_data_path for in_memory).
    :param qdrant_url: URL to the Qdrant instance (or QDRANT_URL env var).
    :param qdrant_index: Qdrant collection name.
    :param weaviate_url: URL to the Weaviate instance (or WEAVIATE_URL env var).
    :param filters: Optional metadata filters to scope the validation.
    :param expected_metadata_keys: List of metadata keys expected in every document.
    :param expected_embedding_dim: Expected embedding dimension size.
    :param short_chunk_threshold: Minimum character length for document text.
    :param validate_embeddings: Whether to fetch and validate embeddings.
    """
    try:
        document_store = _load_document_store(
            store_type=store_type,
            documents_data=documents_data,
            store_data_path=store_data_path,
            qdrant_url=qdrant_url,
            qdrant_index=qdrant_index,
            weaviate_url=weaviate_url,
        )
    except Exception as e:
        return f"Error loading document store: {str(e)}"

    try:
        report = validate_document_store(
            document_store=document_store,
            expected_metadata_keys=expected_metadata_keys,
            expected_embedding_dim=expected_embedding_dim,
            short_chunk_threshold=short_chunk_threshold,
            filters=filters,
            validate_embeddings=validate_embeddings,
        )
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing document store validation: {str(e)}"


@mcp.tool()
def inspect_pipeline_graph(
    pipeline_config_path: Optional[str] = None,
    pipeline_config_content: Optional[str] = None,
) -> str:
    """
    Inspects and serializes a Haystack Pipeline configuration.

    Precedence: pipeline_config_content wins over pipeline_config_path if both are provided.
    Error is raised if neither is provided.

    :param pipeline_config_path: Path to the YAML or JSON serialized pipeline file.
    :param pipeline_config_content: YAML or JSON string of the serialized pipeline.
    """
    try:
        pipeline = _load_pipeline(pipeline_config_path, pipeline_config_content)
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

    try:
        report = inspect_pipeline(pipeline)
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing pipeline inspection: {str(e)}"


@mcp.tool()
def diagnose_retrieval(
    query: str,
    pipeline_config_path: Optional[str] = None,
    pipeline_config_content: Optional[str] = None,
    retriever_component_name: Optional[str] = None,
    reranker_component_name: Optional[str] = None,
    pipeline_inputs: Optional[str] = None,
    expected_answer: Optional[str] = None,
    ranking_threshold: float = 0.5,
    pipeline_outputs: Optional[str] = None,
    relevant_doc_id: Optional[str] = None,
) -> str:
    """
    Runs retrieval diagnostics on a query for a serialized Haystack pipeline.

    Precedence: pipeline_config_content wins over pipeline_config_path if both are provided.
    Error is raised if neither is provided.

    Failure types returned:
    - NO_RESULTS, EMPTY_CONTEXT, GENERATOR_FAILURE
    - RANKING_FAILURE (legacy, no relevant_doc_id provided)
    - SCORE_BELOW_CUTOFF (relevant_doc_id provided, doc not in retriever output)
    - CONTEXT_LOSS (relevant_doc_id in retriever output but dropped by reranker)

    :param query: Query string to run diagnostics against.
    :param pipeline_config_path: Path to the serialized pipeline config file.
    :param pipeline_config_content: YAML or JSON string of the serialized pipeline.
    :param retriever_component_name: Optional retriever component name (auto-discovered if None).
    :param reranker_component_name: Optional reranker component name (auto-discovered if None).
    :param pipeline_inputs: Optional JSON string of pipeline.run() inputs.
    :param expected_answer: Optional ground truth answer string.
    :param ranking_threshold: Score threshold below which ranking failure is triggered.
    :param pipeline_outputs: Optional JSON string of pre-computed pipeline outputs.
    :param relevant_doc_id: Optional doc ID expected to be retrieved; enables SCORE_BELOW_CUTOFF
        vs CONTEXT_LOSS distinction.
    """
    inputs_dict = None
    if pipeline_inputs:
        try:
            inputs_dict = json.loads(pipeline_inputs)
        except Exception as e:
            return f"Error parsing pipeline_inputs JSON: {str(e)}"

    outputs_dict = None
    if pipeline_outputs:
        try:
            outputs_dict = json.loads(pipeline_outputs)
        except Exception as e:
            return f"Error parsing pipeline_outputs JSON: {str(e)}"

    try:
        pipeline = _load_pipeline(pipeline_config_path, pipeline_config_content)
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

    try:
        report = diagnose_retrieval_failure(
            pipeline=pipeline,
            query=query,
            retriever_component_name=retriever_component_name,
            reranker_component_name=reranker_component_name,
            pipeline_inputs=inputs_dict,
            expected_answer=expected_answer,
            ranking_threshold=ranking_threshold,
            pipeline_outputs=outputs_dict,
            relevant_doc_id=relevant_doc_id,
        )
        return json.dumps(report, indent=2)
    except Exception as e:
        return f"Error executing failure diagnostics: {str(e)}"


@mcp.tool()
def collect_debug_bundle_tool(
    query: str,
    pipeline_config_path: Optional[str] = None,
    pipeline_config_content: Optional[str] = None,
    store_type: str = "in_memory",
    documents_data: Optional[List[Dict[str, Any]]] = None,
    store_data_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_index: Optional[str] = None,
    weaviate_url: Optional[str] = None,
    retriever_component_name: Optional[str] = None,
    reranker_component_name: Optional[str] = None,
    ranking_threshold: float = 0.5,
    relevant_doc_id: Optional[str] = None,
    output_dir: str = "./debug_bundles",
) -> str:
    """
    Captures the full debug bundle for a single query: pipeline graph, component params,
    raw top-k docs (pre-reranker), reranked top-k docs, prompt snapshot, generated answer,
    failure classification, and corpus health checks scoped to retrieved doc IDs.

    Persists the bundle to {output_dir}/{query_slug}_{timestamp}.json.
    Returns the bundle as a JSON string.

    Precedence: pipeline_config_content wins over pipeline_config_path if both are provided.
    Error is raised if neither is provided.

    :param query: The query string.
    :param pipeline_config_path: Path to the serialized pipeline config file.
    :param pipeline_config_content: YAML or JSON string of the serialized pipeline.
    :param store_type: 'in_memory', 'qdrant', or 'weaviate' (for scoped corpus checks).
    :param documents_data: Inline list of document dicts (for in_memory store).
    :param store_data_path: Path to JSON document file (for in_memory store).
    :param qdrant_url: Qdrant instance URL (or QDRANT_URL env var).
    :param qdrant_index: Qdrant collection name.
    :param weaviate_url: Weaviate instance URL (or WEAVIATE_URL env var).
    :param retriever_component_name: Optional retriever name (auto-discovered if None).
    :param reranker_component_name: Optional reranker name (auto-discovered if None).
    :param ranking_threshold: Score threshold for SCORE_BELOW_CUTOFF classification.
    :param relevant_doc_id: Optional expected doc ID for CONTEXT_LOSS detection.
    :param output_dir: Directory to write bundle JSON. Defaults to ./debug_bundles.
    """
    try:
        pipeline = _load_pipeline(pipeline_config_path, pipeline_config_content)
    except Exception as e:
        return f"Error loading pipeline: {str(e)}"

    # Document store is optional (only needed for scoped corpus checks)
    document_store = None
    if documents_data or store_data_path or qdrant_url or weaviate_url:
        try:
            document_store = _load_document_store(
                store_type=store_type,
                documents_data=documents_data,
                store_data_path=store_data_path,
                qdrant_url=qdrant_url,
                qdrant_index=qdrant_index,
                weaviate_url=weaviate_url,
            )
        except Exception as e:
            return f"Error loading document store: {str(e)}"

    try:
        bundle = collect_debug_bundle(
            query=query,
            pipeline=pipeline,
            document_store=document_store,
            retriever_component_name=retriever_component_name,
            reranker_component_name=reranker_component_name,
            ranking_threshold=ranking_threshold,
            relevant_doc_id=relevant_doc_id,
            output_dir=output_dir,
        )
        return json.dumps(bundle, indent=2, default=str)
    except Exception as e:
        return f"Error collecting debug bundle: {str(e)}"


if __name__ == "__main__":
    mcp.run()
