"""
diagnostics/debug_bundler.py

Provides collect_debug_bundle() and diff_debug_bundles() for capturing and
comparing full query execution state in Haystack RAG pipelines.

CLI usage:
    python -m diagnostics.debug_bundler diff <bundle_a.json> <bundle_b.json>
"""

import difflib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

from diagnostics.failure_diagnoser import (
    _clean_metadata,
    _discover_component_names,
    _discover_reranker_name,
    diagnose_retrieval_failure,
)
from diagnostics.pipeline_inspector import inspect_pipeline


# ---------------------------------------------------------------------------
# Default set of known volatile init_parameter keys to suppress in config diffs.
#
# These are fields that change between runs even when the pipeline is logically
# identical — most notably InMemoryDocumentStore.index, which is a random UUID
# generated at instantiation time.
#
# Format: "component_name.param_key"  — suppresses only for that component.
#         "*.param_key"               — suppresses the key across ALL components.
#
# Pass an empty set to disable all filtering:
#   diff_debug_bundles(a, b, ignore_config_paths=set())
# ---------------------------------------------------------------------------
DEFAULT_VOLATILE_CONFIG_PATHS: FrozenSet[str] = frozenset({
    # InMemoryDocumentStore generates a new UUID index on every instantiation.
    "*.index",
})


def _query_slug(query: str, max_len: int = 40) -> str:
    """Converts a query string to a filesystem-safe slug (lowercase, underscores)."""
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower().strip())
    slug = slug.strip("_")
    return slug[:max_len] if len(slug) > max_len else slug


def _extract_component_params(pipeline) -> Dict[str, Any]:
    """
    Extracts init_parameters for every component in the pipeline via to_dict().
    Falls back to class name only if introspection fails.
    """
    params: Dict[str, Any] = {}
    try:
        pipe_dict = pipeline.to_dict()
        for comp_name, comp_info in pipe_dict.get("components", {}).items():
            params[comp_name] = {
                "type": comp_info.get("type", ""),
                "init_parameters": comp_info.get("init_parameters", {}),
            }
    except Exception:
        if hasattr(pipeline, "graph") and pipeline.graph is not None:
            for node_name, attrs in pipeline.graph.nodes(data=True):
                instance = attrs.get("instance")
                class_name = instance.__class__.__name__ if instance else "UnknownComponent"
                module_name = instance.__class__.__module__ if instance else "unknown"
                params[node_name] = {
                    "type": f"{module_name}.{class_name}",
                    "init_parameters": {},
                }
    return params


def _extract_doc_snapshots(docs: List[Any]) -> List[Dict[str, Any]]:
    """Converts a list of Haystack Document objects to serializable snapshot dicts."""
    snapshots = []
    for idx, doc in enumerate(docs):
        content = doc.content or ""
        snapshots.append({
            "rank": idx + 1,
            "id": doc.id or f"autogen_id_{idx}",
            "score": doc.score,
            "content_preview": (content[:200] + "...") if len(content) > 200 else content,
            "meta": _clean_metadata(doc.meta or {}),
        })
    return snapshots


def _scoped_corpus_check(document_store, doc_ids: List[str]) -> Dict[str, Any]:
    """
    Runs a lightweight health check scoped only to the retrieved document IDs —
    not the full document store. Keeps corpus-level and query-level concerns separate
    (per GioiaZheng's feedback: mixing them inflates the debugging surface).

    Checks: content_none, empty_content, null_embedding.
    """
    if document_store is None or not doc_ids:
        return {
            "scoped_to_doc_ids": doc_ids,
            "docs_found": 0,
            "issues": [],
            "skipped": True,
            "reason": "No document store provided or no documents were retrieved.",
        }

    issues: List[Dict[str, str]] = []
    seen_ids: set = set()

    try:
        # Attempt per-ID filter first; fall back to full scan once if it fails
        docs: List[Any] = []
        fetched_all = False
        for doc_id in doc_ids:
            if fetched_all:
                break
            try:
                results = document_store.filter_documents(
                    filters={"field": "id", "operator": "==", "value": doc_id}
                )
                docs.extend(results)
            except Exception:
                try:
                    all_docs = document_store.filter_documents()
                    docs = [d for d in all_docs if d.id in set(doc_ids)]
                    fetched_all = True
                except Exception:
                    break

        for doc in docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)

            if doc.content is None:
                issues.append({"doc_id": doc.id, "issue": "content_none"})
            elif doc.content.strip() == "":
                issues.append({"doc_id": doc.id, "issue": "empty_content"})

            if doc.embedding is None:
                issues.append({"doc_id": doc.id, "issue": "null_embedding"})

        return {
            "scoped_to_doc_ids": doc_ids,
            "docs_found": len(seen_ids),
            "issues": issues,
            "skipped": False,
        }
    except Exception as e:
        return {
            "scoped_to_doc_ids": doc_ids,
            "docs_found": 0,
            "issues": [],
            "skipped": True,
            "reason": f"Corpus check failed: {str(e)}",
        }


def collect_debug_bundle(
    query: str,
    pipeline,
    document_store=None,
    retriever_component_name: Optional[str] = None,
    reranker_component_name: Optional[str] = None,
    pipeline_inputs: Optional[Dict[str, Any]] = None,
    ranking_threshold: float = 0.5,
    relevant_doc_id: Optional[str] = None,
    output_dir: str = "./debug_bundles",
    pipeline_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Captures the full state of a single query execution as a structured, diffable JSON bundle.

    The bundle includes:
    - Pipeline graph, haystack version, and component init_parameters
    - Raw top-k documents from the retriever (scores, metadata, content previews)
    - Reranked top-k documents if a reranker is present in the pipeline
    - Final prompt snapshot (if PromptBuilder is in the pipeline)
    - Generated answer
    - Failure classification (NO_RESULTS, SCORE_BELOW_CUTOFF, CONTEXT_LOSS, etc.)
    - Corpus health checks scoped only to the retrieved document IDs (not the full store)

    CONTEXT_LOSS scope: v1 detects reranker demotion only — the expected document is present in
    raw_top_k but absent from the reranker's output. Context-packing truncation (doc survives
    reranking but is dropped by the prompt builder) is deferred; detecting it requires capturing
    PromptBuilder intermediate output, which couples the tool to prompt builder internals.

    Bundle filename: {query_slug}_{timestamp}.json
    The bundle_id UUID is stored inside the JSON for programmatic reference.
    Filenames are human-readable and sort naturally across runs of the same query.

    :param query: The query string.
    :param pipeline: The Haystack Pipeline instance.
    :param document_store: Optional DocumentStore for scoped corpus checks on retrieved docs.
    :param retriever_component_name: Optional retriever name. Auto-discovered if None.
    :param reranker_component_name: Optional reranker name. Auto-discovered if None.
    :param pipeline_inputs: Optional pipeline.run() input dict. Auto-populated if None.
    :param ranking_threshold: Score threshold for SCORE_BELOW_CUTOFF classification.
    :param relevant_doc_id: Optional expected doc ID for CONTEXT_LOSS detection.
    :param output_dir: Directory to write bundle JSON. Defaults to ./debug_bundles (cwd).
    :param pipeline_outputs: Optional pre-computed pipeline run outputs.
    :return: The bundle as a Python dict (also persisted to disk).
    """
    bundle_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    # --- Discover component names ---
    discovered_retriever, _ = _discover_component_names(pipeline)
    if not retriever_component_name:
        retriever_component_name = discovered_retriever
    if not retriever_component_name:
        raise ValueError(
            "Could not auto-discover a retriever component. "
            "Pass retriever_component_name explicitly."
        )

    if not reranker_component_name:
        reranker_component_name = _discover_reranker_name(pipeline)

    # --- Auto-discover prompt builder ---
    prompt_builder_name = None
    if hasattr(pipeline, "graph") and pipeline.graph is not None:
        for node_name, attrs in pipeline.graph.nodes(data=True):
            instance = attrs.get("instance")
            class_name = instance.__class__.__name__ if instance else ""
            if "PromptBuilder" in class_name:
                prompt_builder_name = node_name
                break
    else:
        try:
            pipe_dict = pipeline.to_dict()
            for comp_name, comp_info in pipe_dict.get("components", {}).items():
                type_str = comp_info.get("type", "")
                class_name = type_str.split(".")[-1]
                if "PromptBuilder" in class_name:
                    prompt_builder_name = comp_name
                    break
        except Exception:
            pass

    # --- Build include_outputs_from ---
    include_from = {retriever_component_name}
    if reranker_component_name:
        include_from.add(reranker_component_name)
    if prompt_builder_name:
        include_from.add(prompt_builder_name)

    # --- Execute pipeline or use pre-computed outputs ---
    if pipeline_outputs is not None:
        results = pipeline_outputs
    else:
        # --- Build pipeline inputs ---
        if pipeline_inputs is None:
            pipeline_inputs = {}
            try:
                inputs_schema = pipeline.inputs()
            except Exception:
                inputs_schema = {}
            for comp_name, sockets in inputs_schema.items():
                comp_inputs = pipeline_inputs.setdefault(comp_name, {})
                for socket_name in sockets:
                    if socket_name in ("query", "text", "question") and socket_name not in comp_inputs:
                        comp_inputs[socket_name] = query

        # --- Run pipeline ---
        try:
            results = pipeline.run(data=pipeline_inputs, include_outputs_from=include_from)
        except Exception as e:
            return {
                "bundle_id": bundle_id,
                "created_at": created_at,
                "query": query,
                "status": "error",
                "error": f"Pipeline execution failed: {str(e)}",
            }

    # --- Extract retriever docs ---
    retriever_output = results.get(retriever_component_name, {})
    raw_docs = retriever_output.get("documents", [])

    # --- Extract reranker docs ---
    reranked_docs = None
    reranker_note = None
    if reranker_component_name:
        reranker_raw = results.get(reranker_component_name, {})
        if reranker_raw:
            reranked_docs = reranker_raw.get("documents", [])
        else:
            reranker_note = (
                f"Reranker '{reranker_component_name}' detected in pipeline but its output "
                "was not captured. CONTEXT_LOSS detection unavailable."
            )

    # --- Extract prompt snapshot ---
    prompt_snapshot = None
    for comp_out in results.values():
        if isinstance(comp_out, dict) and "prompt" in comp_out:
            prompt_snapshot = comp_out["prompt"]
            break

    # --- Extract generator output ---
    generator_name = None
    generated_answer = None
    for comp_name, comp_out in results.items():
        if isinstance(comp_out, dict) and "replies" in comp_out:
            generator_name = comp_name
            replies = comp_out.get("replies", [])
            generated_answer = replies[0] if replies else None
            break

    # --- Run failure classification (pass pre-computed results to avoid double run) ---
    diagnosis = diagnose_retrieval_failure(
        pipeline=pipeline,
        query=query,
        retriever_component_name=retriever_component_name,
        reranker_component_name=reranker_component_name,
        ranking_threshold=ranking_threshold,
        relevant_doc_id=relevant_doc_id,
        pipeline_outputs=results,
    )

    # --- Inspect pipeline structure ---
    pipeline_inspection = inspect_pipeline(pipeline)
    component_params = _extract_component_params(pipeline)

    # --- Haystack version ---
    try:
        import haystack
        haystack_version = getattr(haystack, "__version__", "unknown")
    except Exception:
        haystack_version = "unknown"

    # --- Scoped corpus check ---
    retrieved_doc_ids = [doc.id for doc in raw_docs if doc.id is not None]
    corpus_checks = _scoped_corpus_check(document_store, retrieved_doc_ids)

    # --- Assemble bundle ---
    bundle = {
        "bundle_id": bundle_id,
        "created_at": created_at,
        "query": query,
        "pipeline": {
            "haystack_version": haystack_version,
            "metadata": pipeline_inspection.get("metadata", {}),
            "components": component_params,
            "connections": pipeline_inspection.get("connections", []),
            "mermaid": pipeline_inspection.get("mermaid", ""),
        },
        "retrieval": {
            "retriever_component": retriever_component_name,
            "doc_count": len(raw_docs),
            "raw_top_k": _extract_doc_snapshots(raw_docs),
        },
        "reranker": {
            "detected": reranker_component_name is not None,
            "component": reranker_component_name,
            "reranked_top_k": _extract_doc_snapshots(reranked_docs) if reranked_docs is not None else None,
            "note": reranker_note,
        },
        "generation": {
            "component": generator_name,
            "prompt_snapshot": prompt_snapshot,
            "answer": generated_answer,
        },
        "failure_type": diagnosis.get("failure_type", "UNKNOWN"),
        "diagnostics": diagnosis.get("diagnostics", {}),
        "corpus_checks": corpus_checks,
    }

    # --- Persist to disk ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    slug = _query_slug(query)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Append first 8 characters of bundle_id to guarantee unique filenames (prevent collisions)
    filename = f"{slug}_{timestamp}_{bundle_id[:8]}.json"
    bundle_path = out_path / filename

    bundle["_bundle_path"] = str(bundle_path)

    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, default=str)

    return bundle


def diff_debug_bundles(
    bundle_a_path: str,
    bundle_b_path: str,
    ignore_config_paths: Optional[Union[Set[str], FrozenSet[str]]] = None,
) -> Dict[str, Any]:
    """
    Compares two persisted debug bundle JSON files and returns a structured diff.

    Reports:
    - Failure type change between runs
    - Score delta per document ID (documents that appeared, disappeared, or changed score)
    - Component init_parameters changes (with volatile fields suppressed by default)
    - Answer diff (character-level unified diff via difflib — no tokenizer dependency)

    :param bundle_a_path: Path to the baseline bundle JSON file.
    :param bundle_b_path: Path to the comparison bundle JSON file.
    :param ignore_config_paths: Set of ``"component_name.param_key"`` strings to exclude
        from the config diff.  Use ``"*.param_key"`` to ignore a param across *all*
        components.  Defaults to :data:`DEFAULT_VOLATILE_CONFIG_PATHS` which suppresses
        known runtime-volatile fields (e.g. ``InMemoryDocumentStore.index`` UUID).
        Pass an empty set to disable all filtering.
    :return: A structured diff dictionary.
    """
    if ignore_config_paths is None:
        ignore_config_paths = DEFAULT_VOLATILE_CONFIG_PATHS

    with open(bundle_a_path, "r", encoding="utf-8") as f:
        a = json.load(f)
    with open(bundle_b_path, "r", encoding="utf-8") as f:
        b = json.load(f)

    def _score_map(bundle: Dict) -> Dict[str, Optional[float]]:
        return {
            (doc.get("id") or f"autogen_id_{idx}"): doc.get("score")
            for idx, doc in enumerate(bundle.get("retrieval", {}).get("raw_top_k", []))
        }

    scores_a = _score_map(a)
    scores_b = _score_map(b)
    all_doc_ids = set(scores_a) | set(scores_b)

    score_deltas: Dict[str, Any] = {}
    appeared: List[Dict] = []
    disappeared: List[Dict] = []

    for doc_id in all_doc_ids:
        in_a, in_b = doc_id in scores_a, doc_id in scores_b
        if in_a and in_b:
            delta = (scores_b[doc_id] or 0.0) - (scores_a[doc_id] or 0.0)
            if abs(delta) > 1e-6:
                score_deltas[doc_id] = {
                    "score_a": scores_a[doc_id],
                    "score_b": scores_b[doc_id],
                    "delta": round(delta, 6),
                }
        elif in_b:
            appeared.append({"id": doc_id, "score_b": scores_b[doc_id]})
        else:
            disappeared.append({"id": doc_id, "score_a": scores_a[doc_id]})

    # Config diff — filter out known volatile (or user-specified) param keys
    def _filter_volatile(comp_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove ignored keys from a component's init_parameters before diffing."""
        result = {}
        for key, value in params.items():
            wildcard = f"*.{key}"
            specific = f"{comp_name}.{key}"
            if wildcard in ignore_config_paths or specific in ignore_config_paths:
                continue
            result[key] = value
        return result

    comps_a = a.get("pipeline", {}).get("components", {})
    comps_b = b.get("pipeline", {}).get("components", {})
    config_changes: Dict[str, Any] = {}
    for comp in set(comps_a) | set(comps_b):
        params_a = _filter_volatile(comp, comps_a.get(comp, {}).get("init_parameters", {}))
        params_b = _filter_volatile(comp, comps_b.get(comp, {}).get("init_parameters", {}))
        if params_a != params_b:
            config_changes[comp] = {"before": params_a, "after": params_b}

    # Answer diff — character-level unified diff, no tokenizer dependency
    answer_a = (a.get("generation", {}).get("answer") or "")
    answer_b = (b.get("generation", {}).get("answer") or "")
    answer_diff_lines = list(
        difflib.unified_diff(
            answer_a.splitlines(keepends=True),
            answer_b.splitlines(keepends=True),
            fromfile="bundle_a_answer",
            tofile="bundle_b_answer",
        )
    )

    return {
        "bundle_a": bundle_a_path,
        "bundle_b": bundle_b_path,
        "query_a": a.get("query"),
        "query_b": b.get("query"),
        "failure_type_change": {
            "before": a.get("failure_type"),
            "after": b.get("failure_type"),
            "changed": a.get("failure_type") != b.get("failure_type"),
        },
        "score_deltas": score_deltas,
        "docs_appeared": appeared,
        "docs_disappeared": disappeared,
        "config_changes": config_changes,
        "answer_diff": "".join(answer_diff_lines) if answer_diff_lines else "(no change)",
    }


def _cli_diff(argv: List[str]) -> None:
    """
    CLI handler for bundle diffing.
    Usage: python -m diagnostics.debug_bundler diff <bundle_a.json> <bundle_b.json>
    """
    if len(argv) < 2:
        print("Usage: python -m diagnostics.debug_bundler diff <bundle_a.json> <bundle_b.json>")
        sys.exit(1)
    try:
        diff = diff_debug_bundles(argv[0], argv[1])
        print(json.dumps(diff, indent=2, default=str))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        _cli_diff(sys.argv[2:])
    else:
        print("Usage: python -m diagnostics.debug_bundler diff <bundle_a.json> <bundle_b.json>")
        sys.exit(1)
