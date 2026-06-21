import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts non-serializable fields (like UUIDs/datetimes) to strings."""
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, uuid.UUID):
            cleaned[k] = str(v)
        elif isinstance(v, datetime):
            cleaned[k] = v.isoformat()
        elif isinstance(v, dict):
            cleaned[k] = _clean_metadata(v)
        elif isinstance(v, list):
            cleaned[k] = [str(x) if isinstance(x, uuid.UUID) else x for x in v]
        else:
            cleaned[k] = v
    return cleaned


def _discover_component_names(pipeline):
    """
    Scans the pipeline structure to automatically find retriever and generator component names.
    """
    retriever_name = None
    generator_name = None

    if hasattr(pipeline, "graph") and pipeline.graph is not None:
        for node_name, attrs in pipeline.graph.nodes(data=True):
            instance = attrs.get("instance")
            class_name = instance.__class__.__name__ if instance else ""
            if "Retriever" in class_name and not retriever_name:
                retriever_name = node_name
            if ("Generator" in class_name or "LLM" in class_name) and not generator_name:
                generator_name = node_name
    else:
        pipe_dict = pipeline.to_dict()
        for comp_name, comp_info in pipe_dict.get("components", {}).items():
            type_str = comp_info.get("type", "")
            class_name = type_str.split(".")[-1]
            if "Retriever" in class_name and not retriever_name:
                retriever_name = comp_name
            if ("Generator" in class_name or "LLM" in class_name) and not generator_name:
                generator_name = comp_name

    return retriever_name, generator_name


def diagnose_retrieval_failure(
    pipeline,
    query: str,
    retriever_component_name: Optional[str] = None,
    pipeline_inputs: Optional[Dict[str, Any]] = None,
    expected_answer: Optional[str] = None,
    ranking_threshold: float = 0.5,
    pipeline_outputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Runs the pipeline (or uses pre-computed outputs), extracts intermediate retriever output, and classifies RAG failure:
    1. No Results: Retriever returned 0 documents.
    2. Empty Context: Retriever returned documents, but their combined content is empty or missing.
    3. Generator Failure: Generator output is empty, indicates refusal, or doesn't match expected answer.
    4. Ranking Failure: Top document score is below the ranking_threshold.

    :param pipeline: The Haystack Pipeline instance.
    :param query: The query text.
    :param retriever_component_name: Optional name of the retriever component. Auto-discovered if None.
    :param pipeline_inputs: Optional dict of inputs for pipeline.run(). Auto-populated if None.
    :param expected_answer: Optional ground truth answer to validate generator output.
    :param ranking_threshold: Threshold below which top document score flags ranking degradation.
    :param pipeline_outputs: Optional dict of pre-computed outputs from a single pipeline execution.
    :return: A structured diagnostic dictionary report.
    """
    # 1. Component discovery
    discovered_retriever, discovered_generator = _discover_component_names(pipeline)
    
    if not retriever_component_name:
        retriever_component_name = discovered_retriever

    if not retriever_component_name:
        raise ValueError(
            "Could not auto-discover any retriever component in the pipeline. "
            "Please specify retriever_component_name."
        )

    # 2. Run pipeline or use pre-computed outputs
    if pipeline_outputs is not None:
        results = pipeline_outputs
    else:
        # Auto-populate inputs if not provided
        if pipeline_inputs is None:
            pipeline_inputs = {}

            # Inspect required inputs from pipeline
            try:
                inputs_schema = pipeline.inputs()
            except Exception:
                inputs_schema = {}

            for comp_name, sockets in inputs_schema.items():
                comp_inputs = pipeline_inputs.setdefault(comp_name, {})
                for socket_name, socket_info in sockets.items():
                    # Map the query to input sockets representing text inputs
                    if socket_name in ("query", "text", "question") and socket_name not in comp_inputs:
                        comp_inputs[socket_name] = query

        try:
            results = pipeline.run(
                data=pipeline_inputs,
                include_outputs_from={retriever_component_name}
            )
        except Exception as e:
            return {
                "status": "error",
                "error": f"Pipeline execution failed: {str(e)}",
                "failure_type": "PIPELINE_EXECUTION_ERROR",
            }

    # 4. Extract retriever and generator outputs
    retriever_output = results.get(retriever_component_name, {})
    retrieved_docs = retriever_output.get("documents", [])

    # Find the generator output in the results
    generator_output = {}
    generator_name = discovered_generator
    
    # Locate generator output in the results dictionary
    if generator_name and generator_name in results:
        generator_output = results[generator_name]
    else:
        # Fallback: scan results for components containing 'replies'
        for comp_name, comp_out in results.items():
            if isinstance(comp_out, dict) and "replies" in comp_out:
                generator_name = comp_name
                generator_output = comp_out
                break

    generated_replies = generator_output.get("replies", [])
    generated_answer = generated_replies[0] if generated_replies else None

    # 5. Execute Failure Classification checks
    triggered_checks = {
        "no_results": False,
        "empty_context": False,
        "generator_failure": False,
        "ranking_failure": False,
    }

    # Check 1: No Results
    if not retrieved_docs:
        triggered_checks["no_results"] = True

    # Check 2: Ranking Failure
    top_score = None
    if not triggered_checks["no_results"] and retrieved_docs:
        # Sort documents by score descending (handling None scores gracefully)
        docs_with_scores = [doc for doc in retrieved_docs if doc.score is not None]
        if docs_with_scores:
            top_doc = max(docs_with_scores, key=lambda d: d.score)
            top_score = top_doc.score
            if top_score < ranking_threshold:
                triggered_checks["ranking_failure"] = True

    # Check 3: Empty Context
    total_context_len = 0
    if not triggered_checks["no_results"] and not triggered_checks["ranking_failure"]:
        combined_content = "".join([doc.content for doc in retrieved_docs if doc.content is not None])
        total_context_len = len(combined_content.strip())
        if total_context_len == 0:
            triggered_checks["empty_context"] = True

    # Check 4: Generator Failure
    is_refusal = False
    matches_expected = True
    
    if not triggered_checks["no_results"] and not triggered_checks["ranking_failure"] and not triggered_checks["empty_context"]:
        if generated_answer:
            # Check for typical LLM refusal keywords
            refusal_keywords = [
                r"i do not know",
                r"i don't know",
                r"not mentioned",
                r"sorry, but",
                r"unable to answer",
                r"cannot find any information",
                r"no information in the context",
                r"don't have.*enough",
                r"do not have.*enough",
                r"sorry, i don't",
                r"i'm sorry, but",
                r"i'm sorry, i don't",
                r"not enough.*information",
                r"insufficient information",
            ]
            answer_lower = generated_answer.lower()
            if any(re.search(pattern, answer_lower) for pattern in refusal_keywords):
                is_refusal = True
                triggered_checks["generator_failure"] = True

            # If expected_answer is provided, check if the output matches it
            if expected_answer:
                expected_lower = expected_answer.lower().strip()
                # Simple keyword matching: split expected answer into words longer than 3 chars
                keywords = [w for w in re.findall(r"\b\w{4,}\b", expected_lower)]
                if keywords:
                    matches_count = sum(1 for kw in keywords if kw in answer_lower)
                    match_ratio = matches_count / len(keywords)
                    # If less than 20% of keywords match, we flag mismatch
                    if match_ratio < 0.2:
                        matches_expected = False
                        triggered_checks["generator_failure"] = True
                else:
                    # Fallback to simple inclusion check
                    if expected_lower not in answer_lower:
                        matches_expected = False
                        triggered_checks["generator_failure"] = True
        else:
            # Generator returned nothing
            triggered_checks["generator_failure"] = True

    # 6. Determine primary classification in sequence
    primary_failure = "SUCCESS"
    if triggered_checks["no_results"]:
        primary_failure = "NO_RESULTS"
    elif triggered_checks["ranking_failure"]:
        primary_failure = "RANKING_FAILURE"
    elif triggered_checks["empty_context"]:
        primary_failure = "EMPTY_CONTEXT"
    elif triggered_checks["generator_failure"]:
        primary_failure = "GENERATOR_FAILURE"

    # Assemble document info for detailed diagnostics
    documents_diagnostics = []
    for idx, doc in enumerate(retrieved_docs):
        documents_diagnostics.append({
            "id": doc.id,
            "score": doc.score,
            "content_preview": doc.content[:150] + "..." if doc.content else None,
            "meta": _clean_metadata(doc.meta or {}),
            "rank": idx + 1,
        })

    return {
        "status": "success",
        "failure_type": primary_failure,
        "query": query,
        "diagnostics": {
            "retriever": {
                "component_name": retriever_component_name,
                "documents_retrieved": len(retrieved_docs),
                "top_score": top_score,
                "documents": documents_diagnostics,
            },
            "context": {
                "total_length_chars": total_context_len,
                "has_usable_context": not triggered_checks["empty_context"] and len(retrieved_docs) > 0,
            },
            "generator": {
                "component_name": generator_name,
                "generated_answer": generated_answer,
                "is_refusal": is_refusal,
                "matches_expected": matches_expected if expected_answer else None,
            },
            "ranking": {
                "ranking_threshold": ranking_threshold,
                "top_score": top_score,
                "is_below_threshold": triggered_checks["ranking_failure"],
            },
            "triggered_checks": triggered_checks,
        },
    }
