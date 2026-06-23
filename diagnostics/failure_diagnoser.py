import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively converts non-serializable fields to strings and handles numpy/Decimal types."""
    import decimal
    import json

    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, uuid.UUID):
            cleaned[k] = str(v)
        elif isinstance(v, datetime):
            cleaned[k] = v.isoformat()
        elif isinstance(v, decimal.Decimal):
            cleaned[k] = float(v)
        elif hasattr(v, 'tolist') and hasattr(v, '__len__'):
            cleaned[k] = v.tolist()
        elif hasattr(v, 'item') and type(v).__module__ == 'numpy':
            cleaned[k] = v.item()
        elif isinstance(v, dict):
            cleaned[k] = _clean_metadata(v)
        elif isinstance(v, list):
            cleaned_list = []
            for x in v:
                if isinstance(x, dict):
                    cleaned_list.append(_clean_metadata(x))
                elif isinstance(x, uuid.UUID):
                    cleaned_list.append(str(x))
                elif isinstance(x, datetime):
                    cleaned_list.append(x.isoformat())
                elif isinstance(x, decimal.Decimal):
                    cleaned_list.append(float(x))
                elif hasattr(x, 'tolist') and hasattr(x, '__len__'):
                    cleaned_list.append(x.tolist())
                elif hasattr(x, 'item') and type(x).__module__ == 'numpy':
                    cleaned_list.append(x.item())
                else:
                    try:
                        json.dumps(x)
                        cleaned_list.append(x)
                    except Exception:
                        cleaned_list.append(str(x))
            cleaned[k] = cleaned_list
        else:
            try:
                json.dumps(v)
                cleaned[k] = v
            except Exception:
                cleaned[k] = str(v)
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


def _discover_reranker_name(pipeline) -> Optional[str]:
    """
    Scans the pipeline structure to automatically find a reranker component name.
    Alphabetically sorts discovered ranker names to make selection deterministic.
    Returns None if no reranker is present in the pipeline.
    """
    names = []
    if hasattr(pipeline, "graph") and pipeline.graph is not None:
        for node_name, attrs in pipeline.graph.nodes(data=True):
            instance = attrs.get("instance")
            class_name = instance.__class__.__name__ if instance else ""
            if "rank" in class_name.lower():
                names.append(node_name)
    else:
        try:
            pipe_dict = pipeline.to_dict()
            for comp_name, comp_info in pipe_dict.get("components", {}).items():
                type_str = comp_info.get("type", "")
                class_name = type_str.split(".")[-1]
                if "rank" in class_name.lower():
                    names.append(comp_name)
        except Exception:
            pass
    if names:
        return sorted(names)[0]
    return None


def diagnose_retrieval_failure(
    pipeline,
    query: str,
    retriever_component_name: Optional[str] = None,
    reranker_component_name: Optional[str] = None,
    pipeline_inputs: Optional[Dict[str, Any]] = None,
    expected_answer: Optional[str] = None,
    ranking_threshold: float = 0.5,
    pipeline_outputs: Optional[Dict[str, Any]] = None,
    relevant_doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the pipeline (or uses pre-computed outputs), extracts intermediate retriever output,
    and classifies RAG failure in the following sequence:

    1. NO_RESULTS: Retriever returned 0 documents.
    2. SCORE_BELOW_CUTOFF / RANKING_FAILURE / CONTEXT_LOSS: Score or ranking problem.
       - With relevant_doc_id hint:
         - Expected doc not in retrieved docs → SCORE_BELOW_CUTOFF
         - Expected doc in retrieved docs but dropped by reranker → CONTEXT_LOSS
         - Score below threshold (no doc-level issue identified) → SCORE_BELOW_CUTOFF
       - Without relevant_doc_id hint: score < threshold → RANKING_FAILURE (legacy, backwards-compat)
    3. EMPTY_CONTEXT: Retrieved docs exist but combined content is empty.
    4. GENERATOR_FAILURE: Generator returned nothing, refused, or mismatched expected answer.

    CONTEXT_LOSS scope (v1): Covers reranker demotion only — i.e., the expected doc is present
    in the retriever's raw_top_k but absent from the reranker's output. Context-packing truncation
    (doc survives reranking but is dropped by the prompt builder) is deferred to a future version
    and requires capturing PromptBuilder intermediate output.

    Runtime behaviour for missing reranker output: If a reranker is detected in the pipeline but
    its output is absent from pipeline_outputs (e.g., pre-computed without include_outputs_from),
    a warning note is surfaced in the return dict. CONTEXT_LOSS will not fire silently.

    :param pipeline: The Haystack Pipeline instance.
    :param query: The query text.
    :param retriever_component_name: Optional retriever component name. Auto-discovered if None.
    :param reranker_component_name: Optional reranker component name. Auto-discovered if None.
    :param pipeline_inputs: Optional dict of inputs for pipeline.run(). Auto-populated if None.
    :param expected_answer: Optional ground truth answer to validate generator output.
    :param ranking_threshold: Threshold below which top document score flags ranking degradation.
    :param pipeline_outputs: Optional dict of pre-computed outputs from a single pipeline execution.
    :param relevant_doc_id: Optional document ID expected to be retrieved. Enables SCORE_BELOW_CUTOFF
        vs CONTEXT_LOSS distinction. When absent, RANKING_FAILURE is used as the legacy bucket.
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

    if not reranker_component_name:
        reranker_component_name = _discover_reranker_name(pipeline)

    # 2. Run pipeline or use pre-computed outputs
    if pipeline_outputs is not None:
        results = pipeline_outputs
    else:
        if pipeline_inputs is None:
            pipeline_inputs = {}
            try:
                inputs_schema = pipeline.inputs()
            except Exception:
                inputs_schema = {}

            for comp_name, sockets in inputs_schema.items():
                comp_inputs = pipeline_inputs.setdefault(comp_name, {})
                for socket_name, socket_info in sockets.items():
                    if socket_name in ("query", "text", "question") and socket_name not in comp_inputs:
                        comp_inputs[socket_name] = query

        # Always include retriever; include reranker so CONTEXT_LOSS detection works
        include_from = {retriever_component_name}
        if reranker_component_name:
            include_from.add(reranker_component_name)

        try:
            results = pipeline.run(
                data=pipeline_inputs,
                include_outputs_from=include_from,
            )
        except Exception as e:
            return {
                "status": "error",
                "error": f"Pipeline execution failed: {str(e)}",
                "failure_type": "PIPELINE_EXECUTION_ERROR",
            }

    # 3. Extract retriever output
    retriever_output = results.get(retriever_component_name, {})
    retrieved_docs = retriever_output.get("documents", [])

    # 4. Extract reranker output
    reranked_docs = None
    reranker_output_note = None
    if reranker_component_name:
        reranker_raw = results.get(reranker_component_name, {})
        if reranker_raw:
            reranked_docs = reranker_raw.get("documents", [])
        else:
            # Reranker detected but output not captured — surface explicit warning.
            # No silent fallback to RANKING_FAILURE (decision #2).
            reranker_output_note = (
                f"Reranker '{reranker_component_name}' was detected in the pipeline but its "
                "intermediate output was not captured in pipeline_outputs. "
                "CONTEXT_LOSS detection is unavailable. "
                "Ensure include_outputs_from includes the reranker component name."
            )

    # 5. Find generator output
    generator_output = {}
    generator_name = discovered_generator

    if generator_name and generator_name in results:
        generator_output = results[generator_name]
    else:
        for comp_name, comp_out in results.items():
            if isinstance(comp_out, dict) and "replies" in comp_out:
                generator_name = comp_name
                generator_output = comp_out
                break

    generated_replies = generator_output.get("replies", [])
    generated_answer = generated_replies[0] if generated_replies else None

    # 6. Failure classification checks
    triggered_checks = {
        "no_results": False,
        "empty_context": False,
        "generator_failure": False,
        "ranking_failure": False,
    }

    # Check 1: No Results
    if not retrieved_docs:
        triggered_checks["no_results"] = True

    # Check 2: Ranking / Score / Context failure
    #
    # With relevant_doc_id:
    #   - Expected doc absent from retriever output → SCORE_BELOW_CUTOFF
    #   - Expected doc in retriever but absent from reranker output → CONTEXT_LOSS
    #   - Score below threshold (no doc-level issue) → SCORE_BELOW_CUTOFF
    #
    # Without relevant_doc_id (legacy):
    #   - Score below threshold → RANKING_FAILURE (backwards-compat)
    top_score = None
    ranking_failure_subtype = None  # "RANKING_FAILURE" | "SCORE_BELOW_CUTOFF" | "CONTEXT_LOSS"

    if not triggered_checks["no_results"] and retrieved_docs:
        docs_with_scores = [doc for doc in retrieved_docs if doc.score is not None]
        if docs_with_scores:
            top_doc = max(docs_with_scores, key=lambda d: d.score)
            top_score = top_doc.score

        if relevant_doc_id is not None:
            retrieved_ids = {doc.id for doc in retrieved_docs if doc.id is not None}

            if relevant_doc_id not in retrieved_ids:
                # Expected doc was never retrieved
                triggered_checks["ranking_failure"] = True
                ranking_failure_subtype = "SCORE_BELOW_CUTOFF"
            elif reranker_component_name and reranked_docs is not None:
                # Expected doc was retrieved — check if reranker dropped it
                reranked_ids = {doc.id for doc in reranked_docs if doc.id is not None}
                if relevant_doc_id not in reranked_ids:
                    triggered_checks["ranking_failure"] = True
                    ranking_failure_subtype = "CONTEXT_LOSS"
                elif top_score is not None and top_score < ranking_threshold:
                    triggered_checks["ranking_failure"] = True
                    ranking_failure_subtype = "SCORE_BELOW_CUTOFF"
            elif top_score is not None and top_score < ranking_threshold:
                triggered_checks["ranking_failure"] = True
                ranking_failure_subtype = "SCORE_BELOW_CUTOFF"
        else:
            # Legacy path — no hint provided
            if top_score is not None and top_score < ranking_threshold:
                triggered_checks["ranking_failure"] = True
                ranking_failure_subtype = "RANKING_FAILURE"

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

            if expected_answer:
                expected_lower = expected_answer.lower().strip()
                keywords = [w for w in re.findall(r"\b\w{4,}\b", expected_lower)]
                if keywords:
                    matches_count = sum(1 for kw in keywords if kw in answer_lower)
                    match_ratio = matches_count / len(keywords)
                    if match_ratio < 0.2:
                        matches_expected = False
                        triggered_checks["generator_failure"] = True
                else:
                    if expected_lower not in answer_lower:
                        matches_expected = False
                        triggered_checks["generator_failure"] = True
        else:
            triggered_checks["generator_failure"] = True

    # 7. Primary failure classification
    primary_failure = "SUCCESS"
    if triggered_checks["no_results"]:
        primary_failure = "NO_RESULTS"
    elif triggered_checks["ranking_failure"]:
        primary_failure = ranking_failure_subtype  # RANKING_FAILURE | SCORE_BELOW_CUTOFF | CONTEXT_LOSS
    elif triggered_checks["empty_context"]:
        primary_failure = "EMPTY_CONTEXT"
    elif triggered_checks["generator_failure"]:
        primary_failure = "GENERATOR_FAILURE"

    # Assemble document diagnostics
    documents_diagnostics = []
    for idx, doc in enumerate(retrieved_docs):
        documents_diagnostics.append({
            "id": doc.id,
            "score": doc.score,
            "content_preview": doc.content[:150] + "..." if doc.content else None,
            "meta": _clean_metadata(doc.meta or {}),
            "rank": idx + 1,
        })

    # Assemble reranker diagnostics
    reranked_diagnostics = None
    if reranked_docs is not None:
        reranked_diagnostics = []
        for idx, doc in enumerate(reranked_docs):
            reranked_diagnostics.append({
                "id": doc.id,
                "score": doc.score,
                "content_preview": doc.content[:150] + "..." if doc.content else None,
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
            "reranker": {
                "component_name": reranker_component_name,
                "detected": reranker_component_name is not None,
                "reranked_documents": reranked_diagnostics,
                "note": reranker_output_note,
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
                "relevant_doc_id": relevant_doc_id,
                "failure_subtype": ranking_failure_subtype,
            },
            "triggered_checks": triggered_checks,
        },
    }
