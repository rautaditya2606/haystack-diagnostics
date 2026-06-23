import pytest
from haystack import Document, Pipeline
from haystack.core.component import component

from diagnostics.failure_diagnoser import diagnose_retrieval_failure


@component
class MockRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else []

    @component.output_types(documents=list)
    def run(self, query: str):
        return {"documents": self.docs}


@component
class MockGenerator:
    def __init__(self, replies=None):
        self.replies = replies if replies is not None else []

    @component.output_types(replies=list)
    def run(self, prompt: list):
        return {"replies": self.replies}


def test_diagnoser_no_results():
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[]))
    pipe.add_component("generator", MockGenerator(replies=["Test"]))
    pipe.connect("retriever.documents", "generator.prompt")  # dummy connection

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "NO_RESULTS"
    assert report["diagnostics"]["triggered_checks"]["no_results"] is True


def test_diagnoser_empty_context():
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[Document(content="")]))
    pipe.add_component("generator", MockGenerator(replies=["Test"]))
    pipe.connect("retriever.documents", "generator.prompt")

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "EMPTY_CONTEXT"
    assert report["diagnostics"]["triggered_checks"]["empty_context"] is True


def test_diagnoser_generator_failure_refusal():
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[Document(content="Good context", score=0.9)]))
    pipe.add_component("generator", MockGenerator(replies=["I do not know the answer to this question."]))
    pipe.connect("retriever.documents", "generator.prompt")

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "GENERATOR_FAILURE"
    assert report["diagnostics"]["triggered_checks"]["generator_failure"] is True
    assert report["diagnostics"]["generator"]["is_refusal"] is True


def test_diagnoser_generator_failure_wrong_answer():
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[Document(content="Good context", score=0.9)]))
    pipe.add_component("generator", MockGenerator(replies=["The capital of France is Paris."]))
    pipe.connect("retriever.documents", "generator.prompt")

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
        expected_answer="London",
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "GENERATOR_FAILURE"
    assert report["diagnostics"]["triggered_checks"]["generator_failure"] is True
    assert report["diagnostics"]["generator"]["matches_expected"] is False


def test_diagnoser_ranking_failure():
    pipe = Pipeline()
    # Score 0.3 is below threshold 0.6
    pipe.add_component("retriever", MockRetriever(docs=[Document(content="Some context", score=0.3)]))
    pipe.add_component("generator", MockGenerator(replies=["Answer matching expected."]))
    pipe.connect("retriever.documents", "generator.prompt")

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
        expected_answer="expected",
        ranking_threshold=0.6,
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "RANKING_FAILURE"
    assert report["diagnostics"]["triggered_checks"]["ranking_failure"] is True


def test_diagnoser_success():
    pipe = Pipeline()
    pipe.add_component("retriever", MockRetriever(docs=[Document(content="Some context", score=0.8)]))
    pipe.add_component("generator", MockGenerator(replies=["The result is London."]))
    pipe.connect("retriever.documents", "generator.prompt")

    report = diagnose_retrieval_failure(
        pipeline=pipe,
        query="test query",
        retriever_component_name="retriever",
        pipeline_inputs={"retriever": {"query": "test query"}, "generator": {"prompt": "dummy"}},
        expected_answer="London",
        ranking_threshold=0.5,
    )

    assert report["status"] == "success"
    assert report["failure_type"] == "SUCCESS"
    assert report["diagnostics"]["triggered_checks"]["no_results"] is False
    assert report["diagnostics"]["triggered_checks"]["empty_context"] is False
    assert report["diagnostics"]["triggered_checks"]["generator_failure"] is False
    assert report["diagnostics"]["triggered_checks"]["ranking_failure"] is False


def test_diagnoser_magic_mock_no_results():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": []},
        "generator": {"replies": ["No results response"]}
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "NO_RESULTS"


def test_diagnoser_magic_mock_ranking_failure():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="Good context", score=0.2)]},
        "generator": {"replies": ["Low confidence response"]}
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "RANKING_FAILURE"


def test_diagnoser_magic_mock_empty_context():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="", score=0.8)]},
        "generator": {"replies": ["Some response"]}
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "EMPTY_CONTEXT"


def test_diagnoser_magic_mock_generator_failure():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="Good context", score=0.8)]},
        "generator": {"replies": ["I do not know the answer"]}
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "GENERATOR_FAILURE"


def test_diagnoser_magic_mock_success_without_expected():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="Valid content", score=0.8)]},
        "generator": {"replies": ["Here is the correct answer"]}
    }
    # No expected_answer provided
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "SUCCESS"


def test_diagnoser_custom_retriever_name():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("my_custom_retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    mock_pipeline.run.return_value = {
        "my_custom_retriever": {"documents": []},
        "generator": {"replies": ["Response"]}
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="my_custom_retriever",
        ranking_threshold=0.5
    )
    assert report["failure_type"] == "NO_RESULTS"


def test_diagnoser_with_pipeline_outputs():
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()})
    ])
    
    # We will pass pre-computed outputs, so mock_pipeline.run should NOT be called.
    def should_not_run(*args, **kwargs):
        raise AssertionError("pipeline.run should not be called when pipeline_outputs is provided")
    mock_pipeline.run = should_not_run
    
    pre_computed_outputs = {
        "retriever": {"documents": []},
        "generator": {"replies": ["I do not know the answer"]}
    }
    
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test query",
        retriever_component_name="retriever",
        pipeline_outputs=pre_computed_outputs,
        ranking_threshold=0.5
    )

    assert report["failure_type"] == "NO_RESULTS"



# ---------------------------------------------------------------------------
# Taxonomy split tests: RANKING_FAILURE / SCORE_BELOW_CUTOFF / CONTEXT_LOSS
# ---------------------------------------------------------------------------

def test_ranking_failure_backwards_compat():
    """Without relevant_doc_id, low score must still produce legacy RANKING_FAILURE."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()}),
    ])
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [Document(content="Some context", score=0.2)]},
        "generator": {"replies": ["Some answer"]},
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test",
        retriever_component_name="retriever",
        ranking_threshold=0.5,
        relevant_doc_id=None,  # no hint — legacy path
    )
    assert report["failure_type"] == "RANKING_FAILURE"
    assert report["diagnostics"]["ranking"]["failure_subtype"] == "RANKING_FAILURE"


def test_score_below_cutoff_with_hint():
    """With relevant_doc_id not in retrieved docs, failure must be SCORE_BELOW_CUTOFF."""
    from unittest.mock import MagicMock
    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()}),
    ])
    retrieved_doc = Document(content="Wrong document", score=0.9, id="doc-wrong")
    mock_pipeline.run.return_value = {
        "retriever": {"documents": [retrieved_doc]},
        "generator": {"replies": ["Some answer"]},
    }
    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test",
        retriever_component_name="retriever",
        ranking_threshold=0.5,
        relevant_doc_id="doc-expected",  # not in retrieved docs
    )
    assert report["failure_type"] == "SCORE_BELOW_CUTOFF"
    assert report["diagnostics"]["ranking"]["failure_subtype"] == "SCORE_BELOW_CUTOFF"


def test_context_loss_with_hint():
    """CONTEXT_LOSS fires when relevant_doc_id is in retriever output but absent from reranker output."""
    from unittest.mock import MagicMock

    expected_doc = Document(content="Expected document.", score=0.9, id="doc-expected")
    other_doc = Document(content="Other document.", score=0.7, id="doc-other")

    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("ranker", {"instance": MagicMock()}),
        ("generator", {"instance": MagicMock()}),
    ])

    # Pre-computed: retriever found both; ranker dropped doc-expected
    pre_computed = {
        "retriever": {"documents": [expected_doc, other_doc]},
        "ranker": {"documents": [other_doc]},
        "generator": {"replies": ["Partial answer."]},
    }

    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="context loss test",
        retriever_component_name="retriever",
        reranker_component_name="ranker",
        relevant_doc_id="doc-expected",
        pipeline_outputs=pre_computed,
    )
    assert report["failure_type"] == "CONTEXT_LOSS"
    assert report["diagnostics"]["ranking"]["failure_subtype"] == "CONTEXT_LOSS"


def test_reranker_detected_but_not_captured_raises():
    """
    Decision #2: If a reranker is detected in the pipeline but its output is missing
    from pre-computed pipeline_outputs, the return dict must include an explicit warning
    note — no silent fallback to RANKING_FAILURE.
    """
    from unittest.mock import MagicMock

    ranker_instance = MagicMock()
    ranker_instance.__class__.__name__ = "SentenceTransformersRanker"

    mock_pipeline = MagicMock()
    mock_pipeline.graph.nodes = MagicMock(return_value=[
        ("retriever", {"instance": MagicMock()}),
        ("ranker", {"instance": ranker_instance}),
        ("generator", {"instance": MagicMock()}),
    ])

    # pipeline_outputs deliberately omits the reranker key
    pre_computed_without_reranker = {
        "retriever": {"documents": [Document(content="Some context", score=0.9, id="doc-1")]},
        "generator": {"replies": ["Some answer"]},
    }

    report = diagnose_retrieval_failure(
        pipeline=mock_pipeline,
        query="test",
        retriever_component_name="retriever",
        reranker_component_name="ranker",
        relevant_doc_id="doc-expected",
        pipeline_outputs=pre_computed_without_reranker,
    )

    reranker_note = report["diagnostics"]["reranker"]["note"]
    assert reranker_note is not None, "Expected a warning note for uncaptured reranker output."
    assert "ranker" in reranker_note.lower() or "reranker" in reranker_note.lower()
    # Must NOT silently classify as RANKING_FAILURE
    assert report["failure_type"] != "RANKING_FAILURE", (
        "Should not silently fall back to RANKING_FAILURE when reranker output is missing."
    )
