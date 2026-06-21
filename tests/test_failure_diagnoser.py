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

