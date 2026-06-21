from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.core.component import component

from diagnostics.pipeline_inspector import inspect_pipeline


@component
class SimpleGenerator:
    @component.output_types(reply=str)
    def run(self, prompt: str):
        return {"reply": f"Answered: {prompt}"}


def test_inspect_pipeline():
    pipe = Pipeline()
    pipe.add_component("prompt_builder", PromptBuilder(template="Query: {{query}}"))
    pipe.add_component("generator", SimpleGenerator())
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    report = inspect_pipeline(pipe)

    # Validate basic metadata
    assert "metadata" in report
    assert "mermaid" in report
    assert isinstance(report["mermaid"], str)
    assert "graph" in report["mermaid"] or "mermaid" in report["mermaid"]

    # Validate components
    assert "prompt_builder" in report["components"]
    assert "generator" in report["components"]

    pb_info = report["components"]["prompt_builder"]
    assert pb_info["class_name"] == "PromptBuilder"
    assert any(s["name"] == "query" for s in pb_info["input_sockets"])
    assert any(s["name"] == "prompt" for s in pb_info["output_sockets"])

    gen_info = report["components"]["generator"]
    assert gen_info["class_name"] == "SimpleGenerator"
    assert any(s["name"] == "prompt" for s in gen_info["input_sockets"])
    assert any(s["name"] == "reply" for s in gen_info["output_sockets"])

    # Validate connections
    assert len(report["connections"]) == 1
    connection = report["connections"][0]
    assert connection["sender"] == "prompt_builder"
    assert connection["sender_socket"] == "prompt"
    assert connection["receiver"] == "generator"
    assert connection["receiver_socket"] == "prompt"


@component
class MockEmbedder:
    @component.output_types(embedding=list)
    def run(self, text: str):
        return {"embedding": [0.1, 0.2]}


@component
class MockRetriever:
    @component.output_types(documents=list)
    def run(self, query_embedding: list):
        return {"documents": []}


@component
class MockLLM:
    @component.output_types(replies=list)
    def run(self, prompt: str):
        return {"replies": ["Hello"]}


def test_inspect_pipeline_four_components():
    pipe = Pipeline()
    pipe.add_component("embedder", MockEmbedder())
    pipe.add_component("retriever", MockRetriever())
    pipe.add_component("prompt_builder", PromptBuilder(template="Hello {{documents}}"))
    pipe.add_component("llm", MockLLM())
    
    pipe.connect("embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "llm.prompt")
    
    report = inspect_pipeline(pipe)
    
    assert "embedder" in report["components"]
    assert "retriever" in report["components"]
    assert "prompt_builder" in report["components"]
    assert "llm" in report["components"]
    
    assert report["components"]["embedder"]["class_name"] == "MockEmbedder"
    assert report["components"]["retriever"]["class_name"] == "MockRetriever"
    assert report["components"]["prompt_builder"]["class_name"] == "PromptBuilder"
    assert report["components"]["llm"]["class_name"] == "MockLLM"
    
    assert len(report["connections"]) == 3
    
    mermaid = report["mermaid"]
    assert "embedder" in mermaid
    assert "retriever" in mermaid
    assert "prompt_builder" in mermaid
    assert "llm" in mermaid


def test_inspect_pipeline_no_connections():
    pipe = Pipeline()
    pipe.add_component("embedder", MockEmbedder())
    pipe.add_component("llm", MockLLM())
    
    report = inspect_pipeline(pipe)
    assert len(report["connections"]) == 0
    assert "embedder" in report["components"]
    assert "llm" in report["components"]
    assert "embedder" in report["mermaid"]
    assert "llm" in report["mermaid"]

