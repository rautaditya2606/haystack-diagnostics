import os
import sys
import pathlib

# Append project root to sys.path to allow importing from diagnostics module directly
sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

import json
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.core.component import component

from diagnostics.document_validator import validate_document_store
from diagnostics.pipeline_inspector import inspect_pipeline
from diagnostics.failure_diagnoser import diagnose_retrieval_failure


@component
class SimpleDemoGenerator:
    """
    A lightweight, local generator component for demo and testing.
    Responds deterministically based on keyword matching.
    """
    @component.output_types(replies=list)
    def run(self, prompt: str):
        # Normalize prompt for simple keyword matching
        prompt_lower = prompt.lower()
        
        # Extract the question part from the template structure
        import re
        question_match = re.search(r"question:\s*(.*)", prompt_lower)
        if question_match:
            question = question_match.group(1).strip()
        else:
            question = prompt_lower
        
        if "paris" in question:
            return {"replies": ["The capital of France is Paris."]}
        elif "berlin" in question:
            return {"replies": ["The capital of Germany is Berlin."]}
        elif "rome" in question:
            return {"replies": ["I'm sorry, I do not know the answer based on the provided context."]}
        else:
            return {"replies": ["I cannot find any information about that topic in the context."]}


def main():
    print("=" * 60)
    print("             HAYSTACK DIAGNOSTICS DEMO RUN")
    print("=" * 60)

    # 1. Setup Document Store and populate with healthy and buggy data
    print("\n[Step 1] Initializing Document Store and Indexing Documents...")
    document_store = InMemoryDocumentStore()
    
    docs = [
        # Healthy document 1
        Document(
            content="Paris is the capital of France. It is known for the Eiffel Tower, art museums, and fine dining.",
            meta={"source": "wiki", "language": "en"}
        ),
        # Healthy document 2
        Document(
            content="Berlin is the capital of Germany. It is known for its art scene and modern landmarks like the Brandenburg Gate.",
            meta={"source": "wiki", "language": "en"}
        ),
        # Healthy document 3 (for generator refusal testing)
        Document(
            content="Rome is a historic city in Italy, famous for the Colosseum.",
            meta={"source": "wiki", "language": "en"}
        ),
        # Buggy document 1: Content is None
        Document(
            content=None,
            meta={"source": "corrupted_pdf", "language": "en"}
        ),
        # Buggy document 2: Empty content
        Document(
            content="",
            meta={"source": "empty_file", "language": "en"}
        ),
        # Buggy document 3: Very short chunk
        Document(
            content="short",
            meta={"source": "snippet", "language": "en"}
        ),
        # Buggy document 4: Duplicate of Document 2
        Document(
            content="Berlin is the capital of Germany. It is known for its art scene and modern landmarks like the Brandenburg Gate.",
            meta={"source": "wiki_backup", "language": "en"}
        ),
    ]
    
    # We index without embeddings first to showcase general validation
    document_store.write_documents(docs)
    print(f"Indexed {len(docs)} documents.")

    # 2. Run Document Store Validation
    print("\n[Step 2] Running validate_document_store()...")
    validation_report = validate_document_store(
        document_store=document_store,
        expected_metadata_keys=["source", "language"],
        short_chunk_threshold=20
    )
    print(json.dumps(validation_report, indent=2))

    # 3. Build RAG Pipeline
    print("\n[Step 3] Constructing RAG Pipeline...")
    retriever = InMemoryBM25Retriever(document_store=document_store)
    
    # Simple Jinja2 prompt template
    template = """
    Context:
    {% for doc in documents %}
      - {{ doc.content }}
    {% endfor %}
    
    Question: {{ query }}
    Answer:
    """
    prompt_builder = PromptBuilder(template=template, required_variables=["documents", "query"])
    generator = SimpleDemoGenerator()

    pipeline = Pipeline()
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)

    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    print("Pipeline constructed and connected successfully.")

    # 4. Run Pipeline Inspection
    print("\n[Step 4] Running inspect_pipeline()...")
    inspection_report = inspect_pipeline(pipeline)
    
    # Print a brief summary of inspection (excluding full sockets for brevity)
    print("Pipeline Metadata:", inspection_report["metadata"])
    print("Pipeline Components found:", list(inspection_report["components"].keys()))
    print("Pipeline Connections count:", len(inspection_report["connections"]))
    print("\nGenerated Mermaid Diagram:")
    print("-" * 60)
    print(inspection_report["mermaid"])
    print("-" * 60)

    # 5. Run Failure Diagnostics
    print("\n[Step 5] Running failure diagnostics for different query cases...")

    # Case A: Success Query
    print("\n--- CASE A: SUCCESS QUERY (Query: 'Paris') ---")
    diag_success = diagnose_retrieval_failure(
        pipeline=pipeline,
        query="Paris",
        expected_answer="Paris",
        ranking_threshold=0.1
    )
    print(f"Detected Failure Type: {diag_success['failure_type']}")
    print(f"Answer: {diag_success['diagnostics']['generator']['generated_answer']}")
    print("Triggered Checks:", diag_success['diagnostics']['triggered_checks'])

    # Case B: No Results Failure (Query: 'Tokyo' - not indexed)
    print("\n--- CASE B: NO RESULTS FAILURE (Query: 'Tokyo') ---")
    diag_no_results = diagnose_retrieval_failure(
        pipeline=pipeline,
        query="Tokyo",
        ranking_threshold=0.1
    )
    print(f"Detected Failure Type: {diag_no_results['failure_type']}")
    print("Triggered Checks:", diag_no_results['diagnostics']['triggered_checks'])

    # Case C: Generator Failure (Query: 'Rome' - causes refusal reply)
    print("\n--- CASE C: GENERATOR FAILURE - REFUSAL (Query: 'Rome') ---")
    diag_generator_fail = diagnose_retrieval_failure(
        pipeline=pipeline,
        query="Rome",
        ranking_threshold=0.1
    )
    print(f"Detected Failure Type: {diag_generator_fail['failure_type']}")
    print(f"Answer: {diag_generator_fail['diagnostics']['generator']['generated_answer']}")
    print("Triggered Checks:", diag_generator_fail['diagnostics']['triggered_checks'])

    # Case D: Ranking Failure (Query: 'Berlin' but with high threshold)
    print("\n--- CASE D: RANKING FAILURE (Query: 'Berlin', Threshold: 5.0) ---")
    diag_ranking_fail = diagnose_retrieval_failure(
        pipeline=pipeline,
        query="Berlin",
        ranking_threshold=5.0  # BM25 scores are generally low for short documents
    )
    print(f"Detected Failure Type: {diag_ranking_fail['failure_type']}")
    print(f"Top Document Score: {diag_ranking_fail['diagnostics']['retriever']['top_score']}")
    print("Triggered Checks:", diag_ranking_fail['diagnostics']['triggered_checks'])

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
