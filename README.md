# Haystack Diagnostics Engine

A dedicated diagnostics and observability engine built specifically for **Haystack 2.x** RAG pipelines and document stores. 

It connects the dots between pipeline introspection, native tracing, and validation to solve ingestion-layer anomalies and retrieval failure modes before they compromise downstream applications.

---

## The Gap

While Haystack already offers robust pipeline graphing, OpenTelemetry tracing, and evaluator wrappers (like `RagasEvaluator`), there has been no unified tool to:
1. **Validate Document Store Health**: Catching silent ingestion bugs (such as `content=None` which crashes downstream language classifiers or prompt builders) before queries run.
2. **Automate Retrieval Failure Classification**: Systematically categorizing *why* a query returned a wrong answer or low-quality documents.
3. **Unify the Debugging Workflow**: Bringing Inspect → Validate → Diagnose into a single library and a lightweight Model Context Protocol (MCP) server.

---

## The Core Tools

### 1. `validate_document_store(document_store, ...)`
Evaluates the health of documents written to a Haystack document store. It checks for:
- `content=None` (blob-only documents causing pipeline crashes)
- Empty documents (`content=""`)
- Very short chunks (character count below threshold)
- Duplicate chunks (detected using MD5 hash)
- Missing metadata fields needed by pipeline filters
- Null embeddings
- Embedding dimension mismatch (both store-wide consistency and against an expected dimension)

### 2. `inspect_pipeline(pipeline)`
Inspects a live Haystack pipeline and returns its structural metadata:
- All components and their class paths
- Input and output sockets (including types, mandatory/optional status, and default values)
- Component-to-component connections
- A native **Mermaid.js** graph representation of the pipeline topology (drawn using Haystack's native layout engine)

### 3. `diagnose_retrieval_failure(pipeline, query, ...)`
Classifies query-level retrieval and pipeline failures by running the pipeline and extracting intermediate retriever outputs. It classifies failures in the following sequence:
1. **No Results**: The retriever returned 0 documents.
2. **Ranking Failure**: The retriever/ranker returned documents, but the top result's score (extracted from Haystack's `Document.score` attribute) was below the `ranking_threshold`.
3. **Empty Context**: Documents were retrieved, but their combined content is empty or contains no usable text.
4. **Generator Failure**: Relevant context was supplied, but the generator returned an empty response, an LLM refusal (e.g., "I don't know"), or did not match the `expected_answer` (if provided).

---

## Repository Structure

```
haystack-diagnostics/
│
├── diagnostics/
│   ├── __init__.py
│   ├── document_validator.py
│   ├── pipeline_inspector.py
│   └── failure_diagnoser.py
│
├── mcp/
│   └── server.py                  # MCP server wrapper (<200 lines)
│
├── demo/
│   └── sample_pipeline.py         # Out-of-the-box local demo run
│
├── tests/                         # Unit tests
│   ├── test_document_validator.py
│   ├── test_pipeline_inspector.py
│   └── test_failure_diagnoser.py
│
├── pyproject.toml                 # Package metadata and build system setup
├── requirements.txt               # Pinned dependencies for environment replication
└── README.md
```

---

## Installation & Setup

1. **Clone the repository** and navigate to it:
   ```bash
   cd Haystack_Inspector
   ```

2. **Install the package and dependencies**:
   You can install the package in editable mode along with all core dependencies:
   ```bash
   pip install -e .
   ```
   
   To install with development dependencies (e.g. `pytest`):
   ```bash
   pip install -e ".[dev]"
   ```
   
   Or replicate the exact conda/pip environment using pinned versions:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Examples

### Programmatic Diagnostics

```python
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

# Import the diagnostic tools
from diagnostics import validate_document_store, inspect_pipeline, diagnose_retrieval_failure

# 1. Validate Document Store Ingestion Health
document_store = InMemoryDocumentStore()
# (Write documents to your store...)

report = validate_document_store(
    document_store=document_store,
    expected_metadata_keys=["source", "language"],
    expected_embedding_dim=1536,
    short_chunk_threshold=50
)
print("Store Health Report:", report["summary"])

# 2. Inspect Pipeline Structure
pipe = Pipeline()
# (Add components and connect them...)

structure = inspect_pipeline(pipe)
print("Mermaid graph:\n", structure["mermaid"])

# 3. Diagnose Retrieval Failures
diagnostics = diagnose_retrieval_failure(
    pipeline=pipe,
    query="What is the capital of France?",
    expected_answer="Paris",
    ranking_threshold=0.6
)
print("Primary Failure Type:", diagnostics["failure_type"])
print("Diagnostics Summary:", diagnostics["diagnostics"])
```

---

## Running the Demo

To run the local diagnostics demo, execute:
```bash
python demo/sample_pipeline.py
```
This script populates an in-memory document store with a mix of healthy and corrupted documents, builds a RAG pipeline, and demonstrates the output of the validator, inspector, and all four failure diagnoses cases (Success, No Results, Generator Refusal, and Ranking Failure) with zero external API dependencies.

---

## Running the Tests

The project includes a complete suite of unit tests verifying validator edge cases, Mermaid graph exports, and sequential RAG query classification. All tests are fully implemented, execute offline, and require zero external API keys:

- `tests/test_document_validator.py`: Verifies the 7 document store validation checks using mock documents.
- `tests/test_pipeline_inspector.py`: Validates component detail extraction, socket parsing, and Mermaid graph output.
- `tests/test_failure_diagnoser.py`: Verifies the sequential failure engine (`NO_RESULTS`, `RANKING_FAILURE`, `EMPTY_CONTEXT`, `GENERATOR_FAILURE`) using mock retrievers and generators.

To run the test suite:
```bash
pytest tests/
```

---

## Exposing as an MCP Server

The project includes an MCP server (`mcp/server.py`) that exposes the core diagnostics tools to LLM clients (like Claude Desktop or Cursor).

To start the MCP server:
```bash
mcp dev mcp/server.py
```

### Claude Desktop Integration
Add the following configuration to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "haystack-diagnostics": {
      "command": "python",
      "args": ["-m", "mcp.server", "/path/to/Haystack_Inspector/mcp/server.py"]
    }
  }
}
```
Once connected, Claude can automatically validate document store health, query pipeline graphs, and debug retrieval issues on your local workspace.
