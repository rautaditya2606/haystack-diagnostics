"""
Smoke test for MCP collect_debug_bundle_tool.

Exercises:
  1. collect_debug_bundle_tool with pipeline_config_content (inline YAML)
  2. collect_debug_bundle_tool with pipeline_config_path (file path)
  3. Confirms _load_pipeline() content > path precedence
  4. Confirms error when neither is provided

Runs fully offline with InMemoryBM25Retriever + InMemoryDocumentStore.
No API keys required.
"""

import json
import sys
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Import the MCP tool directly (not via the MCP protocol)
import importlib.util

_server_path = Path(__file__).parent.parent / "mcp" / "server.py"
_spec = importlib.util.spec_from_file_location("local_mcp_server", _server_path)
_mcp_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcp_server)
collect_debug_bundle_tool = _mcp_server.collect_debug_bundle_tool



# ---------------------------------------------------------------------------
# Build a serializable pipeline with documents already in the store
# ---------------------------------------------------------------------------

def build_pipeline_yaml() -> str:
    store = InMemoryDocumentStore()
    store.write_documents([
        Document(content="The Eiffel Tower is in Paris, France.", id="doc-eiffel"),
        Document(content="The Colosseum is located in Rome, Italy.", id="doc-colosseum"),
        Document(content="Big Ben is a famous clock tower in London.", id="doc-bigben"),
    ])

    pipe = Pipeline()
    pipe.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    return pipe.dumps()  # YAML string


PIPELINE_YAML = build_pipeline_yaml()
QUERY = "Where is the Eiffel Tower?"

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
failures = []


def check(label: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
        failures.append(label)


# ---------------------------------------------------------------------------
# Check 1: pipeline_config_content (inline YAML)
# ---------------------------------------------------------------------------
print("\n[1] collect_debug_bundle_tool — pipeline_config_content (inline YAML)")
result_str = collect_debug_bundle_tool(
    query=QUERY,
    pipeline_config_content=PIPELINE_YAML,
    retriever_component_name="retriever",
    output_dir="/tmp/mcp_smoke_bundles",
)

try:
    result = json.loads(result_str)
    check("Returns valid JSON", True)
    check("Has bundle_id", "bundle_id" in result)
    check("Has query", result.get("query") == QUERY)
    check("Has pipeline section", "pipeline" in result)
    check("Has retrieval section", "retrieval" in result)
    check("Retrieved docs > 0", result.get("retrieval", {}).get("doc_count", 0) > 0,
          f"got {result.get('retrieval', {}).get('doc_count')}")
    check("failure_type is SUCCESS or classified", "failure_type" in result)
    check("_bundle_path written", "_bundle_path" in result and Path(result["_bundle_path"]).exists(),
          result.get("_bundle_path"))
    print(f"  failure_type: {result.get('failure_type')}")
    print(f"  docs retrieved: {result.get('retrieval', {}).get('doc_count')}")
    print(f"  bundle written to: {result.get('_bundle_path')}")
except json.JSONDecodeError:
    print(f"  {FAIL}  Returns valid JSON: got non-JSON: {result_str[:200]}")
    failures.append("Returns valid JSON (content)")

# ---------------------------------------------------------------------------
# Check 2: pipeline_config_path (file path)
# ---------------------------------------------------------------------------
print("\n[2] collect_debug_bundle_tool — pipeline_config_path (file path)")
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(PIPELINE_YAML)
    yaml_path = f.name

result_str2 = collect_debug_bundle_tool(
    query=QUERY,
    pipeline_config_path=yaml_path,
    retriever_component_name="retriever",
    output_dir="/tmp/mcp_smoke_bundles",
)

try:
    result2 = json.loads(result_str2)
    check("Returns valid JSON", True)
    check("Has bundle_id", "bundle_id" in result2)
    check("Retrieved docs > 0", result2.get("retrieval", {}).get("doc_count", 0) > 0,
          f"got {result2.get('retrieval', {}).get('doc_count')}")
    check("_bundle_path written", "_bundle_path" in result2 and Path(result2["_bundle_path"]).exists())
    print(f"  failure_type: {result2.get('failure_type')}")
    print(f"  bundle written to: {result2.get('_bundle_path')}")
except json.JSONDecodeError:
    print(f"  {FAIL}  Returns valid JSON: got non-JSON: {result_str2[:200]}")
    failures.append("Returns valid JSON (path)")

# ---------------------------------------------------------------------------
# Check 3: content wins over path (precedence rule)
# ---------------------------------------------------------------------------
print("\n[3] Precedence: pipeline_config_content wins over pipeline_config_path")
result_str3 = collect_debug_bundle_tool(
    query=QUERY,
    pipeline_config_content=PIPELINE_YAML,   # valid content
    pipeline_config_path="/nonexistent/path/pipeline.yaml",  # bad path — should be ignored
    retriever_component_name="retriever",
    output_dir="/tmp/mcp_smoke_bundles",
)
try:
    result3 = json.loads(result_str3)
    check("Does not error on bad path when content is provided", "error" not in result_str3.lower()[:50])
    check("Returns bundle with content taking precedence", "bundle_id" in result3)
except json.JSONDecodeError:
    check("Does not error on bad path when content is provided", False, result_str3[:200])

# ---------------------------------------------------------------------------
# Check 4: error when neither pipeline param is provided
# ---------------------------------------------------------------------------
print("\n[4] Error when neither pipeline_config_content nor pipeline_config_path provided")
result_str4 = collect_debug_bundle_tool(
    query=QUERY,
    output_dir="/tmp/mcp_smoke_bundles",
)
check("Returns error string (not exception)", "error" in result_str4.lower() or "Error" in result_str4)
check("Does not contain bundle_id", "bundle_id" not in result_str4)
print(f"  error message: {result_str4[:120]}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*50}")
if failures:
    print(f"FAILED: {len(failures)} check(s):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED")
    sys.exit(0)
