import asyncio
import os
import sys
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load RAG Studio environment variables
dotenv_path = "/home/adityaraut/Documents/verba_haystack_rag_studio/rag-studio/backend/.env"
load_dotenv(dotenv_path)

# Add paths to sys.path to construct pipeline
sys.path.append("/home/adityaraut/Documents/verba_haystack_rag_studio/rag-studio/backend/app")
sys.path.append("/home/adityaraut/Documents/Haystack_Inspector")

import logging.handlers
# Monkey-patch RotatingFileHandler to avoid write permissions issues in Docker-owned logs dir
logging.handlers.RotatingFileHandler = lambda *args, **kwargs: logging.NullHandler()

from rag_studio_stack.rag_studio_lib.services.haystack_manager import HaystackManager
from haystack import Pipeline

# 1. Serialize the RAG Pipeline for inspection
pipeline_path = Path("/home/adityaraut/Documents/Haystack_Inspector/tests/rag_pipeline.yaml")
print(f"Generating serialized RAG pipeline at {pipeline_path}...")
manager = HaystackManager(weaviate_url="http://localhost:8080")

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", manager.text_embedder)
rag_pipeline.add_component("retriever", manager.retriever)
rag_pipeline.add_component("prompt_builder", manager.prompt_builder)
rag_pipeline.add_component("llm", manager.llm)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.prompt")

# Save serialized pipeline
rag_pipeline.dumps() # Just a check
with open(pipeline_path, "w", encoding="utf-8") as f:
    f.write(rag_pipeline.dumps())

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_tool_benchmark(session: ClientSession, tool_name: str, arguments: dict, call_id: int):
    start_time = time.perf_counter()
    try:
        # Call the tool through the MCP session
        result = await session.call_tool(tool_name, arguments)
        latency = time.perf_counter() - start_time
        # Check if result indicates failure
        is_error = "Error" in result.content[0].text if result.content else False
        return {
            "call_id": call_id,
            "latency": latency,
            "success": not is_error,
            "error_msg": result.content[0].text if is_error else None
        }
    except Exception as e:
        latency = time.perf_counter() - start_time
        return {
            "call_id": call_id,
            "latency": latency,
            "success": False,
            "error_msg": str(e)
        }

async def run_benchmark(concurrency_count: int = 15):
    # Setup server parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp/server.py"],
        env={
            **os.environ,
            "PYTHONPATH": ".:/home/adityaraut/Documents/verba_haystack_rag_studio/rag-studio/backend/app",
        }
    )

    print(f"\nStarting MCP Server and executing {concurrency_count} concurrent requests...")
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize session
            await session.initialize()
            
            # Fire concurrent calls
            tasks = []
            for i in range(concurrency_count):
                tasks.append(
                    call_tool_benchmark(
                        session=session,
                        tool_name="inspect_pipeline_graph",
                        arguments={"pipeline_config_path": str(pipeline_path)},
                        call_id=i
                    )
                )
            
            start_total = time.perf_counter()
            results = await asyncio.gather(*tasks)
            total_duration = time.perf_counter() - start_total
            
            # Print performance metrics
            latencies = [r["latency"] for r in results]
            successes = [r for r in results if r["success"]]
            failures = [r for r in results if not r["success"]]
            
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
            
            print("=" * 60)
            print("             CONCURRENCY BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total concurrent tasks:  {concurrency_count}")
            print(f"Successful tasks:       {len(successes)}")
            print(f"Failed tasks:           {len(failures)}")
            print(f"Total duration:         {total_duration:.4f} seconds")
            print(f"Average latency:        {avg_latency:.4f} seconds")
            print(f"Min latency:            {min_latency:.4f} seconds")
            print(f"Max latency:            {max_latency:.4f} seconds")
            print("=" * 60)
            
            if failures:
                print("Failures details:")
                for f in failures:
                    print(f"  Task {f['call_id']}: {f['error_msg']}")
            else:
                print("All tasks completed successfully with no lock contention or failures!")

if __name__ == "__main__":
    asyncio.run(run_benchmark(15))
