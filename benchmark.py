#!/usr/bin/env python3
"""
Benchmark: MLX embedding server vs Ollama

Simulates Roo Code indexing pattern with synthetic code-like text data:
10 concurrent requests of 60 texts each, matching real-world indexing bursts.

Usage:
    python benchmark.py                          # benchmark all available servers
    python benchmark.py --mlx-only               # only benchmark MLX server
    python benchmark.py --ollama-only             # only benchmark Ollama
    python benchmark.py --requests 10 --texts 60  # customize load
"""

import asyncio
import random
import time
import statistics
import argparse

import httpx

# Synthetic code snippets that resemble real codebase content (file paths,
# function signatures, docstrings, imports) — the kind of text Roo Code
# sends during indexing.
CODE_FRAGMENTS = [
    "import os\nimport sys\nfrom pathlib import Path",
    "def calculate_embeddings(texts: list[str], model: str = 'default') -> np.ndarray:",
    "class DatabaseConnection:\n    def __init__(self, host: str, port: int = 5432):",
    "async def fetch_user_data(user_id: int) -> dict[str, Any]:\n    \"\"\"Fetch user profile and preferences from the API.\"\"\"",
    "# TODO: Refactor this module to use dependency injection instead of global state",
    "from fastapi import FastAPI, HTTPException, Depends\nfrom sqlalchemy.orm import Session",
    "def test_embedding_dimensions():\n    result = model.encode('hello world')\n    assert len(result) == 1024",
    "CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    email VARCHAR(255) UNIQUE NOT NULL,\n    created_at TIMESTAMP DEFAULT NOW()\n);",
    "export interface EmbeddingResponse {\n  data: Array<{ embedding: number[]; index: number }>;\n  model: string;\n  usage: { prompt_tokens: number };\n}",
    "const handleSubmit = async (e: React.FormEvent) => {\n  e.preventDefault();\n  const response = await fetch('/api/embed', { method: 'POST', body: JSON.stringify({ text }) });\n};",
    "func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {\n    var req EmbeddingRequest\n    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {",
    "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: embedding-server\nspec:\n  replicas: 3",
    "FROM python:3.12-slim\nRUN pip install mlx mlx-lm fastapi uvicorn\nCOPY . /app\nCMD [\"uvicorn\", \"server:app\"]",
    "name: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: macos-latest\n    steps:\n      - uses: actions/checkout@v4",
    "/**\n * Computes cosine similarity between two embedding vectors.\n * @param a First embedding vector\n * @param b Second embedding vector\n * @returns Similarity score between -1 and 1\n */",
    "model = load('mlx-community/Qwen3-Embedding-4B-4bit-DWQ')\ntokenizer = AutoTokenizer.from_pretrained(model_name)\nembeddings = model.encode(texts, batch_size=32)",
    "SELECT p.name, COUNT(e.id) as embedding_count\nFROM projects p\nJOIN embeddings e ON e.project_id = p.id\nGROUP BY p.name\nORDER BY embedding_count DESC\nLIMIT 10;",
    "error[E0308]: mismatched types\n  --> src/embedding.rs:42:5\n   |\n42 |     vec![0.0f32; dim]\n   |     ^^^^^^^^^^^^^^^^^ expected `Vec<f64>`, found `Vec<f32>`",
    "The embedding server uses MLX framework optimized for Apple Silicon. It provides an OpenAI-compatible /v1/embeddings endpoint that accepts batched text inputs and returns normalized embedding vectors.",
    "Performance optimization: by sorting texts by token length before batching, we minimize padding waste. A 60-text batch with lengths ranging from 10-200 tokens would otherwise pad everything to 200.",
]


def generate_payloads(num_requests: int, texts_per_request: int) -> list[list[str]]:
    """Generate synthetic code-like payloads matching Roo Code's indexing pattern."""
    rng = random.Random(42)  # Deterministic for reproducibility
    payloads = []
    for _ in range(num_requests):
        texts = []
        for _ in range(texts_per_request):
            # Pick a random fragment and optionally combine/truncate
            base = rng.choice(CODE_FRAGMENTS)
            # ~30% chance of combining two fragments (simulates longer files)
            if rng.random() < 0.3:
                base = base + "\n\n" + rng.choice(CODE_FRAGMENTS)
            texts.append(base)
        payloads.append(texts)
    return payloads


async def bench_single(client, url, model, texts, timeout=120.0):
    """Send a single embedding request and return (elapsed_ms, num_texts)."""
    body = {"input": texts, "model": model}
    t0 = time.time()
    resp = await client.post(url, json=body, timeout=timeout)
    elapsed = (time.time() - t0) * 1000
    resp.raise_for_status()
    return elapsed, len(texts)


async def bench_concurrent(url, model, payloads, label, runs=3):
    """Run all payloads concurrently and report stats."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Requests: {len(payloads)} x {len(payloads[0])} texts")
    print(f"{'='*60}")

    # Warmup
    async with httpx.AsyncClient() as client:
        print("  Warming up...", end=" ", flush=True)
        try:
            warmup_ms, _ = await bench_single(client, url, model, ["warmup test"])
            print(f"{warmup_ms:.0f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            return None

    results = []
    for run in range(runs):
        async with httpx.AsyncClient() as client:
            t_wall_start = time.time()
            tasks = [bench_single(client, url, model, p) for p in payloads]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            wall_time = (time.time() - t_wall_start) * 1000

            errors = [r for r in completed if isinstance(r, Exception)]
            successes = [r for r in completed if not isinstance(r, Exception)]

            if errors:
                print(f"  Run {run+1}: {len(errors)} errors: {errors[0]}")
                continue

            per_req = [ms for ms, _ in successes]
            total_texts = sum(n for _, n in successes)

            results.append({
                "wall_ms": wall_time,
                "per_req_ms": per_req,
                "total_texts": total_texts,
            })

            print(f"  Run {run+1}: wall={wall_time:.0f}ms, "
                  f"avg_req={statistics.mean(per_req):.0f}ms, "
                  f"min={min(per_req):.0f}ms, max={max(per_req):.0f}ms, "
                  f"texts/sec={total_texts / (wall_time / 1000):.0f}")

    if not results:
        return None

    # Use first run (cold cache) for fair comparison
    first = results[0]
    return {
        "label": label,
        "wall_ms": first["wall_ms"],
        "avg_req_ms": statistics.mean(first["per_req_ms"]),
        "texts_per_sec": first["total_texts"] / (first["wall_ms"] / 1000),
        "total_texts": first["total_texts"],
    }


async def main():
    parser = argparse.ArgumentParser(description="Benchmark MLX vs Ollama embedding servers")
    parser.add_argument("--requests", type=int, default=10, help="Number of concurrent requests (default: 10)")
    parser.add_argument("--texts", type=int, default=60, help="Texts per request (default: 60)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per config (default: 3)")
    parser.add_argument("--mlx-only", action="store_true", help="Only benchmark MLX server")
    parser.add_argument("--ollama-only", action="store_true", help="Only benchmark Ollama")
    parser.add_argument("--mlx-url", default="http://localhost:8000", help="MLX server URL")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    args = parser.parse_args()

    payloads = generate_payloads(args.requests, args.texts)
    total_texts = sum(len(p) for p in payloads)
    print(f"Generated {len(payloads)} requests x {args.texts} texts = {total_texts} total texts")

    configs = []
    if not args.mlx_only:
        configs.extend([
            (f"{args.ollama_url}/v1/embeddings", "qwen3-embedding:0.6b", "Ollama qwen3-embedding:0.6b"),
            (f"{args.ollama_url}/v1/embeddings", "qwen3-embedding:4b", "Ollama qwen3-embedding:4b"),
            (f"{args.ollama_url}/v1/embeddings", "qwen3-embedding:8b", "Ollama qwen3-embedding:8b"),
        ])
    if not args.ollama_only:
        configs.extend([
            (f"{args.mlx_url}/v1/embeddings", "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", "MLX Qwen3-Embedding-0.6B"),
            (f"{args.mlx_url}/v1/embeddings", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ", "MLX Qwen3-Embedding-4B"),
            (f"{args.mlx_url}/v1/embeddings", "mlx-community/Qwen3-Embedding-8B-4bit-DWQ", "MLX Qwen3-Embedding-8B"),
        ])

    results = []
    for url, model, label in configs:
        try:
            r = await bench_concurrent(url, model, payloads, label, runs=args.runs)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  SKIPPED ({label}): {e}")

    if results:
        print(f"\n{'='*60}")
        print(f"  SUMMARY (run 1 / cold cache, {args.requests} concurrent requests)")
        print(f"{'='*60}")
        print(f"{'Server':<35} {'Wall(ms)':>8} {'Avg/req':>8} {'texts/s':>8}")
        print("-" * 65)
        for r in results:
            print(f"{r['label']:<35} {r['wall_ms']:>8.0f} {r['avg_req_ms']:>8.0f} {r['texts_per_sec']:>8.0f}")


if __name__ == "__main__":
    asyncio.run(main())
