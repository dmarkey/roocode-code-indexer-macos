# roocode-code-indexer-macos

High-performance OpenAI-compatible embedding server for [Roo Code](https://docs.roocode.com/) codebase indexing on Apple Silicon.

Runs Qwen3 embedding models locally via MLX with optimized batched GPU inference — no API keys needed. **~5x faster than Ollama** for the same models.

## Quick Start

```bash
uvx --from git+https://github.com/dmarkey/roocode-code-indexer-macos roocode-code-indexer-macos
```

Or with the recommended 4B model:

```bash
uvx --from git+https://github.com/dmarkey/roocode-code-indexer-macos roocode-code-indexer-macos \
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ
```

## Roo Code Configuration

In Roo Code settings, set the embedding provider to OpenAI-compatible and point it at:

```
http://localhost:8000/v1/embeddings
```

## Available Models

| Model | Embedding Dim | Size | Description |
|-------|--------------|------|-------------|
| `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | 1024 | ~900MB | Fast, good for small codebases (default) |
| `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | 2560 | ~2.5GB | Recommended — best speed/quality tradeoff |
| `mlx-community/Qwen3-Embedding-8B-4bit-DWQ` | 4096 | ~4.5GB | Highest quality, slower |

Models are downloaded automatically from HuggingFace on first use.

## Benchmarks: MLX vs Ollama

Tested with real Roo Code indexing traffic: 10 concurrent requests of 60 texts each (541 texts total), simulating a typical codebase indexing burst.

### Cold cache (first indexing run)

| Server | Wall time | texts/sec | vs Ollama |
|--------|----------|-----------|-----------|
| Ollama qwen3-embedding:8b | 74.8s | 7 | baseline |
| Ollama qwen3-embedding:4b | 43.3s | 13 | 1.7x |
| **MLX Qwen3-Embedding-4B** | **9.1s** | **60** | **4.8x faster** |
| **MLX Qwen3-Embedding-8B** | **20.9s** | **26** | **3.6x faster** |

### Warm cache (re-indexing with overlapping files)

| Server | Wall time | texts/sec |
|--------|----------|-----------|
| MLX Qwen3-Embedding-4B | 90ms | 6,001 |
| MLX Qwen3-Embedding-8B | 147ms | 3,675 |

Ollama has no embedding cache — every re-index pays full inference cost.

### Why is MLX faster?

- **Request coalescing** — concurrent requests are merged into single GPU batches instead of queuing
- **Length-bucketed sub-batching** — minimizes padding waste across variable-length texts
- **Compiled Metal kernels** — `mx.compile` with fixed padding buckets for graph reuse
- **LRU embedding cache** — repeated texts skip inference entirely
- **Last-token pooling** — matches Qwen3-Embedding training objective

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | Default model |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_BATCH_SIZE` | `1024` | Max texts per batch |
| `MAX_TEXT_LENGTH` | `8192` | Max tokens per text |
| `LOG_LEVEL` | `INFO` | Logging level |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## License

MIT
