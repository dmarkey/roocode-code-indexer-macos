# roocode-code-indexer-macos

High-performance OpenAI-compatible embedding server for [Roo Code](https://docs.roocode.com/) codebase indexing on Apple Silicon.

Runs Qwen3 embedding models locally via MLX with optimized batched GPU inference — no API keys needed.

## Quick Start

```bash
uvx --from git+https://github.com/dmarkey/roocode-code-indexer-macos roocode-indexer
```

Or with options:

```bash
uvx --from git+https://github.com/dmarkey/roocode-code-indexer-macos roocode-indexer --model medium --port 8000
```

## Roo Code Configuration

In Roo Code settings, set the embedding provider to OpenAI-compatible and point it at:

```
http://localhost:8000/v1/embeddings
```

## Available Models

| Alias | Model | Dim | Description |
|-------|-------|-----|-------------|
| `small` | Qwen3-Embedding-0.6B-4bit | 1024 | Fast, good for small codebases |
| `medium` | Qwen3-Embedding-4B-4bit | 2560 | Recommended — best speed/quality tradeoff |
| `large` | Qwen3-Embedding-8B-4bit | 4096 | Highest quality, slower |

Default model is `small`. Use `--model medium` for the recommended balance.

## Performance

Optimized for the burst-request pattern Roo Code uses during indexing:

- **Request coalescing** — concurrent requests are merged into single GPU batches
- **Length-bucketed sub-batching** — minimizes padding waste across variable-length texts
- **Compiled Metal kernels** — `mx.compile` with fixed padding buckets for graph reuse
- **LRU embedding cache** — repeated texts skip inference entirely
- **Last-token pooling** — matches Qwen3-Embedding training objective

Typical performance on Apple Silicon (M-series):

| Model | Single request (60 texts) | 4 requests coalesced |
|-------|--------------------------|---------------------|
| small (0.6B) | ~200ms | ~400ms |
| medium (4B) | ~2s | ~4-5s |
| large (8B) | ~3s | ~8-9s |

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
- Models are downloaded automatically from HuggingFace on first use

## License

MIT
