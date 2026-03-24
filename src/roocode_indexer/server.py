#!/usr/bin/env python3
"""
Roo Code Indexer — MLX Embedding Server for macOS

A high-performance OpenAI-compatible embedding server for Roo Code codebase
indexing on Apple Silicon. Runs Qwen3 embedding models via MLX with batched
inference, request coalescing, and compiled Metal kernels.

Usage:
    uvx --from git+https://github.com/dmarkey/roocode-code-indexer-macos roocode-indexer
    # or
    roocode-indexer --model medium --port 8000
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum

import numpy as np
import mlx.core as mx
from mlx_lm import load
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, field_validator

import uvicorn

# Constants
DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

# Available models and their embedding dimensions
AVAILABLE_MODELS = {
    "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ": {"embedding_dim": 1024},
    "mlx-community/Qwen3-Embedding-4B-4bit-DWQ": {"embedding_dim": 2560},
    "mlx-community/Qwen3-Embedding-8B-4bit-DWQ": {"embedding_dim": 4096},
}
MIN_BATCH_SIZE = 1
DEFAULT_MAX_BATCH = 1024  # Increased for stress testing
DEFAULT_MAX_LENGTH = 8192
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"

# Configure logging
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure application logging"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging(os.getenv("LOG_LEVEL", "INFO"))

# Configuration
@dataclass
class ServerConfig:
    """Server configuration"""
    model_name: str = os.getenv("MODEL_NAME", DEFAULT_MODEL)
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", str(DEFAULT_MAX_BATCH)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", str(DEFAULT_MAX_LENGTH)))
    port: int = int(os.getenv("PORT", str(DEFAULT_PORT)))
    host: str = os.getenv("HOST", DEFAULT_HOST)
    enable_cors: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    cors_origins: List[str] = None
    
    def __post_init__(self):
        """Validate configuration"""
        if self.cors_origins is None:
            self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        if self.max_batch_size < MIN_BATCH_SIZE:
            raise ValueError(f"max_batch_size must be at least {MIN_BATCH_SIZE}")
        if self.max_text_length < 1:
            raise ValueError("max_text_length must be positive")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")

# Load configuration
config = ServerConfig()

class ModelStatus(str, Enum):
    """Model status enumeration"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"

class ModelManager:
    """
    Manages MLX model loading, caching, and inference.
    
    This class handles the lifecycle of multiple embedding models,
    including loading, warming up, and generating embeddings.
    """
    
    # Fixed padding buckets — allows mx.compile to reuse compiled graphs
    PAD_BUCKETS = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192]

    # How long (seconds) to wait for more requests before running a coalesced batch.
    # Shorter = lower latency for single requests; longer = more coalescing opportunity.
    COALESCE_DEADLINE_S = 0.005  # 5ms

    def __init__(self, config: ServerConfig):
        self.config = config
        self.models: Dict[str, Tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
        self.model_status: Dict[str, ModelStatus] = {}  # model_name -> status
        self.model_load_times: Dict[str, float] = {}  # model_name -> load_time
        self._locks: Dict[str, asyncio.Lock] = {}  # model_name -> lock
        self._embedding_cache: OrderedDict[int, Tuple[np.ndarray, int]] = OrderedDict()  # hash -> (embedding, token_count)
        self._cache_max_size = 2000
        self._global_lock = asyncio.Lock()  # For managing model dict
        self.max_loaded_models = int(os.getenv("MAX_LOADED_MODELS", "2"))
        # Compiled inference functions per model (populated on first use)
        self._compiled_fns: Dict[str, Any] = {}
        # Cached causal masks by padded length
        self._causal_mask_cache: Dict[int, mx.array] = {}
        # Coalescing queue: list of (token_lists, model_name, normalize, future)
        self._coalesce_queue: asyncio.Queue = None  # initialized in start_coalescer
        self._coalescer_task: Optional[asyncio.Task] = None
        
    def _resolve_model_name(self, model_identifier: Optional[str] = None) -> str:
        """Resolve and validate model name."""
        if not model_identifier:
            return self.config.model_name

        if model_identifier in AVAILABLE_MODELS:
            return model_identifier

        raise ValueError(
            f"Unknown model: {model_identifier}. "
            f"Available: {list(AVAILABLE_MODELS.keys())}"
        )
    
    async def load_model(self, model_name: Optional[str] = None) -> str:
        """Load and initialize the specified embedding model
        
        Args:
            model_name: Full HuggingFace model name. If None, uses default.
            
        Returns:
            The resolved model name
        """
        model_name = self._resolve_model_name(model_name)
        
        # Check if already loaded
        if model_name in self.models and self.model_status.get(model_name) == ModelStatus.READY:
            return model_name
        
        # Get or create lock for this model
        async with self._global_lock:
            if model_name not in self._locks:
                self._locks[model_name] = asyncio.Lock()
        
        async with self._locks[model_name]:
            # Double-check after acquiring lock
            if model_name in self.models and self.model_status.get(model_name) == ModelStatus.READY:
                return model_name
            
            self.model_status[model_name] = ModelStatus.LOADING
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            try:
                # Check if we need to evict a model
                await self._manage_memory(model_name)
                
                # Load model and tokenizer
                model, tokenizer = load(model_name)
                
                # Validate model architecture
                if not hasattr(model, 'model'):
                    raise ValueError("Invalid model architecture: missing 'model' attribute")
                
                # Store the model
                self.models[model_name] = (model, tokenizer)
                
                # Warm up the model
                logger.info(f"Warming up model {model_name}...")
                await self._warmup(model_name)
                
                self.model_load_times[model_name] = time.time() - start_time
                self.model_status[model_name] = ModelStatus.READY
                logger.info(f"Model {model_name} loaded successfully in {self.model_load_times[model_name]:.2f}s")
                
                return model_name
                
            except Exception as e:
                self.model_status[model_name] = ModelStatus.ERROR
                logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
                raise RuntimeError(f"Model loading failed: {e}") from e
    
    async def _manage_memory(self, new_model: str) -> None:
        """Manage memory by evicting models if necessary"""
        if len(self.models) >= self.max_loaded_models:
            # Find least recently used model (simple strategy)
            # In production, you'd want proper LRU tracking
            models_to_evict = [m for m in self.models.keys() if m != new_model]
            if models_to_evict:
                evict_model = models_to_evict[0]  # Simple: evict first
                logger.info(f"Evicting model {evict_model} to make room for {new_model}")
                del self.models[evict_model]
                self.model_status[evict_model] = ModelStatus.UNLOADED
                self._compiled_fns.pop(evict_model, None)
                # Clear cache (hash keys don't allow per-model filtering;
                # model eviction is rare so clearing all is acceptable)
                self._embedding_cache.clear()
    
    async def _warmup(self, model_name: str) -> None:
        """Warm up model and pre-compile inference function for common bucket sizes."""
        try:
            model, tokenizer = self.models[model_name]
            compiled_fn = self._get_compiled_fn(model_name)

            # Warm up with a few bucket sizes to pre-compile Metal kernels
            for bucket in [16, 32, 64]:
                tokens = tokenizer.encode("warmup test text for compilation")
                padded_np = np.zeros((1, bucket), dtype=np.int32)
                padded_np[0, :len(tokens)] = tokens[:bucket]
                input_ids = mx.array(padded_np)
                hidden_states = compiled_fn(input_ids)
                pooled = mx.mean(hidden_states, axis=1)
                mx.eval(pooled)

            logger.info(f"Warmup complete: compiled fn + Metal kernels for {model_name}")

        except Exception as e:
            logger.warning(f"Warmup failed for {model_name} (non-critical): {e}")
    
    @staticmethod
    def _cache_key(model_name: str, text: str, normalize: bool) -> int:
        """Fast hash-based cache key instead of string concatenation."""
        return hash((model_name, text, normalize))

    def _cache_get(self, key: int) -> Optional[Tuple[np.ndarray, int]]:
        """LRU cache lookup — moves hit entry to end (most recent).
        Returns (embedding, token_count) or None."""
        if key in self._embedding_cache:
            self._embedding_cache.move_to_end(key)
            return self._embedding_cache[key]
        return None

    def _cache_put(self, key: int, value: np.ndarray, token_count: int) -> None:
        """LRU cache insert — evicts oldest entry if at capacity."""
        self._embedding_cache[key] = (value, token_count)
        self._embedding_cache.move_to_end(key)
        if len(self._embedding_cache) > self._cache_max_size:
            self._embedding_cache.popitem(last=False)

    @staticmethod
    def _pad_bucket(length: int) -> int:
        """Round up to the next fixed padding bucket size."""
        for b in ModelManager.PAD_BUCKETS:
            if length <= b:
                return b
        return length  # Fallback for very long sequences

    def _get_hidden_states(self, input_ids: mx.array, model: Any, mask: Optional[mx.array] = None) -> mx.array:
        """
        Extract hidden states from the model before output projection.

        Args:
            input_ids: Token IDs as MLX array [batch_size, seq_len]
            mask: Optional causal attention mask for padded batches

        Returns:
            Hidden states [batch_size, seq_len, hidden_dim]
        """
        # Get token embeddings
        h = model.model.embed_tokens(input_ids)

        # Pass through transformer layers
        for layer in model.model.layers:
            h = layer(h, mask=mask, cache=None)

        # Apply final layer normalization
        h = model.model.norm(h)

        return h

    def _get_compiled_fn(self, model_name: str):
        """Get or create a compiled inference function for a model."""
        if model_name not in self._compiled_fns:
            model, _ = self.models[model_name]

            def _forward(input_ids, mask=None):
                h = model.model.embed_tokens(input_ids)
                for layer in model.model.layers:
                    h = layer(h, mask=mask, cache=None)
                h = model.model.norm(h)
                return h

            self._compiled_fns[model_name] = mx.compile(_forward)
            logger.info(f"Created compiled inference function for {model_name}")
        return self._compiled_fns[model_name]

    # Max texts per sub-batch — keeps padding waste low by grouping similar lengths
    SUB_BATCH_SIZE = 16

    def _get_causal_mask(self, size: int) -> mx.array:
        """Get or create a cached causal (lower-triangular) mask of given size."""
        if size not in self._causal_mask_cache:
            self._causal_mask_cache[size] = mx.tril(mx.ones((size, size)))
        return self._causal_mask_cache[size]

    def _create_attention_mask(self, lengths: List[int], padded_len: int) -> mx.array:
        """Create a causal attention mask that also masks padding tokens.

        Fully vectorized — no Python loops. Returns mask of shape
        [batch, 1, padded_len, padded_len] in bfloat16 suitable for additive attention.
        """
        positions = mx.arange(padded_len)
        lengths_arr = mx.array(lengths)[:, None]
        pad_mask = (positions[None, :] < lengths_arr).astype(mx.float32)  # [B, S]

        causal = self._get_causal_mask(padded_len)
        combined = pad_mask[:, None, None, :] * causal[None, None, :, :]

        return (mx.array(1.0, dtype=mx.bfloat16) - combined.astype(mx.bfloat16)) * mx.array(-1e9, dtype=mx.bfloat16)

    def _run_sub_batch(self, token_lists: List[List[int]], model: Any, model_name: str) -> mx.array:
        """Run inference on a sub-batch of similar-length texts.

        Pads to fixed bucket sizes so mx.compile can reuse compiled graphs.
        Uses last-token pooling (takes the hidden state at the final real token),
        which matches the Qwen3-Embedding training objective.

        Returns pooled hidden states [N, hidden_dim] (lazy, not yet evaluated).
        """
        n = len(token_lists)
        lengths = [len(t) for t in token_lists]
        max_len = max(lengths)
        compiled_fn = self._get_compiled_fn(model_name)

        if n == 1:
            # Single text — pad to bucket for compiled fn reuse
            padded_len = self._pad_bucket(max_len)
            if padded_len == max_len:
                input_ids = mx.array([token_lists[0]])
            else:
                padded_np = np.zeros((1, padded_len), dtype=np.int32)
                padded_np[0, :max_len] = token_lists[0]
                input_ids = mx.array(padded_np)
            hidden_states = compiled_fn(input_ids)
            # Last-token pooling: take hidden state at final token position
            return hidden_states[:, max_len - 1, :].reshape(1, -1)

        padded_len = self._pad_bucket(max_len)

        padded_np = np.zeros((n, padded_len), dtype=np.int32)
        for i, t in enumerate(token_lists):
            padded_np[i, :len(t)] = t
        input_ids = mx.array(padded_np)

        mask = self._create_attention_mask(lengths, padded_len)
        hidden_states = compiled_fn(input_ids, mask)

        # Last-token pooling: extract hidden state at each text's final token
        last_indices = mx.array([l - 1 for l in lengths])
        return hidden_states[mx.arange(n), last_indices, :]

    # How many sub-batches to accumulate before calling mx.eval().
    # Smaller = GPU starts sooner (better pipelining), but more eval overhead.
    # 4 sub-batches × 16 texts = 64 texts per eval — matches the ~60 text
    # request size from roo-code, so single requests still get one eval.
    EVAL_EVERY_N_SUB_BATCHES = 4

    def _generate_batch(
        self,
        token_lists: List[List[int]],
        model: Any,
        model_name: str,
        normalize: bool,
    ) -> np.ndarray:
        """Generate embeddings for a list of pre-tokenized inputs.

        Sorts by length, then splits into sub-batches of similar-length texts.
        For large coalesced batches, evaluates incrementally every N sub-batches
        to let the GPU start work early instead of building one huge lazy graph.

        Returns np.ndarray of shape [len(token_lists), hidden_dim].
        """
        batch_size = len(token_lists)
        t0 = time.time()

        # Sort by length to group similar lengths together
        indexed = sorted(enumerate(token_lists), key=lambda x: len(x[1]))
        sort_order = [idx for idx, _ in indexed]
        sorted_tokens = [toks for _, toks in indexed]

        # Split into sub-batches of similar lengths, eval incrementally
        evaluated_chunks: List[np.ndarray] = []
        pending_sub_batches: List[mx.array] = []
        total_padding_tokens = 0
        total_compute_tokens = 0
        num_sub_batches = 0

        for sb_start in range(0, batch_size, self.SUB_BATCH_SIZE):
            sb_tokens = sorted_tokens[sb_start:sb_start + self.SUB_BATCH_SIZE]
            sb_lengths = [len(t) for t in sb_tokens]
            sb_padded = self._pad_bucket(max(sb_lengths))
            total_padding_tokens += sum(sb_padded - l for l in sb_lengths)
            total_compute_tokens += sb_padded * len(sb_tokens)
            num_sub_batches += 1

            pooled = self._run_sub_batch(sb_tokens, model, model_name)
            pending_sub_batches.append(pooled)

            # Evaluate incrementally for large batches
            if len(pending_sub_batches) >= self.EVAL_EVERY_N_SUB_BATCHES:
                chunk = mx.concatenate(pending_sub_batches, axis=0)
                chunk = chunk.astype(mx.float32)
                mx.eval(chunk)
                evaluated_chunks.append(np.asarray(chunk))
                pending_sub_batches = []

        # Evaluate remaining sub-batches
        if pending_sub_batches:
            if len(pending_sub_batches) == 1:
                chunk = pending_sub_batches[0]
            else:
                chunk = mx.concatenate(pending_sub_batches, axis=0)
            chunk = chunk.astype(mx.float32)
            mx.eval(chunk)
            evaluated_chunks.append(np.asarray(chunk))

        # Concatenate all evaluated chunks (numpy, already on CPU)
        if len(evaluated_chunks) == 1:
            sorted_result = evaluated_chunks[0]
        else:
            sorted_result = np.concatenate(evaluated_chunks, axis=0)

        # Unsort back to original order
        unsort_order = [0] * batch_size
        for new_pos, orig_pos in enumerate(sort_order):
            unsort_order[orig_pos] = new_pos
        result = sorted_result[unsort_order]

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            result = result / norms

        padding_pct = (total_padding_tokens / max(total_compute_tokens, 1)) * 100
        logger.info(
            f"[PERF-BATCH] {batch_size} texts in {num_sub_batches} sub-batches, "
            f"{len(evaluated_chunks)} evals, padding_waste={padding_pct:.0f}%, "
            f"total_ms={(time.time()-t0)*1000:.1f}"
        )
        return result

    # ── Request coalescing ──────────────────────────────────────────────
    # Instead of each HTTP request running GPU inference independently
    # (and queuing behind a mutex), requests submit tokenized work to a
    # shared queue.  A background task drains the queue, merges all
    # pending texts into one GPU batch, runs inference once, then fans
    # results back out via asyncio.Future.

    def start_coalescer(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the background coalescing task. Call once after event loop is running."""
        self._coalesce_queue = asyncio.Queue()
        self._coalescer_task = loop.create_task(self._coalescer_loop())
        logger.info("[COALESCE] Background coalescer started")

    async def _coalescer_loop(self) -> None:
        """Background loop: drain queue → merge → infer → distribute results."""
        while True:
            # Block until at least one item is available
            first = await self._coalesce_queue.get()
            pending = [first]

            # Wait briefly for more requests to arrive (coalescing window)
            deadline = time.time() + self.COALESCE_DEADLINE_S
            while time.time() < deadline:
                try:
                    item = self._coalesce_queue.get_nowait()
                    pending.append(item)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

            # Also grab anything that arrived during our sleep
            while not self._coalesce_queue.empty():
                try:
                    pending.append(self._coalesce_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            await self._process_coalesced(pending)

    # Maximum texts per coalesced GPU batch. Beyond this, throughput degrades
    # due to memory pressure. Excess requests spill to the next cycle.
    MAX_COALESCE_TEXTS = 256

    async def _process_coalesced(self, pending: list) -> None:
        """Merge pending requests by (model, normalize) key and run batched inference.

        Caps each GPU batch at MAX_COALESCE_TEXTS to stay in the throughput
        sweet spot. Excess requests are processed in subsequent batches
        within the same call, so early requests get results sooner.
        """
        from collections import defaultdict
        groups: Dict[Tuple[str, bool], list] = defaultdict(list)
        for item in pending:
            key = (item["model_name"], item["normalize"])
            groups[key].append(item)

        for (model_name, normalize), items in groups.items():
            # Pack whole requests into chunks that fit under MAX_COALESCE_TEXTS.
            # Never split a request across chunks — this ensures each chunk's
            # completion immediately resolves all its requests' futures.
            chunks: List[List[dict]] = []  # each chunk is a list of request items
            current_chunk: List[dict] = []
            current_size = 0

            for item in items:
                req_size = len(item["tokens"])
                if current_chunk and current_size + req_size > self.MAX_COALESCE_TEXTS:
                    # Start a new chunk
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                current_chunk.append(item)
                current_size += req_size

            if current_chunk:
                chunks.append(current_chunk)

            # Process each chunk
            for chunk_items in chunks:
                merged_tokens: List[List[int]] = []
                request_slices: List[Tuple[int, int, asyncio.Future]] = []

                for item in chunk_items:
                    start = len(merged_tokens)
                    merged_tokens.extend(item["tokens"])
                    end = len(merged_tokens)
                    request_slices.append((start, end, item["future"]))

                t0 = time.time()
                try:
                    model, _ = self.models[model_name]
                    chunk_embeddings = self._generate_batch(
                        merged_tokens, model, model_name, normalize
                    )
                    elapsed = (time.time() - t0) * 1000

                    logger.info(
                        f"[COALESCE] {len(chunk_items)} requests, "
                        f"{len(merged_tokens)} texts, inference_ms={elapsed:.1f}"
                    )

                    # Fan results back — each request is fully within this chunk
                    for start, end, future in request_slices:
                        if not future.cancelled():
                            future.set_result(chunk_embeddings[start:end])

                except Exception as e:
                    logger.error(f"[COALESCE] Batch inference failed: {e}", exc_info=True)
                    for _, _, future in request_slices:
                        if not future.done():
                            future.set_exception(e)

    async def generate_embeddings_coalesced(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, str, int, int]:
        """Like generate_embeddings, but submits to the coalescing queue."""
        t0 = time.time()
        model_name = await self.load_model(model_name)

        if self.model_status.get(model_name) != ModelStatus.READY:
            raise RuntimeError(f"Model {model_name} not ready")

        if not texts:
            embedding_dim = AVAILABLE_MODELS[model_name]["embedding_dim"]
            return np.array([]), model_name, embedding_dim, 0

        model, tokenizer = self.models[model_name]
        embedding_dim = AVAILABLE_MODELS[model_name]["embedding_dim"]

        # Tokenize and check cache (this is CPU work, safe to do concurrently)
        results: Dict[int, np.ndarray] = {}
        uncached_indices: List[int] = []
        uncached_tokens: List[List[int]] = []
        uncached_texts: List[str] = []
        total_tokens = 0

        t_tok_start = time.time()
        for i, text in enumerate(texts):
            cache_key = self._cache_key(model_name, text, normalize)
            cached = self._cache_get(cache_key)
            if cached is not None:
                emb, tok_count = cached
                results[i] = emb
                total_tokens += tok_count
            else:
                tokens = tokenizer.encode(text)
                if len(tokens) > self.config.max_text_length:
                    tokens = tokens[:self.config.max_text_length]
                total_tokens += len(tokens)
                uncached_indices.append(i)
                uncached_tokens.append(tokens)
                uncached_texts.append(text)
        t_tok_end = time.time()

        token_lengths = [len(t) for t in uncached_tokens] if uncached_tokens else []
        logger.info(
            f"[PERF] texts={len(texts)} cached={len(results)} uncached={len(uncached_tokens)} "
            f"total_tokens={total_tokens} tokenize_ms={(t_tok_end - t_tok_start) * 1000:.1f} "
            f"token_len_range={min(token_lengths) if token_lengths else 0}-{max(token_lengths) if token_lengths else 0}"
        )

        # Submit uncached tokens to the coalescing queue
        if uncached_tokens:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            await self._coalesce_queue.put({
                "tokens": uncached_tokens,
                "model_name": model_name,
                "normalize": normalize,
                "future": future,
            })

            # Wait for the coalescer to process our batch
            new_embeddings = await future

            # Place results and update cache
            for j, idx in enumerate(uncached_indices):
                emb = new_embeddings[j]
                results[idx] = emb
                cache_key = self._cache_key(model_name, uncached_texts[j], normalize)
                self._cache_put(cache_key, emb, len(uncached_tokens[j]))

        ordered = np.array([results[i] for i in range(len(texts))], dtype=np.float32)
        return ordered, model_name, embedding_dim, total_tokens

    def get_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current model status and information"""
        if model_name:
            model_name = self._resolve_model_name(model_name)
            return {
                "status": self.model_status.get(model_name, ModelStatus.UNLOADED).value,
                "model_name": model_name,
                "embedding_dim": AVAILABLE_MODELS[model_name]["embedding_dim"],
                "load_time": self.model_load_times.get(model_name),
            }

        models_status = {}
        for name in AVAILABLE_MODELS:
            models_status[name] = {
                "status": self.model_status.get(name, ModelStatus.UNLOADED).value,
                "embedding_dim": AVAILABLE_MODELS[name]["embedding_dim"],
                "load_time": self.model_load_times.get(name),
            }

        return {
            "loaded_models": list(self.models.keys()),
            "default_model": self.config.model_name,
            "cache_size": len(self._embedding_cache),
            "models": models_status,
        }

# Initialize model manager
model_manager = ModelManager(config)

# Pydantic models
class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    model_config = ConfigDict(str_strip_whitespace=True)

    input: str | List[str] = Field(
        ...,
        description="Input text(s) to embed. Can be a string or array of strings."
    )
    model: str = Field(
        default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        description="Model to use (full HuggingFace model name)"
    )
    encoding_format: Optional[str] = Field(
        default="float",
        description="Encoding format. Only 'float' and 'base64' are supported."
    )

    @field_validator('input')
    def validate_input(cls, v):
        if isinstance(v, str):
            if not v or v.isspace():
                raise ValueError("Input text cannot be empty or whitespace only")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input list cannot be empty")
            for i, text in enumerate(v):
                if not isinstance(text, str) or not text or text.isspace():
                    raise ValueError(f"Input at index {i} must be a non-empty string")
        return v


class OpenAIEmbeddingData(BaseModel):
    """Single embedding object in OpenAI format"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class OpenAIUsage(BaseModel):
    """Token usage in OpenAI format"""
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[OpenAIEmbeddingData]
    model: str
    usage: OpenAIUsage


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service health status")
    model_status: str = Field(..., description="Model status")
    model_name: str = Field(..., description="Model name")
    embedding_dim: int = Field(..., description="Embedding dimension")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info(f"Starting Qwen3 Embedding Server v{app.version}")
    logger.info(f"Configuration: {config}")
    logger.info(f"Available models: {list(AVAILABLE_MODELS.keys())}")
    
    try:
        # Load default model at startup
        await model_manager.load_model(config.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize server with default model: {e}")
        # Server can still start, models will be loaded on demand

    # Start the coalescing background task
    model_manager.start_coalescer(asyncio.get_running_loop())

    app.state.start_time = time.time()

    yield

    # Shutdown
    if model_manager._coalescer_task:
        model_manager._coalescer_task.cancel()
    logger.info("Shutting down server...")

# Create FastAPI application
app = FastAPI(
    title="Roo Code Indexer",
    description="OpenAI-compatible embedding server for Roo Code on Apple Silicon",
    version="0.1.2",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware if enabled
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests with timing"""
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Log successful requests
        logger.info(
            f"{request.method} {request.url.path} "
            f"- Status: {response.status_code} "
            f"- Time: {process_time:.2f}ms"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"{request.method} {request.url.path} "
            f"- Error: {e} "
            f"- Time: {process_time:.2f}ms"
        )
        raise

# API Routes
@app.get("/", tags=["General"])
async def root():
    """Get API information"""
    return {
        "service": "Roo Code Indexer",
        "version": app.version,
        "default_model": config.model_name,
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "health": "/health",
            "models": "/models",
        }
    }

@app.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    tags=["OpenAI Compatible"],
    status_code=status.HTTP_200_OK
)
async def openai_embeddings(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Accepts the same request format as OpenAI's /v1/embeddings API,
    making it a drop-in replacement for applications using the OpenAI SDK.
    """
    try:
        start_time = time.time()

        # Normalize input to a list of strings
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings (token count returned to avoid double tokenization)
        embeddings, model_used, embedding_dim, total_tokens = await model_manager.generate_embeddings_coalesced(
            texts,
            model_name=request.model,
            normalize=True
        )

        # Build response data
        data = [
            OpenAIEmbeddingData(
                embedding=embeddings[i].tolist(),
                index=i
            )
            for i in range(len(texts))
        ]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[v1/embeddings] {len(texts)} texts, {total_tokens} tokens, "
            f"{elapsed_ms:.1f}ms total, {elapsed_ms / max(len(texts), 1):.1f}ms/text"
        )

        return OpenAIEmbeddingResponse(
            data=data,
            model=model_used,
            usage=OpenAIUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"OpenAI embeddings endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
    status_code=status.HTTP_200_OK
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current health status of the service,
    including model readiness and resource usage.
    """
    try:
        # Get memory usage if available
        memory_mb = None
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        model_status = model_manager.get_status()
        
        # Check default model status
        default_model_status = model_manager.model_status.get(
            config.model_name, ModelStatus.UNLOADED
        )
        
        return HealthResponse(
            status="healthy" if default_model_status == ModelStatus.READY else "degraded",
            model_status=default_model_status.value,
            model_name=config.model_name,
            embedding_dim=AVAILABLE_MODELS[config.model_name]["embedding_dim"],
            memory_usage_mb=memory_mb,
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/models", tags=["Models"])
async def list_models():
    """
    List available models and their status.
    
    Returns information about all available models,
    their loading status and embedding dimensions.
    """
    return model_manager.get_status()

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )

# Main entry point
def main():
    """Run the embedding server."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Roo Code Indexer — MLX embedding server for macOS"
    )
    parser.add_argument("--model", default=None,
                        help="Model to load at startup (e.g. mlx-community/Qwen3-Embedding-4B-4bit-DWQ)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default=None,
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--log-level", default="info",
                        help="Log level (default: info)")
    args = parser.parse_args()

    # Override config from CLI args (env vars already applied at import time)
    if args.model:
        config.model_name = ModelManager(config)._resolve_model_name(args.model)
    if args.port:
        config.port = args.port
    if args.host:
        config.host = args.host

    uvicorn.run(
        "roocode_indexer.server:app",
        host=config.host,
        port=config.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()