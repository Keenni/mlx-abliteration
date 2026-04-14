"""
ExpertCache: LRU GPU memory cache for MoE expert weights.

Adapted from mlx-flash (MIT license):
  https://github.com/matt-k-wong/mlx-flash

Key design:
- LRU eviction policy (OrderedDict)
- Each expert is stored as an MLX array (on Metal/GPU)
- Eviction triggers mx.free() to actually release Metal memory
- Thread-safe with lock
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Optional


class ExpertCache:
    """
    Manages MoE expert weights in GPU (Metal) memory with LRU eviction.

    For Qwen3.5 MoE with 256 experts:
      - Each expert gate_up_proj: [2048, 3072] BF16 = 24MB
      - At most ~8-16 experts in memory at once (~200-400MB)

    The cache intercepts expert loading so that:
      1. Expert weights are loaded from mmap on first access (cache miss)
      2. Subsequent accesses use the GPU-resident copy (cache hit)
      3. LRU eviction frees GPU memory when full
    """

    def __init__(self, max_experts: int = 16):
        self.max_experts = max_experts
        # Key: (layer_idx, expert_idx), Value: MLX array on GPU
        self._cache: OrderedDict[tuple[int, int], Any] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, layer_idx: int, expert_idx: int) -> Optional[Any]:
        """
        Retrieve expert from cache. Updates LRU order if found.
        Returns None on cache miss.
        """
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, layer_idx: int, expert_idx: int, expert_weights: Any):
        """
        Add expert to cache, evicting LRU entry if at capacity.
        """
        key = (layer_idx, expert_idx)
        with self._lock:
            if key in self._cache:
                # Already present — just move to most recent
                self._cache.move_to_end(key)
                return

            # Evict oldest if full
            if len(self._cache) >= self.max_experts:
                oldest_key, oldest_weights = self._cache.popitem(last=False)
                self._evict(oldest_weights)

            self._cache[key] = expert_weights

    def _evict(self, weights: Any):
        """Free GPU memory for evicted weights."""
        try:
            import mlx.core as mx
            if hasattr(weights, "delete"):
                weights.delete()
            elif hasattr(mx, "free"):
                mx.free(weights)
        except Exception:
            pass

    def prefetch(self, layer_idx: int, expert_idx: int):
        """
        Hint that expert will be needed soon.
        Concrete loading is done by the prefetch worker;
        this just ensures space is reserved in the cache.
        """
        pass  # No-op; actual loading is triggered by forward pass

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_experts,
            }

    def clear(self):
        """Clear all cached experts."""
        with self._lock:
            for weights in self._cache.values():
                self._evict(weights)
            self._cache.clear()

    def resize(self, new_max: int):
        """Change max cache size, evicting if necessary."""
        with self._lock:
            self.max_experts = new_max
            while len(self._cache) > new_max:
                _, oldest = self._cache.popitem(last=False)
                self._evict(oldest)
