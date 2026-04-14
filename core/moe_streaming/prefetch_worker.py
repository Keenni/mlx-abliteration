"""
BackgroundPrefetcher: High-performance SSD prefetching via async os.pread.

Adapted from mlx-flash (MIT license):
  https://github.com/matt-k-wong/mlx-flash

Key design:
- Background thread with a bounded work queue (maxsize=16)
- Adaptive chunking: 16MB base chunks for optimal SSD queue depth
- Uses os.pread() for thread-safe, offset-based reads without seeking
- Tracks compute vs IO time to dynamically adjust prefetch distance (k-distance)
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Optional


class BackgroundPrefetcher:
    """
    Background worker that streams expert weight data from SSD into the
    macOS unified page cache using adaptive chunking and bandwidth control.
    """

    def __init__(self, file_handles: dict[str, Any], ram_budget_gb: float = 16.0):
        self.file_handles = file_handles
        self.ram_budget_gb = ram_budget_gb

        # Adaptive tuning
        self.k_distance = 2  # Prefetch this many layers ahead
        self.max_k = 4
        self.compute_ema = 0.0
        self.io_ema = 0.0

        # Bounded work queue
        self.queue: queue.Queue[tuple[Optional[int], str, int, int, int]] = queue.Queue(maxsize=32)
        self.completed_prefetches: set[int] = set()
        self._lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Main prefetch loop."""
        # 16MB base chunk — good for most NVMe drives
        base_chunk = 16 * 1024 * 1024

        while self.running:
            try:
                task = self.queue.get(timeout=0.05)
                if task is None:
                    continue

                layer_idx, filename, offset, length, align_bytes = task

                # Get or reopen file handle
                f = self.file_handles.get(filename)
                if f is None:
                    try:
                        f = open(filename, "rb")
                        self.file_handles[filename] = f
                    except Exception:
                        continue

                fd = f.fileno()
                end = offset + length
                pos = offset

                while pos < end and self.running:
                    # Adaptive chunk size aligned to expert boundary
                    requested = min(base_chunk, end - pos)
                    chunk = (requested // align_bytes) * align_bytes if align_bytes > 0 else requested
                    if chunk == 0:
                        chunk = align_bytes
                    chunk = min(chunk, end - pos)

                    t0 = time.perf_counter()

                    # os.pread: reads at offset WITHOUT changing file position (thread-safe)
                    try:
                        os.pread(fd, chunk, pos)
                    except Exception:
                        break

                    t1 = time.perf_counter()
                    io_time = t1 - t0

                    # Adaptive: if IO is fast relative to compute, increase k_distance
                    self._update_adaptive(io_time, chunk)

                    pos += chunk

                if layer_idx is not None:
                    with self._lock:
                        self.completed_prefetches.add(layer_idx)

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception:
                pass

    def _update_adaptive(self, io_time: float, bytes_read: int):
        """Dynamically adjust prefetch distance based on IO vs compute ratio."""
        alpha = 0.3
        bw_mbps = bytes_read / 1024 / 1024 / io_time if io_time > 0 else 0
        self.io_ema = alpha * bw_mbps + (1 - alpha) * self.io_ema if self.io_ema > 0 else bw_mbps

        # Rough heuristic: if page cache is keeping up, prefetch more aggressively
        if self.compute_ema > 0 and self.io_ema > 0:
            ratio = self.compute_ema / self.io_ema
            if ratio > 2 and self.k_distance < self.max_k:
                self.k_distance += 1
            elif ratio < 0.5 and self.k_distance > 1:
                self.k_distance -= 1

    def record_compute_time(self, compute_time: float):
        """Main thread calls this after each layer forward pass."""
        alpha = 0.3
        self.compute_ema = alpha * compute_time + (1 - alpha) * self.compute_ema if self.compute_ema > 0 else compute_time

    def wait_for_layer(self, layer_idx: int):
        """
        Block until layer_idx's prefetch is complete.
        Prevents hard page faults during forward pass.
        """
        while self.running:
            with self._lock:
                if layer_idx in self.completed_prefetches:
                    break
            if self.io_ema <= 0:
                break
            time.sleep(0.001)

    def enqueue(
        self,
        filename: str,
        offset: int,
        length: int,
        layer_idx: Optional[int] = None,
        align_bytes: int = 1,
    ):
        """Queue a prefetch task. Thread-safe."""
        if not self.running:
            return
        if layer_idx is not None:
            with self._lock:
                self.completed_prefetches.discard(layer_idx)
        try:
            self.queue.put_nowait((layer_idx, filename, offset, length, align_bytes))
        except queue.Full:
            pass  # Drop if full — the layer will fault-in naturally

    def prefetch_expert(
        self,
        filename: str,
        expert_start: int,
        expert_size: int,
        layer_idx: int,
        expert_idx: int,
    ):
        """Convenience: enqueue a single expert prefetch."""
        self.enqueue(filename, expert_start, expert_size, layer_idx=layer_idx, align_bytes=expert_size)

    def shutdown(self):
        """Stop the prefetch thread."""
        self.running = False
        try:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except Exception:
                    pass
        except Exception:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    @property
    def current_k_distance(self) -> int:
        """Current prefetch lookahead distance."""
        return self.k_distance
