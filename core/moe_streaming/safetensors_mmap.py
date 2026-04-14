"""
SafetensorsMmapCache: Memory-mapped access to safetensors tensor data on SSD.

Adapted from mlx-flash (MIT license):
  https://github.com/matt-k-wong/mlx-flash

Key idea: Use mmap + madvise to let the OS manage which expert weights
stay in the page cache. We track exact byte ranges per tensor so we can
madvise(MADV_WILLNEED) for prefetching and madvise(MADV_DONTNEED) for eviction.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from pathlib import Path
from typing import Optional


class SafetensorsMmapCache:
    """
    Parses .safetensors headers and maintains mmap objects for direct SSD access.

    Tracks (mmap_obj, absolute_start, absolute_end, dtype) for every tensor
    in the model, grouped by layer for efficient prefetching.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.file_mmaps: dict[str, mmap.mmap] = {}
        self.file_handles: dict[str, object] = {}
        # tensor_name -> (mmap, abs_start, abs_end, dtype)
        self.tensor_locations: dict[str, tuple[mmap.mmap, int, int, str]] = {}
        self._load_all()

    def _load_all(self):
        """Parse headers of all safetensors files and build tensor index."""
        safetensor_files = sorted(self.model_path.glob("*.safetensors"))
        if not safetensor_files:
            safetensor_files = sorted(self.model_path.glob("model*.safetensors"))

        for sf in safetensor_files:
            self._load_shard(sf)

    def _load_shard(self, sf: Path):
        """Load one safetensors shard file."""
        try:
            f = open(sf, "rb")
            self.file_handles[sf.name] = f

            # Safetensors header: 8-byte little-endian uint64 = JSON header length
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                return

            header_len = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(header_len).decode("utf-8")
            metadata = json.loads(header_json)

            # Data starts after 8-byte length prefix + JSON bytes
            data_start = 8 + header_len

            # Mmap the whole file (read-only, no copy)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.file_mmaps[sf.name] = mm

            # Index every tensor
            for tensor_name, info in metadata.items():
                if tensor_name == "__metadata__":
                    continue
                offsets = info.get("data_offsets", [0, 0])
                dtype = info.get("dtype", "bf16")
                if len(offsets) != 2:
                    continue

                abs_start = data_start + offsets[0]
                abs_end = data_start + offsets[1]
                self.tensor_locations[tensor_name] = (mm, abs_start, abs_end, dtype, sf.name)

        except Exception as e:
            pass

    def get_tensor_range(self, tensor_name: str) -> Optional[tuple[mmap.mmap, int, int, str, str]]:
        """
        Returns (mmap, abs_start, abs_end, dtype, filename) for a tensor.
        Returns None if tensor not found.
        """
        return self.tensor_locations.get(tensor_name)

    def read_tensor_bytes(self, tensor_name: str) -> Optional[bytes]:
        """
        Directly read raw bytes of a tensor from SSD via mmap.
        Used for on-demand expert weight loading.
        """
        info = self.tensor_locations.get(tensor_name)
        if info is None:
            return None
        mm, start, end, dtype, fname = info
        return mm[start:end]

    def get_layer_ranges(self, layer_idx: int) -> dict[mmap.mmap, tuple[int, int, str, str]]:
        """
        Group all tensors belonging to layer_idx into contiguous byte ranges
        per mmap file, minimizing madvise() system calls.
        """
        import re
        # Qwen3.5 MoE tensor naming: model.language_model.layers.{idx}.*
        patterns = [
            rf"layers\.{layer_idx}\.",
            rf"h\.{layer_idx}\.",
            rf"blocks\.{layer_idx}\.",
        ]

        intervals_by_mmap: dict[mmap.mmap, list[tuple[int, int, str, str]]] = {}

        for t_name, info in self.tensor_locations.items():
            if any(re.search(p, t_name) for p in patterns):
                mm, start, end, dtype, fname = info
                if mm not in intervals_by_mmap:
                    intervals_by_mmap[mm] = []
                intervals_by_mmap[mm].append((start, end, dtype, fname))

        # Merge overlapping/adjacent intervals
        merged = {}
        for mm, intervals in intervals_by_mmap.items():
            intervals.sort(key=lambda x: x[0])
            if not intervals:
                continue
            merged[mm] = (intervals[0][0], intervals[-1][1], intervals[0][2], intervals[0][3])

        return merged

    def get_moe_expert_ranges(self, layer_idx: int) -> list[tuple[str, int, int, int]]:
        """
        Return byte ranges for each individual MoE expert tensor in a layer.

        For Qwen3.5 MoE: gate_up_proj has shape [256, 2048, 3072] (fused gate+up).
        We return the byte range for each of the 256 experts separately.

        Returns list of (tensor_name, expert_idx, start_byte, end_byte).
        """
        import re

        # Find the fused expert tensor for this layer
        expert_tensor_name = None
        for t_name in self.tensor_locations:
            if re.search(rf"layers\.{layer_idx}\..*experts.*gate_up_proj", t_name):
                expert_tensor_name = t_name
                break

        if expert_tensor_name is None:
            return []

        info = self.tensor_locations[expert_tensor_name]
        mm, abs_start, abs_end, dtype, fname = info

        # Parse shape from safetensors header
        # Shape: [256, 2048, 3072], dtype: BF16
        # Each expert: [2048, 3072] = 12,582,912 elements × 2 bytes = 24,414,720 bytes
        # We need to read the actual shape from the file
        shard_file = self.model_path / fname
        expert_size_bytes = self._get_expert_size(shard_file, expert_tensor_name)

        results = []
        current = abs_start
        for expert_idx in range(256):  # Qwen3.5 MoE has 256 experts
            next_pos = current + expert_size_bytes
            results.append((expert_tensor_name, expert_idx, current, next_pos))
            current = next_pos

        return results

    def _get_expert_size(self, shard_file: Path, tensor_name: str) -> int:
        """Get the byte size of a single expert within a fused tensor."""
        import struct

        with open(shard_file, "rb") as f:
            header_size_bytes = f.read(8)
            header_len = struct.unpack("<Q", header_size_bytes)[0]
            header_json = f.read(header_len).decode("utf-8")
            metadata = json.loads(header_json)

        info = metadata[tensor_name]
        total_bytes = info["data_offsets"][1] - info["data_offsets"][0]
        shape = info.get("shape", [])
        num_experts = shape[0] if shape else 256
        return total_bytes // num_experts

    def madvise_willneed(self, mm: mmap.mmap, start: int, length: int):
        """Hint to OS: this memory region will be needed soon (prefetch to page cache)."""
        try:
            os.madvise(mm, start, length, os.MADV_WILLNEED)
        except Exception:
            pass

    def madvise_dontneed(self, mm: mmap.mmap, start: int, length: int):
        """Hint to OS: this memory region is no longer needed (evict from page cache)."""
        try:
            os.madvise(mm, start, length, os.MADV_DONTNEED)
        except Exception:
            pass

    def shutdown(self):
        """Close all mmap handles."""
        for mm in self.file_mmaps.values():
            try:
                mm.close()
            except Exception:
                pass
        for f in self.file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self.file_mmaps.clear()
        self.file_handles.clear()
        self.tensor_locations.clear()
