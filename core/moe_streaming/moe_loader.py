"""
MoELoader — Integrates mmap expert streaming into mlx-abliteration.

Usage (abliteration with SSD-offload for MoE models):
    from core.moe_streaming import MoELoader

    loader = MoELoader(model_path)
    patched_model = loader.patch_all_moe_layers()

    # Now run activation probing with patched_model...
    # The SwitchLinear.__call__ will stream experts from mmap instead of
    # loading the full 3GB fused weight tensor into GPU memory.

Key components:
  1. SafetensorsMmapCache — mmap the safetensor files, build tensor index
  2. BackgroundPrefetcher — madvise(MADV_WILLNEED) prefetch thread
  3. ExpertCache — LRU cache of GPU-resident expert weights
  4. QwenMoeExpertStreamerSplit — patched SwitchLinear.__call__ that streams
     individual experts from mmap on-demand
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .expert_cache import ExpertCache
from .moe_patches import patch_qwen_moe_layer
from .prefetch_worker import BackgroundPrefetcher
from .safetensors_mmap import SafetensorsMmapCache


@dataclass
class LayerPatchStats:
    """Statistics for one patched MoE layer."""
    layer_idx: int
    num_experts: int
    gate_up_expert_mb: float
    down_expert_mb: float
    tensor_files: list[str]


class MoELoader:
    """
    Loads a Qwen3.5-122B-A10B MoE model with expert streaming from SSD.

    Instead of loading the full ~230GB of MoE expert weights into GPU memory,
    this patches the SwitchLinears to load individual experts from mmap on-demand.

    Memory usage:
      Without streaming:  ~230GB (all experts in GPU) → OOM on 128GB Mac
      With streaming:     ~400MB peak (8 active experts × 3 projections × ~16MB)
    """

    def __init__(
        self,
        model_path: str,
        ram_budget_gb: float = 16.0,
        max_cached_experts_per_layer: int = 8,
    ):
        self.model_path = model_path
        self.ram_budget_gb = ram_budget_gb
        self.max_cached = max_cached_experts_per_layer

        # Initialize components
        print(f"[MoELoader] Opening {model_path}...")
        t0 = time.time()
        self.mmap_cache = SafetensorsMmapCache(model_path)
        print(f"[MoELoader] SafetensorsMmapCache ready in {time.time()-t0:.1f}s "
              f"({len(self.mmap_cache.tensor_locations)} tensors across "
              f"{len(self.mmap_cache.file_handles)} files)")

        self.prefetcher = BackgroundPrefetcher(
            self.mmap_cache.file_handles,
            ram_budget_gb=ram_budget_gb,
        )
        print(f"[MoELoader] BackgroundPrefetcher started (budget={ram_budget_gb}GB)")

        self.expert_cache = ExpertCache(max_experts=max_cached_experts_per_layer)
        print(f"[MoELoader] ExpertCache initialized (max_experts={max_cached_experts_per_layer})")

        self._patched_layers: list[LayerPatchStats] = []
        self._patched_streamers: dict[int, list] = {}

    def _get_moe_tensor_info(self, layer_idx: int, tensor_suffix: str) -> Optional[tuple]:
        """Get mmap range info for a MoE expert tensor."""
        tensor_name = f"model.language_model.layers.{layer_idx}.mlp.experts.{tensor_suffix}"
        info = self.mmap_cache.get_tensor_range(tensor_name)
        return info

    def patch_all_moe_layers(self, model: nn.Module) -> nn.Module:
        """
        Patch all MoE layers in a model loaded with mlx_lm.load(lazy=True).

        This replaces SwitchLinear.__call__ for each of the 3 projections
        (gate_proj, up_proj, down_proj) in every MoE layer's SwitchGLU.

        After patching, forward passes will stream expert weights from mmap
        instead of loading the full fused tensor into GPU memory.

        Args:
            model: The nn.Module loaded via mlx_lm.load(..., lazy=True)

        Returns:
            The same model (patched in-place)
        """
        print(f"\n[MoELoader] Patching MoE layers...")

        # Find all MoE layers
        layers = self._find_moe_layers(model)
        print(f"[MoELoader] Found {len(layers)} MoE layers: {[l[0] for l in layers]}")

        for layer_idx, mlp_block in layers:
            self._patch_layer(layer_idx, mlp_block)

        total_experts = sum(s.num_experts for s in self._patched_layers)
        total_files = set()
        for s in self._patched_layers:
            total_files.update(s.tensor_files)
        print(f"[MoELoader] Patched {len(self._patched_layers)} layers, "
              f"{total_experts} total experts, {len(total_files)} safetensor files")

        return model

    def _patch_layer(self, layer_idx: int, mlp_block):
        """Patch a single MoE layer's SwitchLinears to use streaming."""
        t0 = time.time()

        streamers = patch_qwen_moe_layer(
            layer_idx=layer_idx,
            mlp_block=mlp_block,
            mmap_cache=self.mmap_cache,
            expert_cache=self.expert_cache,
            prefetcher=self.prefetcher,
        )

        # Collect stats
        gate_info = self._get_moe_tensor_info(layer_idx, "gate_up_proj")
        down_info = self._get_moe_tensor_info(layer_idx, "down_proj")

        gate_mb = 0.0
        down_mb = 0.0
        files = []

        if gate_info:
            mm, s, e, dtype, fname = gate_info
            total = e - s
            gate_mb = total / 256 / 1024**2
            files.append(fname)

        if down_info:
            mm, s, e, dtype, fname = down_info
            down_mb = (e - s) / 256 / 1024**2
            files.append(fname)

        stats = LayerPatchStats(
            layer_idx=layer_idx,
            num_experts=256,
            gate_up_expert_mb=gate_mb,
            down_expert_mb=down_mb,
            tensor_files=files,
        )
        self._patched_layers.append(stats)
        self._patched_streamers[layer_idx] = streamers

        print(f"[MoELoader] L{layer_idx} patched in {time.time()-t0:.3f}s "
              f"({len(streamers)} projections patched)")

    def _find_moe_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        """Find all MoE MLP blocks in the model."""
        moe_layers = []

        try:
            # Qwen3.5 model structure
            layers = model.language_model.model.layers
        except AttributeError:
            return moe_layers

        for li, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            # Check if it's a MoE block (has switch_mlp)
            if hasattr(mlp, "switch_mlp") or hasattr(mlp, "gate"):
                moe_layers.append((li, mlp))

        return moe_layers

    def get_router_activations(
        self,
        model: nn.Module,
        layer_idx: int,
        x: mx.array,
    ) -> mx.array:
        """
        Run a forward pass up to the router (exclusive) and return the
        hidden state at the MoE layer's input.

        Used for computing the refusal vector from router activations.
        """
        # Run through embedding and all layers up to (but not including) layer_idx
        # Then return the hidden state at that layer's input
        raise NotImplementedError("Use model directly for now")

    def shutdown(self):
        """Clean up resources."""
        self.mmap_cache.shutdown()
        self.prefetcher.shutdown()
        print("[MoELoader] Shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


def load_model_with_moe_streaming(
    model_path: str,
    max_cached_experts: int = 8,
    ram_budget_gb: float = 16.0,
) -> tuple[nn.Module, Any]:
    """
    Load a Qwen3.5 MoE model with expert streaming patches.

    This patches the SwitchLinears to stream individual experts from mmap
    on-demand during forward passes, avoiding the OOM from loading the full
    ~230GB of fused expert weight tensors into GPU memory.

    The model is loaded with lazy=True so weights are NOT loaded eagerly.
    Forward passes will read expert data directly from mmap.

    Args:
        model_path: Path to the model directory
        max_cached_experts: Max experts to keep in GPU per layer (LRU)
        ram_budget_gb: RAM budget for the prefetcher

    Returns:
        Tuple of (patched_model, tokenizer, MoELoader)
    """
    import mlx_lm

    # Load model with lazy=True to avoid loading all weights into GPU
    # (avoids OOM before our mmap patches are installed)
    print(f"[load_model_with_moe_streaming] Loading model from {model_path} (lazy)...")
    t0 = time.time()
    model, tok = mlx_lm.load(model_path, lazy=True)
    print(f"[load_model_with_moe_streaming] Model loaded in {time.time()-t0:.1f}s")

    # Patch MoE layers to stream experts from mmap
    loader = MoELoader(
        model_path=model_path,
        ram_budget_gb=ram_budget_gb,
        max_cached_experts_per_layer=max_cached_experts,
    )
    model = loader.patch_all_moe_layers(model)

    return model, tok, loader
