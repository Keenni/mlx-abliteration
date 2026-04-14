"""
Activation Capture Hooks for MoE Abliteration.

These hooks intercept the model forward pass to capture intermediate
activations needed for computing refusal vectors.

Key insight from mlx-abliteration:
  - We need hidden activations at certain layers (typically early-to-mid layers)
  - For MoE: the MoE router's top-k decision is critical (which experts are selected)
  - The refusal vector = mean(activations_refusal) - mean(activations_normal)

Two activation capture modes:
  1. STANDARD: Uses standard mlx_lm.load() + custom hooks (for dense models or
     when full model fits in memory)
  2. SSD_STREAMING: Uses SafetensorsMmapCache + BackgroundPrefetcher for models
     too large to fit in RAM (e.g. 122B MoE on 128GB Mac)
"""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn


class ActivationBuffer:
    """
    Accumulates captured activations across a batch of inputs.

    For abliteration we need:
      - activations from normal inputs (for computing mean)
      - activations from refusal inputs (for computing mean)
      - The refusal_vector = mean(refusal_activations) - mean(normal_activations)
    """

    def __init__(self):
        self.normal_activations: list[mx.array] = []
        self.refusal_activations: list[mx.array] = []
        self._capture_mode: str | None = None  # "normal" or "refusal"

    def set_mode(self, mode: str):
        """Set current capture mode: 'normal' or 'refusal'."""
        self._capture_mode = mode

    def append(self, activation: mx.array):
        """Append a captured activation to the appropriate buffer."""
        if self._capture_mode == "normal":
            self.normal_activations.append(activation)
        elif self._capture_mode == "refusal":
            self.refusal_activations.append(activation)

    def get_refusal_vector(self) -> mx.array | None:
        """
        Compute the refusal vector: mean(refusal) - mean(normal).
        Returns None if either buffer is empty.
        """
        if not self.normal_activations or not self.refusal_activations:
            return None

        # Stack and compute mean
        normal_stack = mx.stack([mx.as_eval(x) for x in self.normal_activations])
        refusal_stack = mx.stack([mx.as_eval(x) for x in self.refusal_activations])

        mean_normal = mx.mean(normal_stack, axis=0)
        mean_refusal = mx.mean(refusal_stack, axis=0)

        return mean_refusal - mean_normal

    def clear(self):
        self.normal_activations.clear()
        self.refusal_activations.clear()
        self._capture_mode = None


class MoEActivationCaptureHook:
    """
    Hook that intercepts MoE layer execution and captures:
      1. Router top-k decisions (which experts were selected)
      2. Layer hidden activations (input to the MoE layer)

    This hook is designed to work WITH the mlx-flash StreamingProxy architecture.
    It adds on top of the existing hook system rather than replacing it.

    Usage:
        hook = MoEActivationCaptureHook(capture_layers=[4, 12, 20])
        model = hook.patch_model(model)
        # Run forward passes...
        refusal_vector = hook.compute_refusal_vector()
    """

    def __init__(
        self,
        capture_layers: list[int] | None = None,
        use_ssd_streaming: bool = False,
        mmap_cache=None,
        prefetcher=None,
    ):
        """
        Args:
            capture_layers: Which transformer layers to capture activations from.
                           None = capture all MoE layers.
            use_ssd_streaming: If True, use mmap-based expert loading instead of
                              pre-loading all experts into GPU memory.
            mmap_cache: SafetensorsMmapCache instance for SSD streaming.
            prefetcher: BackgroundPrefetcher instance for SSD streaming.
        """
        self.capture_layers = capture_layers
        self.use_ssd_streaming = use_ssd_streaming
        self.mmap_cache = mmap_cache
        self.prefetcher = prefetcher

        # Per-layer capture state
        self._layer_activations: dict[int, list[mx.array]] = {}
        self._layer_router_decisions: dict[int, list[list[int]]] = {}

    def on_layer_start(self, ctx: Any, layer: nn.Module):
        """
        Called before each layer executes (from StreamingProxy dispatch).
        Trigger prefetch for MoE experts if using SSD streaming.
        """
        if not self.use_ssd_streaming:
            return

        layer_idx = ctx.layer_idx
        if self.capture_layers is not None and layer_idx not in self.capture_layers:
            return

        # Prefetch experts for this layer
        self._prefetch_moe_experts(layer_idx)

    def _prefetch_moe_experts(self, layer_idx: int):
        """Issue prefetch requests for MoE experts of a layer."""
        if self.mmap_cache is None or self.prefetcher is None:
            return

        expert_ranges = self.mmap_cache.get_moe_expert_ranges(layer_idx)
        for tensor_name, expert_idx, start, end in expert_ranges:
            self.prefetcher.prefetch_expert(
                filename=str(self.mmap_cache.model_path / expert_ranges[0][3]),
                expert_start=start,
                expert_size=end - start,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
            )

    def on_layer_end(self, ctx: Any, layer: nn.Module, output: mx.array):
        """
        Called after each layer executes.
        Capture the hidden activation (output of this layer).
        """
        layer_idx = ctx.layer_idx

        if self.capture_layers is not None and layer_idx not in self.capture_layers:
            return

        # Check if this is an MoE layer (has experts attribute)
        is_moe = self._is_moe_layer(layer)

        # Materialize the output array
        mx.eval(output)
        # Make a copy so the array doesn't get reused
        activation_copy = mx.copy(output)

        if layer_idx not in self._layer_activations:
            self._layer_activations[layer_idx] = []
        self._layer_activations[layer_idx].append(activation_copy)

        if is_moe:
            self._capture_router_decision(ctx, layer, layer_idx)

    def _is_moe_layer(self, layer: nn.Module) -> bool:
        """Check if a layer is an MoE layer."""
        # MoE layers in Qwen3 have an 'experts' sub-module
        if hasattr(layer, "experts"):
            return True
        # Fallback: check for router/gate attribute
        if hasattr(layer, "gate"):
            return True
        return False

    def _capture_router_decision(self, ctx: Any, layer: nn.Module, layer_idx: int):
        """
        Capture the router's top-k expert selection.
        This tells us which experts were activated for each token.
        """
        # The router decision is computed inside the MoE layer's forward pass.
        # To capture it, we'd need to monkey-patch the router's forward method.
        # For now, we store a placeholder — the concrete router hook is
        # installed separately by patch_moe_router().
        pass

    def get_layer_activations(self, layer_idx: int) -> list[mx.array]:
        """Get all captured activations for a specific layer."""
        return self._layer_activations.get(layer_idx, [])

    def compute_refusal_vector(
        self,
        normal_activations: list[mx.array],
        refusal_activations: list[mx.array],
    ) -> mx.array | None:
        """
        Compute refusal vector = mean(refusal) - mean(normal).
        Uses only first-captured activations (index 0).
        """
        if not normal_activations or not refusal_activations:
            return None

        # Use the first captured activation per layer as representative
        norm_stack = mx.stack([mx.as_eval(a) for a in normal_activations[:1]])
        ref_stack = mx.stack([mx.as_eval(a) for a in refusal_activations[:1]])

        mean_norm = mx.mean(norm_stack, axis=0)
        mean_ref = mx.mean(ref_stack, axis=0)

        return mean_ref - mean_norm

    def clear(self):
        """Clear all captured activations."""
        self._layer_activations.clear()
        self._layer_router_decisions.clear()


class RouterDecisionCapture:
    """
    Installs a monkey-patch on the MoE router to capture top-k decisions.

    This is a surgical patch: we replace the router's forward method with
    a wrapper that calls the original AND saves the top_k_indices.
    """

    def __init__(self):
        self._patches: list[tuple[Any, str, Any]] = []  # (obj, attr, original)
        self._decisions: dict[int, list[mx.array]] = {}  # layer_idx -> [top_k_indices per token]

    def patch(self, model: nn.Module):
        """
        Find all MoE routers in the model and patch their forward methods.
        Returns list of (layer_idx, router_obj) for each patched router.
        """
        patched = []

        def make_wrapper(original_forward, layer_idx):
            def wrapper(*args, **kwargs):
                result = original_forward(*args, **kwargs)
                # result is typically (top_k_indices, top_k_weights)
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    self._record_decision(layer_idx, result[0])
                return result

            return wrapper

        # Walk the model tree
        def walk(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if this is a router (gate) module
                if self._is_router(child):
                    original = child.__class__.forward
                    wrapper = make_wrapper(original, len(patched))
                    child.__class__.forward = wrapper
                    self._patches.append((child.__class__, "forward", original))
                    patched.append((len(patched), child))

                walk(child, full_name)

        walk(model)
        return patched

    def _is_router(self, module: nn.Module) -> bool:
        """Check if module is an MoE router."""
        return hasattr(module, "gate") or type(module).__name__.lower().find("router") >= 0

    def _record_decision(self, layer_idx: int, top_k_indices: mx.array):
        """Record router's top-k expert decisions."""
        if layer_idx not in self._decisions:
            self._decisions[layer_idx] = []
        # Materialize and copy
        mx.eval(top_k_indices)
        self._decisions[layer_idx].append(mx.copy(top_k_indices))

    def get_decisions(self, layer_idx: int) -> list[mx.array]:
        """Get captured router decisions for a layer."""
        return self._decisions.get(layer_idx, [])

    def restore(self):
        """Restore original forward methods."""
        for obj, attr, original in self._patches:
            setattr(obj, attr, original)
        self._patches.clear()
        self._decisions.clear()
