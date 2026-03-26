from collections.abc import Iterable
from functools import wraps

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

SELECTIVE_AC_TARGETS = frozenset({"norm", "attn_proj", "mlp", "mla_up_proj", "routed_experts", "mamba", "linear_attn"})
_PATCHED_METHODS_ATTR = "_prime_rl_selective_ac_patched_methods"


def _is_norm_module(module: nn.Module) -> bool:
    return "norm" in type(module).__name__.lower()


def _should_checkpoint(module: nn.Module) -> bool:
    return torch.is_grad_enabled() and module.training


def checkpoint_method(module: nn.Module, method_name: str) -> None:
    patched_methods = frozenset(getattr(module, _PATCHED_METHODS_ATTR, ()))
    if method_name in patched_methods:
        return

    original = getattr(module, method_name)

    @wraps(original)
    def checkpointed(*args, **kwargs):
        if not _should_checkpoint(module):
            return original(*args, **kwargs)

        def fn(*checkpoint_args):
            return original(*checkpoint_args, **kwargs)

        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)

    setattr(module, method_name, checkpointed)
    setattr(module, _PATCHED_METHODS_ATTR, patched_methods.union({method_name}))


def _configure_norm_checkpointing(layer: nn.Module) -> None:
    for module in layer.modules():
        if module is layer or not _is_norm_module(module):
            continue
        checkpoint_method(module, "forward")


def _is_dense_mlp(mlp: nn.Module) -> bool:
    return not hasattr(mlp, "_run_routed_experts") and not hasattr(mlp, "tokens_per_expert")


def get_supported_targets(layer: nn.Module) -> frozenset[str]:
    supported_targets = {"norm"}
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    mamba = getattr(layer, "mamba", None)
    linear_attn = getattr(layer, "linear_attn", None)

    if self_attn is not None and hasattr(self_attn, "_attn_projections"):
        supported_targets.add("attn_proj")
    if self_attn is not None and hasattr(self_attn, "_mla_up_proj"):
        supported_targets.add("mla_up_proj")
    if mlp is not None and _is_dense_mlp(mlp):
        supported_targets.add("mlp")
    if mlp is not None and hasattr(mlp, "_run_routed_experts"):
        supported_targets.add("routed_experts")
    if mamba is not None:
        supported_targets.add("mamba")
    if linear_attn is not None:
        supported_targets.add("linear_attn")

    return frozenset(supported_targets)


def set_selective_activation_checkpointing(layer: nn.Module, targets: Iterable[str]) -> None:
    normalized_targets = frozenset(targets)
    invalid_targets = normalized_targets - SELECTIVE_AC_TARGETS
    if invalid_targets:
        raise ValueError(f"Unsupported selective activation checkpoint targets: {sorted(invalid_targets)}")

    enabled_targets = normalized_targets & get_supported_targets(layer)
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)

    mamba = getattr(layer, "mamba", None)
    linear_attn = getattr(layer, "linear_attn", None)

    if self_attn is not None and "attn_proj" in enabled_targets:
        checkpoint_method(self_attn, "_attn_projections")
        checkpoint_method(self_attn, "_output_proj")
    if self_attn is not None and "mla_up_proj" in enabled_targets:
        checkpoint_method(self_attn, "_mla_up_proj")
    if mlp is not None and "mlp" in enabled_targets:
        checkpoint_method(mlp, "forward")
    if mlp is not None and "routed_experts" in enabled_targets:
        checkpoint_method(mlp, "_run_routed_experts")
    if mamba is not None and "mamba" in enabled_targets:
        checkpoint_method(mamba, "forward")
    if linear_attn is not None and "linear_attn" in enabled_targets:
        checkpoint_method(linear_attn, "forward")
    if "norm" in enabled_targets:
        _configure_norm_checkpointing(layer)


def supports_selective_activation_checkpointing(layer: nn.Module) -> bool:
    return type(layer).__module__.startswith("prime_rl.trainer.models.")
