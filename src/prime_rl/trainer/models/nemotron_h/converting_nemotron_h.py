"""Weight conversion between HuggingFace and PrimeRL formats for NemotronH.

HF NemotronH uses a unified `mixer` attribute for all layer types:
  - Mamba layers: backbone.layers.{i}.mixer.{in_proj, conv1d, ...}
  - Attention layers: backbone.layers.{i}.mixer.{q_proj, k_proj, v_proj, o_proj}
  - MoE layers: backbone.layers.{i}.mixer.{gate, experts, shared_experts, fc1_latent_proj, fc2_latent_proj}

PrimeRL separates these into distinct namespaces:
  - Mamba layers: model.layers.{i}.mamba.*
  - Attention layers: model.layers.{i}.self_attn.*
  - MoE layers: model.layers.{i}.mlp.{router, experts, shared_expert, fc1_latent_proj, fc2_latent_proj}

Global renames:
  - HF: backbone.embeddings.weight <-> PrimeRL: model.embed_tokens.weight
  - HF: backbone.norm_f.weight <-> PrimeRL: model.norm.weight
  - HF uses "backbone." prefix, PrimeRL uses "model." prefix
  - HF mtp.* keys (multi-token prediction) are converted to model.mtp.* format
"""

import torch
from torch import Tensor


def _rename_keys(state_dict: dict[str, Tensor], old_prefix: str, new_prefix: str):
    """Rename all keys matching old_prefix to new_prefix in-place."""
    keys_to_rename = [k for k in state_dict if k.startswith(old_prefix)]
    for key in keys_to_rename:
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict.pop(key)


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single layer from HF to PrimeRL format in-place."""
    prefix = f"model.layers.{layer_idx}."

    if layer_type == "moe":
        _convert_hf_moe_layer_to_prime(state_dict, prefix)
    elif layer_type == "attention":
        _convert_hf_attention_layer_to_prime(state_dict, prefix)
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}mamba.")


def _convert_hf_moe_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer: mixer.gate -> mlp.router, mixer.experts -> mlp.experts, etc."""
    mixer = f"{prefix}mixer."
    mlp = f"{prefix}mlp."

    # Router: gate.weight -> router.gate (nn.Parameter), gate.e_score_correction_bias -> router.e_score_correction_bias
    if f"{mixer}gate.weight" in state_dict:
        state_dict[f"{mlp}router.gate"] = state_dict.pop(f"{mixer}gate.weight")
    if f"{mixer}gate.e_score_correction_bias" in state_dict:
        state_dict[f"{mlp}router.e_score_correction_bias"] = state_dict.pop(f"{mixer}gate.e_score_correction_bias")

    # Experts: check if stored as individual weights (experts.{i}.up_proj.weight)
    # or fused 3D tensors (experts.up_proj)
    individual_keys = [
        k
        for k in state_dict
        if k.startswith(f"{mixer}experts.") and k[len(f"{mixer}experts.") :].split(".")[0].isdigit()
    ]

    if individual_keys:
        expert_indices = sorted({int(k[len(f"{mixer}experts.") :].split(".")[0]) for k in individual_keys})

        up_projs = [state_dict.pop(f"{mixer}experts.{i}.up_proj.weight") for i in expert_indices]
        state_dict[f"{mlp}experts.w1"] = torch.stack(up_projs)

        down_projs = [state_dict.pop(f"{mixer}experts.{i}.down_proj.weight") for i in expert_indices]
        state_dict[f"{mlp}experts.w2"] = torch.stack(down_projs)
    else:
        # Fused 3D tensors
        if f"{mixer}experts.up_proj" in state_dict:
            state_dict[f"{mlp}experts.w1"] = state_dict.pop(f"{mixer}experts.up_proj")
        if f"{mixer}experts.down_proj" in state_dict:
            state_dict[f"{mlp}experts.w2"] = state_dict.pop(f"{mixer}experts.down_proj")

    # Dummy w3 required by @expert_parallel decorator compatibility
    device = state_dict[f"{mlp}experts.w1"].device if f"{mlp}experts.w1" in state_dict else "cpu"
    state_dict[f"{mlp}experts.w3"] = torch.empty(0, device=device)

    # Shared expert
    _rename_keys(state_dict, f"{mixer}shared_experts.", f"{mlp}shared_expert.")

    # Latent projections
    _rename_keys(state_dict, f"{mixer}fc1_latent_proj.", f"{mlp}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mixer}fc2_latent_proj.", f"{mlp}fc2_latent_proj.")


def _convert_hf_attention_layer_to_prime(state_dict: dict[str, Tensor], prefix: str):
    """Convert attention layer: mixer.{q,k,v,o}_proj -> self_attn.{q,k,v,o}_proj."""
    _rename_keys(state_dict, f"{prefix}mixer.", f"{prefix}self_attn.")


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int, layer_type: str):
    """Convert a single layer from PrimeRL to HF format in-place."""
    prefix = f"model.layers.{layer_idx}."

    if layer_type == "moe":
        _convert_prime_moe_layer_to_hf(state_dict, prefix)
    elif layer_type == "attention":
        _rename_keys(state_dict, f"{prefix}self_attn.", f"{prefix}mixer.")
    elif layer_type == "mamba":
        _rename_keys(state_dict, f"{prefix}mamba.", f"{prefix}mixer.")


def _convert_prime_moe_layer_to_hf(state_dict: dict[str, Tensor], prefix: str):
    """Convert MoE layer back to HF format."""
    mlp = f"{prefix}mlp."
    mixer = f"{prefix}mixer."

    # Router
    if f"{mlp}router.gate" in state_dict:
        state_dict[f"{mixer}gate.weight"] = state_dict.pop(f"{mlp}router.gate")
    if f"{mlp}router.e_score_correction_bias" in state_dict:
        state_dict[f"{mixer}gate.e_score_correction_bias"] = state_dict.pop(f"{mlp}router.e_score_correction_bias")

    # Experts: unstack fused 3D tensors into individual expert weights
    if f"{mlp}experts.w1" in state_dict:
        w1 = state_dict.pop(f"{mlp}experts.w1")
        for i in range(w1.shape[0]):
            state_dict[f"{mixer}experts.{i}.up_proj.weight"] = w1[i]
    if f"{mlp}experts.w2" in state_dict:
        w2 = state_dict.pop(f"{mlp}experts.w2")
        for i in range(w2.shape[0]):
            state_dict[f"{mixer}experts.{i}.down_proj.weight"] = w2[i]
    # Remove dummy w3 (not present in HF format)
    state_dict.pop(f"{mlp}experts.w3", None)

    # Shared expert
    _rename_keys(state_dict, f"{mlp}shared_expert.", f"{mixer}shared_experts.")

    # Latent projections
    _rename_keys(state_dict, f"{mlp}fc1_latent_proj.", f"{mixer}fc1_latent_proj.")
    _rename_keys(state_dict, f"{mlp}fc2_latent_proj.", f"{mixer}fc2_latent_proj.")


def _infer_mtp_sublayer_type(state_dict: dict[str, Tensor], prefix: str) -> str:
    """Infer sublayer type from keys after prefix (before mixer→self_attn/mlp rename)."""
    for k in state_dict:
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix) :]
        if suffix.startswith("mixer.gate.") or suffix.startswith("mixer.experts."):
            return "moe"
        if suffix.startswith("mixer.q_proj") or suffix.startswith("mixer.k_proj"):
            return "attention"
    return "attention"


def _detect_pattern_length(types: list[str]) -> int:
    """Find the shortest repeating period in a sublayer type sequence."""
    n = len(types)
    for period in range(1, n + 1):
        if n % period != 0:
            continue
        if all(types[i] == types[i % period] for i in range(n)):
            return period
    return n


def _convert_mtp_hf_to_prime(state_dict: dict[str, Tensor]):
    """Convert MTP weights from HF format to PrimeRL format in-place.

    HF Nemotron-H stores MTP as: mtp.layers.{flat_idx}.{mixer/norm/enorm/hnorm/eh_proj/final_layernorm}.*
    The flat sublayers follow a repeating pattern (e.g. *E = attention + MoE per step).
    For multi-step MTP, flat indices are grouped by pattern length into prediction steps.
    """
    mtp_keys = [k for k in state_dict if k.startswith("mtp.")]
    if not mtp_keys:
        return

    # Drop shared embedding (reuses backbone embedding)
    for k in [k for k in mtp_keys if "embed_tokens" in k or "embeddings" in k]:
        del state_dict[k]

    flat_indices = sorted({int(k.split(".")[2]) for k in state_dict if k.startswith("mtp.layers.")})
    if not flat_indices:
        return

    # Extract shared fusion from first sublayer
    first = f"mtp.layers.{flat_indices[0]}"
    for fusion_key in ("enorm.", "hnorm.", "eh_proj."):
        for k in [k for k in list(state_dict) if k.startswith(f"{first}.{fusion_key}")]:
            state_dict[f"model.mtp.{fusion_key}{k[len(f'{first}.{fusion_key}') :]}"] = state_dict.pop(k)

    # Extract output norm from last sublayer
    last = f"mtp.layers.{flat_indices[-1]}"
    for k in [k for k in list(state_dict) if k.startswith(f"{last}.final_layernorm.")]:
        suffix = k[len(f"{last}.final_layernorm.") :]
        state_dict[f"model.mtp.norm.{suffix}"] = state_dict.pop(k)

    # Detect pattern length from sublayer types to group into prediction steps
    types = [_infer_mtp_sublayer_type(state_dict, f"mtp.layers.{i}.") for i in flat_indices]
    pattern_len = _detect_pattern_length(types)

    # Rename remaining sublayer keys, distributing across prediction steps
    for j, flat_idx in enumerate(flat_indices):
        step = j // pattern_len
        local = j % pattern_len
        old_prefix = f"mtp.layers.{flat_idx}."
        new_prefix = f"model.mtp.layers.{step}.sublayers.{local}."

        for k in [k for k in list(state_dict) if k.startswith(old_prefix)]:
            state_dict[new_prefix + k[len(old_prefix) :]] = state_dict.pop(k)

        sublayer_type = _infer_mtp_sublayer_type(state_dict, new_prefix)
        if sublayer_type == "moe":
            _convert_hf_moe_layer_to_prime(state_dict, new_prefix)
        elif sublayer_type == "attention":
            _convert_hf_attention_layer_to_prime(state_dict, new_prefix)


def _convert_mtp_prime_to_hf(state_dict: dict[str, Tensor]):
    """Convert MTP weights from PrimeRL format to HF format in-place."""
    mtp_keys = [k for k in state_dict if k.startswith("model.mtp.")]
    if not mtp_keys:
        return

    # Collect all (step, local) pairs from model.mtp.layers.{step}.sublayers.{local}.*
    step_local_pairs = sorted(
        {
            (int(k.split(".")[3]), int(k.split(".")[5]))
            for k in state_dict
            if k.startswith("model.mtp.layers.") and ".sublayers." in k
        }
    )

    # Convert sublayer types back to HF format, then flatten to sequential indices
    flat_idx = 0
    last_flat = 0
    for step, local in step_local_pairs:
        prefix = f"model.mtp.layers.{step}.sublayers.{local}."
        if any(k.startswith(f"{prefix}self_attn.") for k in state_dict):
            _rename_keys(state_dict, f"{prefix}self_attn.", f"{prefix}mixer.")
        elif any(k.startswith(f"{prefix}mlp.") for k in state_dict):
            _convert_prime_moe_layer_to_hf(state_dict, prefix)

        _rename_keys(state_dict, prefix, f"mtp.layers.{flat_idx}.")
        last_flat = flat_idx
        flat_idx += 1

    # Move fusion components to first sublayer
    for fusion_key in ("enorm.", "hnorm.", "eh_proj."):
        _rename_keys(state_dict, f"model.mtp.{fusion_key}", f"mtp.layers.0.{fusion_key}")

    # Move output norm to last sublayer
    _rename_keys(state_dict, "model.mtp.norm.", f"mtp.layers.{last_flat}.final_layernorm.")

    # Clean up any remaining model.mtp keys (e.g. layers container)
    for k in [k for k in list(state_dict) if k.startswith("model.mtp.")]:
        del state_dict[k]


def convert_hf_to_prime(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from HF to PrimeRL format in-place."""
    # Handle backbone.* -> model.* prefix (HF NemotronH uses "backbone" instead of "model")
    _rename_keys(state_dict, "backbone.", "model.")

    # Always convert MTP keys so the cached snapshot works for both MTP-enabled
    # and non-MTP runs.  Unused keys are ignored by dcp_load (it only loads keys
    # present in the model's state_dict).
    has_mtp = any(k.startswith("mtp.") for k in state_dict)
    if has_mtp:
        _convert_mtp_hf_to_prime(state_dict)

    # Global renames
    if "model.embeddings.weight" in state_dict:
        state_dict["model.embed_tokens.weight"] = state_dict.pop("model.embeddings.weight")
    if "model.norm_f.weight" in state_dict:
        state_dict["model.norm.weight"] = state_dict.pop("model.norm_f.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_hf_layer_to_prime(state_dict, i, layer_type)


def convert_prime_to_hf(state_dict: dict[str, Tensor], layers_block_type: list[str]):
    """Convert full model from PrimeRL to HF format in-place."""
    # Convert MTP before global rename
    _convert_mtp_prime_to_hf(state_dict)

    # Global renames
    if "model.embed_tokens.weight" in state_dict:
        state_dict["model.embeddings.weight"] = state_dict.pop("model.embed_tokens.weight")
    if "model.norm.weight" in state_dict:
        state_dict["model.norm_f.weight"] = state_dict.pop("model.norm.weight")

    for i, layer_type in enumerate(layers_block_type):
        convert_prime_layer_to_hf(state_dict, i, layer_type)

    # Rename model.* -> backbone.* (HF checkpoint uses "backbone" prefix)
    _rename_keys(state_dict, "model.", "backbone.")
