from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.trainer import MTPConfig
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL


def _get_next_cp_first_value(t: Tensor, cp_group: dist.ProcessGroup) -> Tensor | None:
    """Return the next CP rank's first sequence element."""
    cp_world_size = dist.get_world_size(group=cp_group)
    first_value = t[..., :1].contiguous()
    gathered = [torch.empty_like(first_value) for _ in range(cp_world_size)]
    dist.all_gather(gathered, first_value, group=cp_group)
    cp_rank = dist.get_rank(group=cp_group)
    if cp_rank + 1 >= cp_world_size:
        return None
    return gathered[cp_rank + 1]


def _shift_left(t: Tensor, position_ids: Tensor | None = None, cp_group: dist.ProcessGroup | None = None) -> Tensor:
    """Shift tensor one position left, zero-padding the end of each packed sequence."""
    shifted = F.pad(t[..., 1:], (0, 1))
    if position_ids is not None:
        boundary = position_ids[..., :-1] >= position_ids[..., 1:]
        shifted[..., :-1][boundary] = 0
        if cp_group is not None and dist.get_world_size(group=cp_group) > 1:
            assert t.shape[0] == 1, "Context-parallel MTP expects a single local batch shard"
            next_position = _get_next_cp_first_value(position_ids, cp_group)
            next_value = _get_next_cp_first_value(t, cp_group)
            if next_position is not None and next_value is not None:
                continues_sequence = next_position > position_ids[..., -1:]
                shifted[..., -1:] = torch.where(continues_sequence, next_value, torch.zeros_like(shifted[..., -1:]))
    return shifted


def compute_mtp_step_masks(
    loss_mask: Tensor,
    num_steps: int = 1,
    position_ids: Tensor | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Return the valid-token mask for each MTP prediction depth."""
    if num_steps < 1:
        raise ValueError(f"MTP requires at least one prediction step, got {num_steps}")

    shifted = _shift_left(loss_mask.float(), position_ids, cp_group=cp_group)
    cumulative = shifted
    step_masks = []
    for _ in range(num_steps):
        shifted = _shift_left(shifted, position_ids, cp_group=cp_group)
        cumulative = cumulative * shifted
        step_masks.append(cumulative.clone())
    return torch.stack(step_masks, dim=0)


def compute_mtp_mask(
    loss_mask: Tensor,
    num_steps: int = 1,
    position_ids: Tensor | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Compute the deepest-step MTP mask for compatibility with older call sites."""
    return compute_mtp_step_masks(loss_mask, num_steps=num_steps, position_ids=position_ids, cp_group=cp_group)[-1]


def _chunked_mtp_cross_entropy(
    hidden_states: Tensor,
    labels: Tensor,
    detached_weight: Tensor,
    chunk_size: int,
) -> Tensor:
    """Compute per-token CE without materializing full-sequence logits."""
    batch_size, seq_len, _ = hidden_states.shape
    vocab_size = detached_weight.shape[0]
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = F.linear(hidden_states[:, start:end, :], detached_weight)
        chunk_loss = F.cross_entropy(logits.view(-1, vocab_size), labels[:, start:end].reshape(-1), reduction="none")
        chunks.append(chunk_loss.view(batch_size, -1))
        del logits
    return torch.cat(chunks, dim=1)


def reduce_mtp_token_loss(mtp_token_loss: Tensor, step_masks: Tensor) -> Tensor:
    """Average the masked per-step MTP loss across prediction depths."""
    if mtp_token_loss.ndim == step_masks.ndim - 1:
        mtp_token_loss = mtp_token_loss.unsqueeze(0)
    if mtp_token_loss.shape != step_masks.shape:
        raise ValueError(
            f"MTP loss tensor shape {tuple(mtp_token_loss.shape)} does not match mask shape {tuple(step_masks.shape)}"
        )

    reduce_dims = tuple(range(1, mtp_token_loss.ndim))
    step_loss_sums = (mtp_token_loss * step_masks).sum(dim=reduce_dims)
    step_token_counts = step_masks.sum(dim=reduce_dims)
    valid_steps = step_token_counts > 0
    if not valid_steps.any():
        return step_loss_sums.new_tensor(0.0)

    step_means = torch.where(
        valid_steps,
        step_loss_sums / step_token_counts.clamp(min=1),
        torch.zeros_like(step_loss_sums),
    )
    return step_means[valid_steps].mean()


def compute_mtp_loss(
    mtp_token_loss: Tensor,
    loss_mask: Tensor,
    num_steps: int = 1,
    position_ids: Tensor | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> Tensor:
    """Compute the auxiliary scalar MTP loss from per-token losses."""
    step_masks = compute_mtp_step_masks(loss_mask, num_steps=num_steps, position_ids=position_ids, cp_group=cp_group)
    return reduce_mtp_token_loss(mtp_token_loss, step_masks)


def compute_mtp_loss_stats(
    mtp_token_loss: Tensor,
    loss_mask: Tensor,
    num_steps: int = 1,
    position_ids: Tensor | None = None,
    cp_group: dist.ProcessGroup | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the scalar MTP mean loss and its token-weighted sum.

    The sum is weighted by the main loss token count so callers can align MTP
    scaling with objectives that normalize by the original masked-token total.
    """
    mtp_loss_mean = compute_mtp_loss(
        mtp_token_loss,
        loss_mask,
        num_steps=num_steps,
        position_ids=position_ids,
        cp_group=cp_group,
    )
    token_count = loss_mask.sum(dtype=mtp_loss_mean.dtype)
    return mtp_loss_mean, mtp_loss_mean * token_count


def compute_mtp_token_losses(
    model: PreTrainedModelPrimeRL,
    hidden_states: Tensor,
    input_ids: Tensor,
    position_ids: Tensor | None,
    chunk_size: int = 512,
) -> tuple[Tensor, int]:
    """Compute per-token MTP CE losses for each prediction step.

    Returns (token_loss [num_steps, B, S], num_steps).
    """
    cp_group = getattr(model, "_mtp_cp_group", None)
    num_steps = model.mtp_num_prediction_steps
    layers = model.mtp_layer_list

    h = hidden_states.detach()
    detached_weight = model.lm_head.weight.detach()
    step_losses = []
    current_ids, current_pos = input_ids, position_ids

    for step in range(num_steps):
        shifted_ids = _shift_left(current_ids, current_pos, cp_group=cp_group)
        labels = _shift_left(shifted_ids, current_pos, cp_group=cp_group)
        shifted_pos = _shift_left(current_pos, current_pos, cp_group=cp_group) if current_pos is not None else None

        with torch.no_grad():
            embeds = model.mtp_embed_tokens(shifted_ids)
            rotary = model.mtp_rotary_emb
            pos_emb = rotary(h, shifted_pos) if rotary is not None and shifted_pos is not None else None

        layer = layers[0] if model.mtp_shared_weights else layers[step]
        mtp_out = model.mtp_layer_forward(layer, h, embeds, shifted_pos, pos_emb)
        step_losses.append(_chunked_mtp_cross_entropy(mtp_out, labels, detached_weight, chunk_size))

        h, current_ids, current_pos = mtp_out, shifted_ids, shifted_pos

    return torch.stack(step_losses, dim=0), num_steps


def setup_mtp_training(model: nn.Module, mtp_config: MTPConfig) -> None:
    """Validate model has MTP support and store config for use during forward."""
    from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

    if not isinstance(model, PreTrainedModelPrimeRL):
        raise TypeError(
            f"MTP training requires a PreTrainedModelPrimeRL subclass, got {type(model).__name__}. "
            "Create a custom model implementation with the MTP interface."
        )
    mtp_layers = model.mtp_layer_list
    mtp_module = model.mtp_module
    if not mtp_layers or mtp_module is None:
        raise ValueError(f"MTP training enabled but {type(model).__name__} does not expose MTP layers.")

    model._mtp_config = mtp_config
    num_params = sum(p.numel() for p in mtp_module.parameters())
    get_logger().info(
        f"MTP training enabled: {len(mtp_layers)} layer(s), {model.mtp_num_prediction_steps} prediction step(s)"
        f"{' (shared weights)' if model.mtp_shared_weights else ''}, {num_params:,} parameters"
    )
