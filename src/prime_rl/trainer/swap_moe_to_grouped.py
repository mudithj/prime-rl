"""
Swap HF per-expert MoE layers in a Qwen3-Omni thinker with GroupedExperts (grouped gemm).

Call `swap_thinker_moe_to_grouped(model)` after loading the HF model but before FSDP/LoRA.
"""
import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.moe import MoE, MoEArgs


def swap_thinker_moe_to_grouped(model: nn.Module) -> int:
    """Replace HF MoE blocks in model.thinker.model.layers with GroupedExperts-based MoE.

    Returns the number of layers swapped.
    """
    thinker = model.thinker
    text_model = thinker.model
    text_config = thinker.config.text_config if hasattr(thinker.config, "text_config") else thinker.config

    num_experts = text_config.num_experts
    num_experts_per_tok = text_config.num_experts_per_tok
    hidden_size = text_config.hidden_size
    moe_intermediate_size = text_config.moe_intermediate_size
    norm_topk_prob = getattr(text_config, "norm_topk_prob", True)

    moe_args = MoEArgs(
        num_experts=num_experts,
        num_shared_experts=0,
        score_func="softmax",
        route_norm=norm_topk_prob,
        route_scale=1.0,
        score_before_experts=False,
        top_k=num_experts_per_tok,
        use_grouped_mm=True,
        load_balance_coeff=getattr(text_config, "router_aux_loss_coef", 0.001),
    )

    swapped = 0
    for layer_idx, layer in enumerate(text_model.layers):
        mlp = layer.mlp
        # Check if this layer has a MoE block (has .gate and .experts)
        if not hasattr(mlp, "gate") or not hasattr(mlp, "experts"):
            continue

        # Create custom MoE module
        custom_moe = MoE(moe_args, dim=hidden_size, hidden_dim=moe_intermediate_size)

        # Copy router weights
        custom_moe.router.gate.weight.data.copy_(mlp.gate.weight.data)

        # Convert expert weights from HF format to grouped format
        # HF new format: experts.gate_up_proj [num_experts, 2*moe_dim, dim]
        #                experts.down_proj [num_experts, dim, moe_dim]
        if hasattr(mlp.experts, "gate_up_proj"):
            gate_up = mlp.experts.gate_up_proj.data
            down = mlp.experts.down_proj.data
            moe_dim = gate_up.shape[1] // 2
            w1 = gate_up[:, :moe_dim, :].contiguous()
            w3 = gate_up[:, moe_dim:, :].contiguous()
            w2 = down
            # Free HF weights immediately
            del mlp.experts.gate_up_proj
            del mlp.experts.down_proj
        else:
            # Old per-expert format
            w1 = torch.stack([mlp.experts[j].gate_proj.weight.data for j in range(num_experts)])
            w2 = torch.stack([mlp.experts[j].down_proj.weight.data for j in range(num_experts)])
            w3 = torch.stack([mlp.experts[j].up_proj.weight.data for j in range(num_experts)])
            # Free HF weights
            for j in range(num_experts):
                del mlp.experts[j]

        # Assign weight data directly (avoids allocating new tensors)
        custom_moe.experts.w1 = nn.Parameter(w1, requires_grad=w1.requires_grad)
        custom_moe.experts.w2 = nn.Parameter(w2, requires_grad=w2.requires_grad)
        custom_moe.experts.w3 = nn.Parameter(w3, requires_grad=w3.requires_grad)

        # Free old mlp before assigning new one
        del mlp
        layer.mlp = custom_moe
        torch.cuda.empty_cache()
        swapped += 1

    return swapped
