import pytest

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.model import apply_ac
from prime_rl.trainer.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from prime_rl.trainer.models.glm4_moe.modeling_glm4_moe import Glm4MoeDecoderLayer
from prime_rl.trainer.models.layers.checkpointing import (
    _PATCHED_METHODS_ATTR,
    get_supported_targets,
    set_selective_activation_checkpointing,
)
from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM, NemotronHMoELayer


def _make_glm4_moe_config() -> Glm4MoeConfig:
    config = Glm4MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        moe_intermediate_size=64,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_routed_experts=4,
        first_k_dense_replace=1,
        use_grouped_mm=False,
    )
    config._attn_implementation = "sdpa"
    return config


def _make_nemotron_h_moe_config() -> NemotronHConfig:
    return NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        intermediate_size=128,
        mamba_expand=2,
        mamba_num_heads=2,
        mamba_head_dim=64,
        ssm_state_size=16,
        mamba_n_groups=1,
        mamba_d_conv=4,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        moe_latent_size=32,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        layers_block_type=["moe"],
        use_grouped_mm=False,
    )


def test_dense_mlp_layers_advertise_mlp_target() -> None:
    layer = Glm4MoeDecoderLayer(_make_glm4_moe_config(), layer_idx=0)

    supported_targets = get_supported_targets(layer)

    assert "mlp" in supported_targets
    assert "routed_experts" not in supported_targets


def test_standard_moe_layers_only_patch_routed_experts() -> None:
    layer = Glm4MoeDecoderLayer(_make_glm4_moe_config(), layer_idx=1)

    supported_targets = get_supported_targets(layer)

    assert "mlp" not in supported_targets
    assert "routed_experts" in supported_targets

    set_selective_activation_checkpointing(layer, ["routed_experts"])

    patched_methods = getattr(layer.mlp, _PATCHED_METHODS_ATTR, frozenset())
    assert "_run_routed_experts" in patched_methods
    assert "forward" not in patched_methods


def test_latent_moe_layers_do_not_advertise_or_patch_mlp_target() -> None:
    layer = NemotronHMoELayer(_make_nemotron_h_moe_config())

    supported_targets = get_supported_targets(layer)

    assert supported_targets == frozenset({"norm"})

    set_selective_activation_checkpointing(layer, ["mlp"])

    patched_methods = getattr(layer.mlp, _PATCHED_METHODS_ATTR, frozenset())
    assert "forward" not in patched_methods


def test_apply_ac_rejects_mlp_target_for_nemotron_h_moe_layers() -> None:
    model = NemotronHForCausalLM(_make_nemotron_h_moe_config())

    with pytest.raises(ValueError, match=r"Selective activation checkpoint targets \['mlp'\] are not supported"):
        apply_ac(model, ActivationCheckpointConfig(mode="selective", targets=["mlp"]))
