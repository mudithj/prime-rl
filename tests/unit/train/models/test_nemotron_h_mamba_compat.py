import math

import torch

from prime_rl.trainer.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from prime_rl.trainer.models.nemotron_h.modeling_nemotron_h import NemotronHMambaLayer

_NONTRIVIAL_MAMBA_EXPAND = dict(
    hidden_size=300,
    mamba_num_heads=7,
    mamba_head_dim=64,
    mamba_expand=2,
    ssm_state_size=80,
    mamba_n_groups=5,
    mamba_d_conv=6,
    layer_norm_epsilon=1e-5,
)


def test_nemotron_h_config_sets_zamba2_compat_fields():
    config = NemotronHConfig(**_NONTRIVIAL_MAMBA_EXPAND, layers_block_type=["mamba"])

    expected_expand = (config.mamba_num_heads * config.mamba_head_dim) / config.hidden_size

    assert config.expand == _NONTRIVIAL_MAMBA_EXPAND["mamba_expand"]
    assert math.isclose(config.mamba_expand, expected_expand)
    assert config.mamba_d_state == config.ssm_state_size
    assert config.mamba_d_conv == _NONTRIVIAL_MAMBA_EXPAND["mamba_d_conv"]
    assert config.mamba_ngroups == _NONTRIVIAL_MAMBA_EXPAND["mamba_n_groups"]
    assert config.mamba_headdim == config.mamba_head_dim
    assert config.n_mamba_heads == config.mamba_num_heads
    assert config.use_mem_eff_path is True
    assert config.add_bias_linear == config.use_bias


def test_nemotron_h_mamba_layer_uses_corrected_zamba2_expand(monkeypatch):
    seen = {}

    class FakeMixer(torch.nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            seen["layer_idx"] = layer_idx
            seen["mamba_expand"] = config.mamba_expand
            seen["mamba_d_state"] = config.mamba_d_state
            seen["mamba_d_conv"] = config.mamba_d_conv
            seen["mamba_ngroups"] = config.mamba_ngroups
            seen["mamba_headdim"] = config.mamba_headdim
            seen["n_mamba_heads"] = config.n_mamba_heads
            seen["use_mem_eff_path"] = config.use_mem_eff_path
            seen["add_bias_linear"] = config.add_bias_linear

        def forward(self, hidden_states):
            return hidden_states

    monkeypatch.setattr(
        "prime_rl.trainer.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer",
        FakeMixer,
    )
    monkeypatch.setattr(
        "prime_rl.trainer.models.nemotron_h.modeling_nemotron_h._patch_mamba2_use_triton_ssd",
        lambda: None,
    )

    config = NemotronHConfig(**_NONTRIVIAL_MAMBA_EXPAND, layers_block_type=["mamba"])
    NemotronHMambaLayer(config, layer_idx=3)

    expected_expand = (config.mamba_num_heads * config.mamba_head_dim) / config.hidden_size

    assert seen == {
        "layer_idx": 3,
        "mamba_expand": expected_expand,
        "mamba_d_state": config.ssm_state_size,
        "mamba_d_conv": _NONTRIVIAL_MAMBA_EXPAND["mamba_d_conv"],
        "mamba_ngroups": _NONTRIVIAL_MAMBA_EXPAND["mamba_n_groups"],
        "mamba_headdim": config.mamba_head_dim,
        "n_mamba_heads": config.mamba_num_heads,
        "use_mem_eff_path": True,
        "add_bias_linear": config.use_bias,
    }
