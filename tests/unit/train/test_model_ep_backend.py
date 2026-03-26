from types import SimpleNamespace

import prime_rl.trainer.model as model_module
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.distributed import DeepEPExpertParallel
from prime_rl.trainer.models.layers.moe import LatentMoE, MoE, MoEArgs
from torchtitan.distributed.expert_parallel import ExpertParallel


def _build_moe_layers() -> tuple[MoE, LatentMoE, SimpleNamespace]:
    moe = MoE(MoEArgs(num_experts=2, top_k=1, use_grouped_mm=False), dim=8, hidden_dim=16)
    latent_moe = LatentMoE(
        dim=8,
        latent_dim=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=8,
        num_experts=2,
        top_k=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        routed_scaling_factor=1.0,
        use_grouped_mm=False,
        load_balance_coeff=None,
    )
    language_model = SimpleNamespace(
        layers=[
            SimpleNamespace(mlp=moe),
            SimpleNamespace(mlp=latent_moe),
            SimpleNamespace(mlp=None),
        ]
    )
    return moe, latent_moe, language_model


def test_apply_ep_keeps_latent_moe_on_standard_plan(monkeypatch) -> None:
    moe, latent_moe, language_model = _build_moe_layers()
    mesh = object()
    captured_calls = []

    def fake_parallelize_module(module, device_mesh, parallelize_plan):
        captured_calls.append((module, device_mesh, parallelize_plan))

    monkeypatch.setattr(model_module, "get_language_model", lambda _model: language_model)
    monkeypatch.setattr(model_module, "parallelize_module", fake_parallelize_module)

    config = ModelConfig(name="Qwen/Qwen3-0.6B", ep=2, ep_comm_backend="deepep")
    parallel_dims = SimpleNamespace(get_mesh=lambda name: mesh)

    model_module.apply_ep(SimpleNamespace(), config, parallel_dims)

    assert captured_calls == [
        (moe.experts, mesh, captured_calls[0][2]),
        (latent_moe.experts, mesh, captured_calls[1][2]),
    ]
    assert isinstance(captured_calls[0][2], DeepEPExpertParallel)
    assert isinstance(captured_calls[1][2], ExpertParallel)
