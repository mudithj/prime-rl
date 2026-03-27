from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import PretrainedConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.nccl import preprocess_layer_checkpoint, preprocess_layer_quantized


class DummyConfig(PretrainedConfig):
    model_type = "dummy-nccl-broadcast"


class DummyDefaultNonLayerModel(PreTrainedModelPrimeRL):
    config_class = DummyConfig

    def __init__(self):
        super().__init__(DummyConfig())
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(1, 1)])

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return not cls.is_prime_state_dict(state_dict)

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(key.startswith("prime.non_layer.") or "mlp.experts.w1" in key for key in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_to_prime(cls, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_hf(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        if layer_idx < 0:
            state_dict["hf.non_layer.weight"] = state_dict.pop("prime.non_layer.weight")
            return state_dict
        state_dict[f"converted.layer.{layer_idx}"] = state_dict.pop(f"model.layers.{layer_idx}.mlp.experts.w1")
        return state_dict

    @classmethod
    def convert_layer_to_prime(cls, state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
        return state_dict

    @classmethod
    def convert_layer_to_vllm_kernel(
        cls,
        state_dict: dict[str, Tensor],
        layer_idx: int,
        quantize_fp8: bool = False,
    ) -> dict[str, Tensor]:
        state_dict[f"kernel.layer.{layer_idx}"] = state_dict.pop(f"model.layers.{layer_idx}.mlp.experts.w1")
        return state_dict

    def init_buffers_post_meta(self) -> None:
        return None


class DummyCustomNonLayerPrimeModel(DummyDefaultNonLayerModel):
    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return any(key.startswith("model.mtp.") or "mlp.experts.w1" in key for key in state_dict)

    def convert_non_layer_to_hf(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        state_dict["mtp.layers.0.weight"] = state_dict.pop("model.mtp.layers.0.weight")
        return state_dict


def test_preprocess_layer_checkpoint_falls_back_for_non_prime_non_layer(
    monkeypatch,
) -> None:
    model = DummyDefaultNonLayerModel()
    state_dict = {"weight": torch.ones(1)}
    calls: list[dict[str, Tensor]] = []

    def fake_revert_weight_conversion(model_arg, state_dict_arg):
        calls.append(state_dict_arg.copy())
        return {"fallback.weight": state_dict_arg["weight"]}

    monkeypatch.setattr("transformers.core_model_loading.revert_weight_conversion", fake_revert_weight_conversion)

    converted = preprocess_layer_checkpoint(model, state_dict, layer_idx=-1)

    assert set(converted) == {"fallback.weight"}
    assert torch.equal(converted["fallback.weight"], torch.ones(1))
    assert len(calls) == 1
    assert set(calls[0]) == {"weight"}
    assert torch.equal(calls[0]["weight"], torch.ones(1))


def test_preprocess_layer_checkpoint_converts_prime_non_layer_with_default_handler() -> None:
    model = DummyDefaultNonLayerModel()
    state_dict = {"prime.non_layer.weight": torch.ones(1)}

    converted = preprocess_layer_checkpoint(model, state_dict, layer_idx=-1)

    assert set(converted) == {"hf.non_layer.weight"}
    assert torch.equal(converted["hf.non_layer.weight"], torch.ones(1))


def test_preprocess_layer_checkpoint_uses_custom_non_layer_conversion_for_prime_non_layer_shard() -> None:
    model = DummyCustomNonLayerPrimeModel()
    state_dict = {"model.mtp.layers.0.weight": torch.ones(1)}

    converted = preprocess_layer_checkpoint(model, state_dict, layer_idx=-1)

    assert set(converted) == {"mtp.layers.0.weight"}
    assert torch.equal(converted["mtp.layers.0.weight"], torch.ones(1))


def test_preprocess_layer_quantized_skips_non_prime_non_layer_conversion() -> None:
    model = DummyDefaultNonLayerModel()
    state_dict = {"weight": torch.ones(1)}

    converted = preprocess_layer_quantized(model, state_dict, layer_idx=-1)

    assert converted is state_dict
    assert set(converted) == {"weight"}
    assert torch.equal(converted["weight"], torch.ones(1))


def test_preprocess_layer_quantized_converts_prime_non_layer_with_default_handler() -> None:
    model = DummyDefaultNonLayerModel()
    state_dict = {"prime.non_layer.weight": torch.ones(1)}

    converted = preprocess_layer_quantized(model, state_dict, layer_idx=-1)

    assert set(converted) == {"hf.non_layer.weight"}
    assert torch.equal(converted["hf.non_layer.weight"], torch.ones(1))


def test_preprocess_layer_quantized_uses_custom_non_layer_conversion_for_prime_non_layer_shard() -> None:
    model = DummyCustomNonLayerPrimeModel()
    state_dict = {"model.mtp.layers.0.weight": torch.ones(1)}

    converted = preprocess_layer_quantized(model, state_dict, layer_idx=-1)

    assert set(converted) == {"mtp.layers.0.weight"}
    assert torch.equal(converted["mtp.layers.0.weight"], torch.ones(1))
