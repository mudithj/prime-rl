import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.server import setup_vllm_env


def test_setup_vllm_env_disables_deep_gemm(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)

    setup_vllm_env(InferenceConfig(use_deep_gemm=False, enable_lora=False))

    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
    assert os.environ.get("VLLM_USE_DEEP_GEMM") == "0"
    assert os.environ.get("VLLM_MOE_USE_DEEP_GEMM") == "0"
    assert os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING") is None


def test_setup_vllm_env_enables_deep_gemm_and_lora(monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", raising=False)

    setup_vllm_env(InferenceConfig(use_deep_gemm=True, enable_lora=True))

    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
    assert os.environ.get("VLLM_USE_DEEP_GEMM") == "1"
    assert os.environ.get("VLLM_MOE_USE_DEEP_GEMM") == "1"
    assert os.environ.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING") == "True"
