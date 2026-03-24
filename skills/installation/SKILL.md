---
name: installation
description: How to install prime-rl and its optional dependencies. Use when setting up the project, installing extras like deep-gemm for FP8 models, or troubleshooting dependency issues.
---

# Installation

## Basic install

```bash
uv sync
```

This installs all core dependencies defined in `pyproject.toml`.

## All extras at once

The recommended way to install for most users:

```bash
uv sync --all-extras
```

This installs all optional extras (flash-attn, flash-attn-cute, etc.) in one go.

## Mamba-SSM (NemotronH models)

For NemotronH (hybrid Mamba-Transformer-MoE) models, install `mamba-ssm` for Triton-based SSD kernels that match vLLM's precision:

```bash
CUDA_HOME=/usr/local/cuda uv pip install mamba-ssm
```

Requires `nvcc` (CUDA toolkit). Without `mamba-ssm`, NemotronH falls back to HF's pure-PyTorch implementation which computes softplus in bf16, causing ~0.4 KL divergence vs vLLM.

Note: do NOT install `causal-conv1d` unless your GPU architecture matches the compiled CUDA kernels. The code automatically falls back to PyTorch nn.Conv1d when it's absent.

## FP8 inference with deep-gemm

For certain models like GLM-5-FP8, you need `deep-gemm`. Install it via the `fp8-inference` dependency group:

```bash
uv sync --group fp8-inference
```

This installs the pre-built `deep-gemm` wheel. No CUDA build step is needed.

## Dev dependencies

```bash
uv sync --group dev
```

Installs pytest, ruff, pre-commit, and other development tools.

## Key files

- `pyproject.toml` — all dependencies, extras, and dependency groups
- `uv.lock` — pinned lockfile (update with `uv sync --all-extras`)
