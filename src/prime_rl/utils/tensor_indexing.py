import torch

_MAX_INT32_INDEX = torch.iinfo(torch.int32).max


def get_index_dtype_for_numel(numel: int) -> torch.dtype:
    """Return the narrowest signed integer dtype that can index a tensor of length ``numel``."""
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if numel <= _MAX_INT32_INDEX + 1:
        return torch.int32
    return torch.int64
