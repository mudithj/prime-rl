import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle


class DeepEPExpertParallel(ParallelStyle):
    """Expert-parallel style backed by DeepEP dispatch/combine.

    Shards weights (Shard(0) on expert dim), stores the EP process group, and
    registers an output hook that runs DeepEP combine + sync. Dispatch stays
    in `MoE.forward()` so it remains outside the selective-AC checkpoint on
    `_run_routed_experts`; `MoE` sets `_deepep_dispatch_state` on this module
    before the local expert forward.
    """

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))
        mod._ep_group = device_mesh.get_group()

    @staticmethod
    def _combine_output_fn(mod: nn.Module, outputs: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
        state = getattr(mod, "_deepep_dispatch_state", None)
        if state is None:
            return outputs
        from prime_rl.trainer.distributed.deepep import DispatchState, combine_tokens, sync_combine

        if not isinstance(state, DispatchState):
            raise TypeError(f"Expected DispatchState on {type(mod).__name__}._deepep_dispatch_state, got {type(state)}")
        out = combine_tokens(outputs, state)
        sync_combine()
        del mod._deepep_dispatch_state
        return out

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            output_fn=self._combine_output_fn,
        )


def get_ep_group(experts: nn.Module) -> ProcessGroup:
    return experts._ep_group
