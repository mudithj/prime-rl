import pickle
from typing import TYPE_CHECKING, Callable, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint,
    load_weights_kernel,
    postprocess_weights_checkpoint,
    postprocess_weights_kernel,
)
from prime_rl.utils.tensor_indexing import get_index_dtype_for_numel

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")

BROADCAST_MODE_FULL = 0
BROADCAST_MODE_DELTA = 1


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def _receive_full_layer(
    communicator: PyNcclCommunicator,
    buffers: dict[tuple[int, int], torch.Tensor] | None = None,
    layer_idx: int = 0,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Receive a full layer state dict over NCCL.

    When buffers is provided, caches the concatenated tensors keyed by
    (layer_idx, group_idx) for later delta application.
    """
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)
    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    for group_idx, (dtype, tensor_info_list) in enumerate(metadata.items()):
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)

        if buffers is not None:
            buffers[(layer_idx, group_idx)] = concatenated.clone()

        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            yield key, tensor
            del tensor
        del concatenated


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        delta_compression: bool = False,
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)
        self.delta_compression = delta_compression
        self._buffers: dict[tuple[int, int], torch.Tensor] = {}

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_layers = receive_integer(self.communicator)
        logger.info(f"Receiving {num_layers} layer state dicts")

        for layer_idx in range(num_layers):
            logger.info(f"Receiving state dict {layer_idx + 1}/{num_layers}")

            if not self.delta_compression:
                yield from _receive_full_layer(self.communicator)
                continue

            mode = receive_integer(self.communicator)
            if mode == BROADCAST_MODE_FULL:
                yield from _receive_full_layer(self.communicator, self._buffers, layer_idx)
            else:
                yield from self._receive_delta(layer_idx)

    def _receive_delta(self, layer_idx: int) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Receive delta-compressed update and apply to cached buffers."""
        size_tensor = torch.tensor([10], dtype=torch.long).to(self.communicator.device)
        self.communicator.broadcast(size_tensor, src=0)
        state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(self.communicator.device)
        self.communicator.broadcast(state_tensor, src=0)
        metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

        total_changed = 0
        total_elements = 0

        for group_idx, (dtype, tensor_info_list) in enumerate(metadata.items()):
            group_elements = sum(numel for _, _, numel in tensor_info_list)
            total_elements += group_elements

            num_changed_tensor = torch.tensor([0], dtype=torch.long, device=self.communicator.device)
            self.communicator.broadcast(num_changed_tensor, src=0)
            num_changed = cast(int, num_changed_tensor.item())
            total_changed += num_changed

            buffer = self._buffers[(layer_idx, group_idx)]

            if num_changed > 0:
                index_dtype = get_index_dtype_for_numel(group_elements)
                indices = torch.empty(num_changed, dtype=index_dtype, device=self.communicator.device)
                self.communicator.broadcast(indices, src=0)
                values = torch.empty(num_changed, dtype=dtype, device=self.communicator.device)
                self.communicator.broadcast(values, src=0)
                buffer[indices.long()] = values
                del indices, values

            offset = 0
            for key, shape, numel in tensor_info_list:
                tensor = buffer[offset : offset + numel].view(shape).clone()
                offset += numel
                yield key, tensor
                del tensor

        if total_elements > 0:
            sparsity = 1.0 - total_changed / total_elements
            logger.info(
                f"Delta received layer {layer_idx}: {total_changed:,}/{total_elements:,} changed ({sparsity:.2%} sparsity)"
            )


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        gpus_per_server: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
        delta_compression: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
            gpus_per_server: Number of GPUs managed by this server instance.
        """
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        global_rank_inference = rank_offset + self.local_rank

        logger.info(
            f"Worker [local_rank={self.local_rank} rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
            delta_compression=delta_compression,
        )

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        device = next(model.parameters()).device
        loader_fn: Callable[[Module, Generator[tuple[str, torch.Tensor], None, None]], None]
        postprocess_fn: Callable[[Module, object, torch.device], None]
        if self.quantize_in_weight_transfer:
            loader_fn = load_weights_kernel
            postprocess_fn = postprocess_weights_kernel
        else:
            loader_fn = load_weights_checkpoint
            postprocess_fn = postprocess_weights_checkpoint

        loader_fn(model, state_iter)
        postprocess_fn(model, self.model_runner.model_config, device)
