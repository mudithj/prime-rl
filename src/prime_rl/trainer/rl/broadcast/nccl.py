import pickle
import time
from pathlib import Path
from threading import Thread
from typing import Callable, Generator, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.tensor_indexing import get_index_dtype_for_numel
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix

NCCL_READY_MARKER = "NCCL_READY"

BROADCAST_MODE_FULL = 0
BROADCAST_MODE_DELTA = 1

_INT_DTYPE_BY_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _int_view(tensor: Tensor) -> Tensor:
    """Reinterpret a tensor as its matching-width integer dtype for bit-exact comparison."""
    return tensor.view(_INT_DTYPE_BY_SIZE[tensor.element_size()])


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def _build_dtype_groups(state_dict: dict[str, Tensor]) -> dict[torch.dtype, list[tuple[str, Tensor]]]:
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        if value.dtype not in dtype_groups:
            dtype_groups[value.dtype] = []
        dtype_groups[value.dtype].append((key, value))
    return dtype_groups


def _broadcast_metadata(
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]], communicator: PyNcclCommunicator
) -> None:
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)


def broadcast_state_dict(
    state_dict: dict[str, Tensor],
    communicator: PyNcclCommunicator,
    *,
    cache: bool = False,
) -> dict[int, Tensor] | None:
    """Broadcast a state dict via NCCL.

    When cache=True, returns a dict mapping dtype-group index to CPU-resident
    concatenated tensors (used as the baseline for subsequent delta broadcasts).
    """
    dtype_groups = _build_dtype_groups(state_dict)
    _broadcast_metadata(dtype_groups, communicator)

    cached: dict[int, Tensor] | None = {} if cache else None
    for group_idx, (dtype, items) in enumerate(dtype_groups.items()):
        concatenated = torch.cat([value.flatten() for _, value in items])
        if cached is not None:
            cached[group_idx] = concatenated.cpu()
        communicator.broadcast(concatenated, src=0)
        del concatenated
    return cached


def _broadcast_delta(
    state_dict: dict[str, Tensor],
    prev_cache: dict[int, Tensor],
    communicator: PyNcclCommunicator,
) -> tuple[dict[int, Tensor], int, int, int, int]:
    """Broadcast delta-compressed state dict.

    Returns:
        Tuple of (new_cache, total_elements, total_changed, full_bytes, delta_bytes).
    """
    dtype_groups = _build_dtype_groups(state_dict)
    _broadcast_metadata(dtype_groups, communicator)

    new_cache: dict[int, Tensor] = {}
    total_elements = 0
    total_changed = 0
    total_full_bytes = 0
    total_delta_bytes = 0

    for group_idx, (dtype, items) in enumerate(dtype_groups.items()):
        current_concat = torch.cat([value.flatten() for _, value in items])
        prev_concat = prev_cache[group_idx].to(current_concat.device)
        index_dtype = get_index_dtype_for_numel(current_concat.numel())

        changed_mask = _int_view(current_concat) != _int_view(prev_concat)
        indices = changed_mask.nonzero(as_tuple=True)[0].to(index_dtype)
        values = current_concat[indices.long()]

        total_elements += current_concat.numel()
        total_changed += indices.numel()
        total_full_bytes += current_concat.numel() * current_concat.element_size()
        total_delta_bytes += values.numel() * values.element_size()
        total_delta_bytes += indices.numel() * indices.element_size()

        num_changed_tensor = torch.tensor([indices.numel()], dtype=torch.long).cuda()
        communicator.broadcast(num_changed_tensor, src=0)
        if indices.numel() > 0:
            communicator.broadcast(indices, src=0)
            communicator.broadcast(values, src=0)

        new_cache[group_idx] = current_concat.cpu()
        del current_concat, prev_concat, changed_mask, indices, values

    return new_cache, total_elements, total_changed, total_full_bytes, total_delta_bytes


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int, layer_prefix: str
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield non-layer weights first, then each layer's weights.

    Yields (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    yield -1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}

    for i in range(num_layers):
        yield (
            i,
            {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")},
        )


def preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


def preprocess_layer_quantized(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return layer_state_dict
    return model.convert_layer_to_vllm_kernel(layer_state_dict, layer_idx, quantize_fp8=True)


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
        quantize_in_weight_transfer: bool = False,
        delta_compression: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        self.delta_compression = delta_compression
        self._layer_cache: dict[int, dict[int, Tensor]] = {}

        if self.world.is_master:
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast model weights. All ranks participate in DTensor gather; only master sends via NCCL."""
        gathered = self.gather_all_layers(model)
        if self.world.is_master:
            self.send_all_layers(gathered, step)

    @torch.no_grad()
    def gather_all_layers(self, model: nn.Module) -> list[dict[str, Tensor]]:
        """Gather and preprocess all layers (collective, all ranks).

        Returns list of CPU state dicts on master, empty list on non-master.
        """
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)

        preprocess_fn: Callable[[nn.Module, dict[str, Tensor], int], dict[str, Tensor]]
        if self.quantize_in_weight_transfer:
            preprocess_fn = preprocess_layer_quantized
        else:
            preprocess_fn = preprocess_layer_checkpoint

        gathered: list[dict[str, Tensor]] = []
        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            layer_state_dict = self._resolve_dtensors(layer_state_dict)
            layer_state_dict = preprocess_fn(model, layer_state_dict, layer_id)
            if self.world.is_master:
                gathered.append({k: v.cpu() for k, v in layer_state_dict.items()})
            del layer_state_dict
        del state_dict
        return gathered

    @torch.no_grad()
    def send_all_layers(self, gathered: list[dict[str, Tensor]], step: int) -> None:
        """Send previously gathered layers via NCCL. Master only."""
        broadcast_integer(len(gathered), self.communicator)

        total_elements = 0
        total_changed = 0
        total_full_bytes = 0
        total_delta_bytes = 0

        for seq_idx, cpu_dict in enumerate(gathered):
            gpu_dict = {k: v.cuda() for k, v in cpu_dict.items()}
            if self.delta_compression:
                n_elem, n_changed, full_bytes, delta_bytes = self._send_delta_aware(gpu_dict, seq_idx)
                total_elements += n_elem
                total_changed += n_changed
                total_full_bytes += full_bytes
                total_delta_bytes += delta_bytes
            else:
                broadcast_state_dict(gpu_dict, self.communicator)
            del gpu_dict

        del gathered
        torch.cuda.empty_cache()

        if self.delta_compression and total_elements > 0:
            self._log_delta_stats(step, total_elements, total_changed, total_full_bytes, total_delta_bytes)

    def _send_delta_aware(self, state_dict: dict[str, Tensor], seq_idx: int) -> tuple[int, int, int, int]:
        """Send full (first time) or delta (subsequent).

        Returns:
            Tuple of (total_elements, changed_elements, full_bytes, delta_bytes).
        """
        prev = self._layer_cache.get(seq_idx)
        if prev is None:
            broadcast_integer(BROADCAST_MODE_FULL, self.communicator)
            self._layer_cache[seq_idx] = broadcast_state_dict(state_dict, self.communicator, cache=True)
            return 0, 0, 0, 0

        broadcast_integer(BROADCAST_MODE_DELTA, self.communicator)
        new_cache, n_total, n_changed, full_bytes, delta_bytes = _broadcast_delta(state_dict, prev, self.communicator)
        self._layer_cache[seq_idx] = new_cache
        return n_total, n_changed, full_bytes, delta_bytes

    def _log_delta_stats(
        self, step: int, total_elements: int, total_changed: int, total_full_bytes: int, total_delta_bytes: int
    ) -> None:
        sparsity = 1.0 - total_changed / total_elements
        full_gb = total_full_bytes / 1e9
        delta_gb = total_delta_bytes / 1e9
        self.logger.info(
            f"Delta broadcast step {step}: {total_changed:,}/{total_elements:,} elements changed "
            f"({sparsity:.2%} sparsity, ~{delta_gb:.3f} GB delta vs ~{full_gb:.3f} GB full)"
        )

    def _resolve_dtensors(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                state_dict[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return state_dict


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL.

    When delta_compression is enabled, the broadcast is pipelined: the DTensor
    gather runs synchronously (all trainer ranks), then the NCCL send runs in a
    background thread on the master rank so training can continue.
    """

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.pipeline = config.delta_compression
        self.sender = NCCLWeightBroadcastSender(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            dtype,
            quantize_in_weight_transfer=config.quantize_in_weight_transfer,
            delta_compression=config.delta_compression,
        )
        self._bg_thread: Thread | None = None
        self._bg_error: Exception | None = None

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()

        self._join_background()

        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            notified_runs = self._notify_orchestrator()

        # All ranks participate in the collective gather
        gathered = self.sender.gather_all_layers(model)

        if self.pipeline and self.world.is_master:
            self.logger.debug(f"Weights gathered in {time.perf_counter() - start_time:.2f}s, starting pipelined send")
            self._bg_thread = Thread(
                target=self._background_send,
                args=(gathered, notified_runs, step, start_time),
                daemon=True,
            )
            self._bg_thread.start()
        elif self.world.is_master:
            self._wait_for_nccl_ready(notified_runs)
            self.sender.send_all_layers(gathered, step)
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _background_send(
        self,
        gathered: list[dict[str, Tensor]],
        notified_runs: list[tuple[int, Path]],
        step: int,
        start_time: float,
    ) -> None:
        try:
            self._wait_for_nccl_ready(notified_runs)
            self.sender.send_all_layers(gathered, step)
        except Exception as error:
            self._bg_error = error
            self.logger.exception("Pipelined NCCL broadcast failed")
        else:
            self.logger.debug(f"Pipelined broadcast completed in {time.perf_counter() - start_time:.2f}s (wall)")

    def _join_background(self) -> None:
        """Wait for a previous pipelined broadcast to finish, re-raising errors."""
        if self._bg_thread is not None:
            self._bg_thread.join()
            self._bg_thread = None
            if self._bg_error is not None:
                error = self._bg_error
                self._bg_error = None
                raise RuntimeError("Background NCCL broadcast failed") from error

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
        """Notify the orchestrator to initiate weight broadcast.

        Returns:
            List of (run_idx, save_dir) tuples for runs that were notified.
        """
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            for idx in self.multi_run_manager.used_idxs:
                if not self.multi_run_manager.ready_to_update[idx]:
                    continue

                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    stable_file = save_dir / "STABLE"
                    stable_file.touch()
                    notified_runs.append((idx, save_dir))
                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False
        return notified_runs

    def _wait_for_nccl_ready(self, notified_runs: list[tuple[int, Path]]):
        """Wait for inference workers to signal they are ready to receive NCCL broadcast."""
        for idx, save_dir in notified_runs:
            nccl_ready_file = save_dir / NCCL_READY_MARKER
            self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
            sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
            self.logger.debug(f"Inference workers ready for NCCL broadcast (run {idx})")
