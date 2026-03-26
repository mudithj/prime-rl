import pickle

import pytest
import torch

from prime_rl.inference.vllm.worker import nccl as worker_nccl
from prime_rl.trainer.rl.broadcast import nccl as trainer_nccl
from prime_rl.utils.tensor_indexing import get_index_dtype_for_numel


class RecordingCommunicator:
    def __init__(self, device: torch.device = torch.device("cpu")) -> None:
        self.broadcasted: list[torch.Tensor] = []
        self.device = device

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        del src
        self.broadcasted.append(tensor.clone())


class ReplayCommunicator:
    def __init__(self, payloads: list[torch.Tensor]) -> None:
        self.payloads = iter(payloads)
        self.device = torch.device("cpu")
        self.requested_dtypes: list[torch.dtype] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        del src
        self.requested_dtypes.append(tensor.dtype)
        payload = next(self.payloads)
        tensor.copy_(payload)


def test_get_index_dtype_for_numel_promotes_past_int32_limit() -> None:
    max_int32 = torch.iinfo(torch.int32).max

    assert get_index_dtype_for_numel(max_int32 + 1) == torch.int32
    assert get_index_dtype_for_numel(max_int32 + 2) == torch.int64


def test_get_index_dtype_for_numel_rejects_negative_sizes() -> None:
    with pytest.raises(ValueError, match="numel must be non-negative"):
        get_index_dtype_for_numel(-1)


def test_broadcast_integer_uses_communicator_device() -> None:
    communicator = RecordingCommunicator(device=torch.device("meta"))

    trainer_nccl.broadcast_integer(7, communicator)

    assert len(communicator.broadcasted) == 1
    assert communicator.broadcasted[0].device == communicator.device
    assert communicator.broadcasted[0].dtype == torch.long


def test_broadcast_metadata_uses_communicator_device() -> None:
    communicator = RecordingCommunicator(device=torch.device("meta"))
    dtype_groups = {torch.float32: [("weight", torch.ones(3, dtype=torch.float32))]}

    trainer_nccl._broadcast_metadata(dtype_groups, communicator)

    assert len(communicator.broadcasted) == 2
    assert communicator.broadcasted[0].device == communicator.device
    assert communicator.broadcasted[0].dtype == torch.long
    assert communicator.broadcasted[1].device == communicator.device
    assert communicator.broadcasted[1].dtype == torch.uint8


def test_broadcast_delta_uses_selected_index_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        trainer_nccl,
        "_broadcast_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        trainer_nccl,
        "get_index_dtype_for_numel",
        lambda _numel: torch.int64,
    )
    monkeypatch.setattr(
        torch.Tensor,
        "cuda",
        lambda self, *args, **kwargs: self,
        raising=False,
    )

    communicator = RecordingCommunicator(device=torch.device("meta"))
    state_dict = {"weight": torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32)}
    prev_cache = {0: torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}

    _, total_elements, total_changed, full_bytes, delta_bytes = trainer_nccl._broadcast_delta(
        state_dict, prev_cache, communicator
    )

    assert total_elements == 3
    assert total_changed == 1
    assert full_bytes == 3 * torch.tensor([], dtype=torch.float32).element_size()
    assert delta_bytes == (
        torch.tensor([], dtype=torch.float32).element_size() + torch.tensor([], dtype=torch.int64).element_size()
    )
    assert communicator.broadcasted[0].device == communicator.device
    assert communicator.broadcasted[1].dtype == torch.int64


def test_send_all_layers_moves_tensors_to_communicator_device(monkeypatch: pytest.MonkeyPatch) -> None:
    sender = object.__new__(trainer_nccl.NCCLWeightBroadcastSender)
    sender.communicator = RecordingCommunicator(device=torch.device("meta"))
    sender.delta_compression = False

    seen_devices: list[dict[str, torch.device]] = []

    monkeypatch.setattr(trainer_nccl, "broadcast_integer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    def fake_broadcast_state_dict(
        state_dict: dict[str, torch.Tensor],
        communicator: RecordingCommunicator,
        *,
        cache: bool = False,
    ) -> None:
        del communicator, cache
        seen_devices.append({key: value.device for key, value in state_dict.items()})

    monkeypatch.setattr(trainer_nccl, "broadcast_state_dict", fake_broadcast_state_dict)

    sender.send_all_layers([{"weight": torch.tensor([1.0], dtype=torch.float32)}], step=0)

    assert seen_devices == [{"weight": torch.device("meta")}]


def test_receive_delta_uses_selected_index_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_nccl, "get_index_dtype_for_numel", lambda _numel: torch.int64)

    metadata = {torch.float32: [("weight", (3,), 3)]}
    state_bytes = pickle.dumps(metadata)
    communicator = ReplayCommunicator(
        [
            torch.tensor([len(state_bytes)], dtype=torch.long),
            torch.tensor(list(state_bytes), dtype=torch.uint8),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([5.0], dtype=torch.float32),
        ]
    )

    receiver = object.__new__(worker_nccl.NCCLWeightBroadcastReceiver)
    receiver.communicator = communicator
    receiver._buffers = {
        (0, 0): torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }

    received = list(worker_nccl.NCCLWeightBroadcastReceiver._receive_delta(receiver, layer_idx=0))

    assert communicator.requested_dtypes[3] == torch.int64
    assert torch.equal(
        receiver._buffers[(0, 0)],
        torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32),
    )
    assert len(received) == 1
    assert received[0][0] == "weight"
    assert torch.equal(
        received[0][1],
        torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32),
    )
