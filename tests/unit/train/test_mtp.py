import torch

from prime_rl.trainer.mtp import _shift_left


def test_shift_left_keeps_cp_collectives_symmetric(monkeypatch):
    cp_group = object()
    local_position_ids = [
        torch.tensor([[0, 1]]),
        torch.tensor([[2, 3]]),
    ]
    local_values = [
        torch.tensor([[10, 11]]),
        torch.tensor([[12, 13]]),
    ]
    position_first_values = [position_ids[..., :1].clone() for position_ids in local_position_ids]
    value_first_values = [values[..., :1].clone() for values in local_values]
    current_rank = {"value": 0}
    all_gather_counts = {0: 0, 1: 0}

    def fake_get_world_size(*, group):
        assert group is cp_group
        return 2

    def fake_get_rank(*, group):
        assert group is cp_group
        return current_rank["value"]

    def fake_all_gather(output_tensors, input_tensor, *, group):
        assert group is cp_group
        rank = current_rank["value"]
        all_gather_counts[rank] += 1
        if torch.equal(input_tensor, position_first_values[rank]):
            source_tensors = position_first_values
        elif torch.equal(input_tensor, value_first_values[rank]):
            source_tensors = value_first_values
        else:
            raise AssertionError(f"Unexpected all_gather input for rank {rank}: {input_tensor}")

        for output_tensor, source_tensor in zip(output_tensors, source_tensors, strict=True):
            output_tensor.copy_(source_tensor)

    monkeypatch.setattr("prime_rl.trainer.mtp.dist.get_world_size", fake_get_world_size)
    monkeypatch.setattr("prime_rl.trainer.mtp.dist.get_rank", fake_get_rank)
    monkeypatch.setattr("prime_rl.trainer.mtp.dist.all_gather", fake_all_gather)

    shifted_outputs = []
    for rank in range(2):
        current_rank["value"] = rank
        shifted_outputs.append(_shift_left(local_values[rank], local_position_ids[rank], cp_group=cp_group))

    assert all_gather_counts == {0: 2, 1: 2}
    torch.testing.assert_close(shifted_outputs[0], torch.tensor([[11, 12]]))
    torch.testing.assert_close(shifted_outputs[1], torch.tensor([[13, 0]]))
