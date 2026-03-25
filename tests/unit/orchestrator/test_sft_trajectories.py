from unittest.mock import MagicMock

import verifiers as vf

from prime_rl.orchestrator.trajectories import interleave_rollout


def test_interleave_rollout_missing_tokens_returns_none():
    output = vf.RolloutOutput(
        example_id=1,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            )
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    assert interleave_rollout(output) is None


def test_interleave_multi_step_with_tokens():
    """Tokens are provided directly (as RendererClient would during rollout)."""
    output = vf.RolloutOutput(
        example_id=42,
        trajectory=[
            vf.TrajectoryStep(
                prompt=[{"role": "user", "content": "U1"}],
                completion=[{"role": "assistant", "content": "A1"}],
                response=MagicMock(),
                tokens={
                    "prompt_ids": [1, 2, 3],
                    "prompt_mask": [0, 0, 0],
                    "completion_ids": [4, 5],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.1, -0.2],
                },
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
            vf.TrajectoryStep(
                prompt=[
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                ],
                completion=[{"role": "assistant", "content": "A2"}],
                response=MagicMock(),
                tokens={
                    "prompt_ids": [1, 2, 3, 4, 5, 6, 7],
                    "prompt_mask": [0, 0, 0, 0, 0, 0, 0],
                    "completion_ids": [8, 9],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-0.3, -0.4],
                },
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id="1",
                extras={},
            ),
        ],
        sampling_args={"temperature": 1.0},
        error=None,
    )

    rollouts = interleave_rollout(output)
    assert rollouts is not None
    assert len(rollouts) == 1

    rollout = rollouts[0]
    # Step 1: prompt=[1,2,3], completion=[4,5]
    # Step 2: prompt=[1,2,3,4,5,6,7] extends step 1 full=[1,2,3,4,5]
    #   new prompt tokens: [6,7], new completion: [8,9]
    assert rollout.prompt_ids == [1, 2, 3]
    assert rollout.completion_ids == [4, 5, 6, 7, 8, 9]
    assert rollout.completion_mask == [True, True, False, False, True, True]
    assert rollout.completion_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
