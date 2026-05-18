import math

import torch
from tensordict import TensorDict

from envs.base import Env
from verl import DataProto


class DummyEnv:
    def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data):
        raise NotImplementedError

    def _compute_score_with_rules(self, data, tokenizer, if_val=False, mcp_worthiness=None, tool_call_count=None):
        return [[1.0], [2.0]]


def _build_data(with_tool_use_scores: bool) -> DataProto:
    batch_data = {
        "prompts": torch.tensor([[1], [1]], dtype=torch.long),
    }
    if with_tool_use_scores:
        batch_data["tool_use_scores"] = torch.tensor(
            [[float("nan"), 0.2], [0.3, float("nan")]], dtype=torch.float32
        )
    batch = TensorDict(batch_data, batch_size=[2])
    return DataProto(batch=batch, non_tensor_batch={})


def test_compute_score_falls_back_when_tool_use_scores_missing():
    env = DummyEnv()
    data = _build_data(with_tool_use_scores=False)
    scores = Env.compute_score(
        env,
        reward_rollout_wg=None,
        reward_tokenizer=None,
        tokenizer=None,
        data=data,
        if_val=False,
        use_process_reward=True,
        mcp_worthiness=None,
        tool_call_count=None,
    )
    assert scores == [[1.0], [2.0]]


def test_compute_score_uses_tool_use_scores_when_present():
    env = DummyEnv()
    data = _build_data(with_tool_use_scores=True)
    scores = Env.compute_score(
        env,
        reward_rollout_wg=None,
        reward_tokenizer=None,
        tokenizer=None,
        data=data,
        if_val=False,
        use_process_reward=True,
        mcp_worthiness=None,
        tool_call_count=None,
    )
    assert len(scores) == 2
    assert len(scores[0]) == 2
    assert len(scores[1]) == 2
    assert math.isclose(scores[0][0], 0.2, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(scores[0][1], 1.0, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(scores[1][0], 0.3, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(scores[1][1], 2.0, rel_tol=1e-6, abs_tol=1e-6)
