import asyncio
from types import SimpleNamespace

import numpy as np

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ChatCompletionScheduler


class _DummyToolManager:
    def get_prompt(self, base_chat, tokenizer, mode="initial", add_generation_prompt=False):
        return "<|im_start|>system\nSYSTEM_PROMPT<|im_end|>"


class _DummyEnv:
    def __init__(self):
        self.tool_manager = _DummyToolManager()


class _RecordingCallback:
    def __init__(self):
        self.tokenizer = object()
        self.recorded_n = None

    def postprocess(self, batch, batch_conversations, n):
        self.recorded_n = n
        return DataProto(non_tensor_batch={"__dummy__": np.arange(len(batch_conversations))})


def _build_batch(batch_size=2, validate=False):
    raw_prompt = np.array(
        [
            np.array([{"role": "user", "content": f"q-{i}"}], dtype=object)
            for i in range(batch_size)
        ],
        dtype=object,
    )
    return DataProto(non_tensor_batch={"raw_prompt": raw_prompt}, meta_info={"validate": validate})


def _build_scheduler(n=3):
    scheduler = ChatCompletionScheduler.__new__(ChatCompletionScheduler)
    scheduler.config = SimpleNamespace(
        temperature=0.7,
        top_p=0.95,
        n=n,
        val_kwargs=SimpleNamespace(top_p=0.5, temperature=0.0),
    )
    scheduler.model_name = "dummy/model"
    scheduler.env_object = _DummyEnv()
    scheduler.completion_callback = _RecordingCallback()
    scheduler._submit_calls = []

    async def _fake_submit(messages, request_id, sampling_params):
        scheduler._submit_calls.append(dict(sampling_params))

    scheduler._submit_chat_completions_semaphore = _fake_submit
    return scheduler


def test_generate_sequences_uses_rollout_n_for_training():
    scheduler = _build_scheduler(n=3)
    batch = _build_batch(batch_size=2, validate=False)

    result = asyncio.run(scheduler.generate_sequences(batch))

    assert scheduler.completion_callback.recorded_n == 3
    assert len(scheduler._submit_calls) == 6
    assert len(result) == 6


def test_generate_sequences_accepts_sampling_overrides():
    scheduler = _build_scheduler(n=2)
    batch = _build_batch(batch_size=1, validate=False)

    asyncio.run(scheduler.generate_sequences(batch, temperature=0.2, top_p=0.3))

    assert len(scheduler._submit_calls) == 2
    assert scheduler._submit_calls[0]["temperature"] == 0.2
    assert scheduler._submit_calls[0]["top_p"] == 0.3


def test_generate_sequences_force_single_sample_in_validation():
    scheduler = _build_scheduler(n=4)
    batch = _build_batch(batch_size=2, validate=True)

    asyncio.run(scheduler.generate_sequences(batch))

    assert scheduler.completion_callback.recorded_n == 1
    assert len(scheduler._submit_calls) == 2
    assert scheduler._submit_calls[0]["temperature"] == 0.0
    assert scheduler._submit_calls[0]["top_p"] == 0.5
