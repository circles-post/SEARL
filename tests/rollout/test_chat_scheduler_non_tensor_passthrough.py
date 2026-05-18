import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.rollout.chat_scheduler import ToolCompletionCallback


class DummyTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, tokenize=False):
        text = "".join(msg.get("content", "") for msg in messages)
        if add_generation_prompt:
            text = text + "<gen>"
        return text

    def __call__(self, texts, return_tensors="pt", padding="longest", padding_side="left"):
        tokenized = []
        for text in texts:
            tokenized.append([idx + 1 for idx, _ in enumerate(text)] or [1])

        max_len = max(len(ids) for ids in tokenized)
        padded_ids = []
        padded_mask = []
        for ids in tokenized:
            pad_len = max_len - len(ids)
            pad = [0] * pad_len
            if padding_side == "left":
                padded = pad + ids
                mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids + pad
                mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            padded_mask.append(mask)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        }


def test_postprocess_keeps_tool_call_count():
    callback = ToolCompletionCallback.__new__(ToolCompletionCallback)
    callback.tokenizer = DummyTokenizer()
    callback._tool_schemas = []
    callback._mask_out_tools_calling_tokens = (
        lambda raw_prompts, conversations, input_ids, attention_mask: attention_mask.to(torch.float32)
    )

    prompt = np.empty((1,), dtype=object)
    prompt[0] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    tool_call_count = np.array([2], dtype=object)
    batch = DataProto(non_tensor_batch={"raw_prompt": prompt, "tool_call_count": tool_call_count})

    batch_conversations = [
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    ]

    output = ToolCompletionCallback.postprocess(callback, batch, batch_conversations, n=1)

    assert "tool_call_count" in output.non_tensor_batch
    np.testing.assert_array_equal(output.non_tensor_batch["tool_call_count"], tool_call_count)
