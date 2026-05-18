# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import difflib
import json
import logging
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class RollbackManager:
    """Manages rollback mechanism for tool call errors."""

    def __init__(self, enable: bool, max_retries: int, error_patterns: list[str]):
        self.enable = enable
        self.max_retries = max_retries
        self.error_patterns = error_patterns

    def can_retry(self, retry_counts: dict[str, int], position_key: str) -> bool:
        return retry_counts.get(position_key, 0) < self.max_retries

    def increment_retry(self, retry_counts: dict[str, int], position_key: str) -> int:
        current = retry_counts.get(position_key, 0)
        retry_counts[position_key] = current + 1
        return retry_counts[position_key]

    def format_error_feedback(self, error_messages: list[str], error_types: list[str]) -> str:
        _ = error_types
        return error_messages[-1]

    def create_checkpoint(self, agent_data: "AgentData") -> dict[str, Any]:
        return {
            "prompt_ids": list(agent_data.prompt_ids),
            "response_ids": list(agent_data.response_ids),
            "response_mask": list(agent_data.response_mask),
            "messages": copy.deepcopy(agent_data.messages),
            "assistant_turns": agent_data.assistant_turns,
            "user_turns": agent_data.user_turns,
        }

    def restore_checkpoint(self, agent_data: "AgentData", checkpoint: dict[str, Any]) -> None:
        agent_data.prompt_ids = checkpoint["prompt_ids"]
        agent_data.response_ids = checkpoint["response_ids"]
        agent_data.response_mask = checkpoint["response_mask"]
        agent_data.messages = checkpoint["messages"]
        agent_data.assistant_turns = checkpoint["assistant_turns"]
        agent_data.user_turns = checkpoint["user_turns"]


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(self, messages: list[dict[str, Any]], metrics: dict[str, Any], request_id: str):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id

        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.tool_calls: list[FunctionCall] = []

        self.retry_counts: dict[str, int] = defaultdict(int)
        self.total_rollbacks = 0
        self.disable_rollback_after_max_retry = False
        self.rollback_recovered_turns: set[str] = set()

        self.tool_call_total = 0
        self.tool_call_success = 0
        self.tool_failure_reasons: defaultdict[str, int] = defaultdict(int)
        self.first_attempt_total = 0
        self.first_attempt_success = 0

        self.global_rollback_triggered = 0
        self.global_rollback_recovered = 0
        self.global_rollback_failed = 0
        self.rollback_full_turn_count = 0
        self.rollback_tool_call_only_count = 0


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    ROLLBACK_SIMILARITY_THRESHOLD = 0.5

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        cls.tool_parser = ToolParser.get_tool_parser(cls.tool_parser_name, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

        enable_rollback = config.actor_rollout_ref.rollout.multi_turn.get("enable_tool_rollback", False)
        max_retries = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_retries", 3)
        error_patterns = config.actor_rollout_ref.rollout.multi_turn.get(
            "rollback_on_errors",
            [
                "ImportError",
                "ModuleNotFoundError",
                "SyntaxError",
                "IndexError",
                "IndentationError",
                "NameError",
                "TypeError",
                "worker_timeout",
                "NotImplementedError",
                "ValueError",
                "ZeroDivisionError",
            ],
        )
        cls.rollback_manager = RollbackManager(enable_rollback, max_retries, error_patterns)
        cls.rollback_probability = config.actor_rollout_ref.rollout.multi_turn.get("rollback_probability", 1.0)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        agent_data = AgentData(messages=list(messages), metrics=metrics, request_id=request_id)

        if random.random() >= self.rollback_probability:
            agent_data.disable_rollback_after_max_retry = True

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        prompt_ids = (
            agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
            if agent_data.response_mask
            else agent_data.prompt_ids
        )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=metrics,
            extra_fields={
                "tool_call_total": agent_data.tool_call_total,
                "tool_call_success": agent_data.tool_call_success,
                "tool_failure_reasons": dict(agent_data.tool_failure_reasons),
                "first_attempt_total": agent_data.first_attempt_total,
                "first_attempt_success": agent_data.first_attempt_success,
                "global_rollback_triggered": agent_data.global_rollback_triggered,
                "global_rollback_recovered": agent_data.global_rollback_recovered,
                "global_rollback_failed": agent_data.global_rollback_failed,
                "rollback_full_turn_count": agent_data.rollback_full_turn_count,
                "rollback_tool_call_only_count": agent_data.rollback_tool_call_only_count,
            },
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        _ = sampling_params
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                agent_data.messages, tools=self.tool_schemas, add_generation_prompt=True, tokenize=True
            ),
        )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            response_ids = await self.server_manager.generate(
                request_id=agent_data.request_id, prompt_ids=agent_data.prompt_ids, sampling_params=sampling_params
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [1] * len(response_ids)

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], tool_position_key: Optional[str] = None
    ) -> AgentState:
        agent_data.total_rollbacks += 1
        max_attempts_before_disable = 30
        if (
            agent_data.total_rollbacks > max_attempts_before_disable
            and not agent_data.disable_rollback_after_max_retry
        ):
            logger.warning(
                f"Total tool attempts ({agent_data.total_rollbacks}) exceeded {max_attempts_before_disable}. "
                "Disabling rollback to prevent infinite loops."
            )
            agent_data.disable_rollback_after_max_retry = True

        if tool_position_key is None:
            tool_position_key = f"turn_{agent_data.assistant_turns}"

        is_first_attempt = agent_data.retry_counts.get(tool_position_key, 0) == 0
        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data, is_first_attempt=is_first_attempt))
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        error_messages, error_types = self._detect_errors(responses)
        is_retrying = agent_data.retry_counts.get(tool_position_key, 0) > 0

        if self.rollback_manager.enable and (not agent_data.disable_rollback_after_max_retry or is_retrying):
            if error_messages:
                if agent_data.retry_counts.get(tool_position_key, 0) == 0:
                    agent_data.global_rollback_triggered += 1
                if not self.rollback_manager.can_retry(agent_data.retry_counts, tool_position_key):
                    agent_data.global_rollback_failed += 1
                    return await self._handle_max_retry_exceeded(
                        agent_data, error_messages, error_types, sampling_params
                    )
                checkpoint = self.rollback_manager.create_checkpoint(agent_data)
                rollback_result = await self._handle_rollback(
                    agent_data, checkpoint, tool_position_key, error_messages, error_types, sampling_params
                )
                if rollback_result is not None:
                    return rollback_result
            else:
                if is_retrying:
                    agent_data.rollback_recovered_turns.add(tool_position_key)
                    agent_data.global_rollback_recovered += 1

        add_messages = [{"role": "tool", "content": tool_response or ""} for tool_response, _, _ in responses]
        tool_response_ids = await self.loop.run_in_executor(
            None,
            lambda messages=add_messages: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        tool_response_ids = tool_response_ids[len(self.system_prompt) :]

        if len(agent_data.response_mask) + len(tool_response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += tool_response_ids
        agent_data.response_mask += [0] * len(tool_response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, agent_data: AgentData, is_first_attempt: bool = False
    ) -> tuple[str, float, dict]:
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            instance_id = await tool.create()
            tool_response, tool_reward, tool_metrics = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"tool call error: {e}")
            error_response = (
                "Tool call failure: tool call format is wrong, please make sure to generate correct "
                "json-format tool call arguments."
            )
            self._record_tool_attempt(
                agent_data, success=False, failure_reason="tool_call_format_error", is_first_attempt=is_first_attempt
            )
            return error_response, 0.0, {}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_response or ""
        has_error, error_type = self._detect_error_from_text(tool_response_text)
        self._record_tool_attempt(
            agent_data, success=not has_error, failure_reason=error_type, is_first_attempt=is_first_attempt
        )
        if len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = (
                    tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
                )
        return tool_response_text, tool_reward, tool_metrics

    def _record_tool_attempt(
        self, agent_data: AgentData, success: bool, failure_reason: Optional[str] = None, is_first_attempt: bool = False
    ) -> None:
        agent_data.tool_call_total += 1
        if success:
            agent_data.tool_call_success += 1
        else:
            reason_key = failure_reason or "unknown_failure"
            agent_data.tool_failure_reasons[reason_key] += 1
        if is_first_attempt:
            agent_data.first_attempt_total += 1
            if success:
                agent_data.first_attempt_success += 1

    def _detect_error_from_text(self, text: str) -> tuple[bool, Optional[str]]:
        if not text:
            return False, None
        if "Tool call success" in text:
            return False, None
        for pattern in self.rollback_manager.error_patterns:
            if pattern in text:
                return True, pattern
        return True, "unknown_error"

    def _detect_errors(self, responses: list[tuple[str, float, dict]]) -> tuple[list[str], list[str]]:
        error_messages = []
        error_types = []
        for tool_response_text, _, _ in responses:
            has_error, error_type = self._detect_error_from_text(tool_response_text or "")
            if has_error and error_type:
                error_messages.append(tool_response_text or "")
                error_types.append(error_type)
        return error_messages, error_types

    async def _handle_rollback(
        self,
        agent_data: AgentData,
        checkpoint: dict[str, Any],
        tool_position_key: str,
        error_messages: list[str],
        error_types: list[str],
        sampling_params: dict[str, Any],
    ) -> Optional[AgentState]:
        if not error_messages:
            return None

        self.rollback_manager.increment_retry(agent_data.retry_counts, tool_position_key)
        error_feedback = self.rollback_manager.format_error_feedback(error_messages, error_types)
        error_message = {"role": "tool", "content": error_feedback}

        error_prompt_ids = await self._encode_error_feedback(error_message)
        agent_data.prompt_ids += error_prompt_ids
        agent_data.response_mask += [0] * len(error_prompt_ids)

        new_state = await self._handle_generating_state(agent_data, sampling_params, ignore_termination=True)
        if new_state == AgentState.TERMINATED and agent_data.tool_calls:
            agent_data.global_rollback_failed += 1
            self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
            return AgentState.TERMINATED
        if not agent_data.tool_calls:
            agent_data.global_rollback_failed += 1
            agent_data.disable_rollback_after_max_retry = True
            return new_state

        new_response_ids = list(agent_data.response_ids)
        if not new_response_ids:
            self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
            return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key)

        self._overwrite_last_assistant_turn(checkpoint, new_response_ids, error_feedback, agent_data)
        self.rollback_manager.restore_checkpoint(agent_data, checkpoint)
        return await self._handle_processing_tools_state(agent_data, sampling_params, tool_position_key)

    async def _encode_error_feedback(self, error_message: dict[str, Any]) -> list[int]:
        error_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                [error_message], add_generation_prompt=True, tokenize=True
            ),
        )
        return error_prompt_ids[len(self.system_prompt) :]

    async def _handle_max_retry_exceeded(
        self,
        agent_data: AgentData,
        error_messages: list[str],
        error_types: list[str],
        sampling_params: dict[str, Any],
    ) -> AgentState:
        _ = error_types
        failure_message = error_messages[-1]
        tool_error_message = {"role": "tool", "content": failure_message}
        agent_data.disable_rollback_after_max_retry = True

        notification_ids = await self._encode_error_feedback(tool_error_message)
        agent_data.prompt_ids += notification_ids
        agent_data.response_mask += [0] * len(notification_ids)
        agent_data.user_turns += 1

        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    def _overwrite_last_assistant_turn(
        self,
        checkpoint: dict[str, Any],
        new_response_ids: list[int],
        error_message: Optional[str],
        agent_data: Optional[AgentData] = None,
    ) -> None:
        _ = error_message
        old_response_ids = list(checkpoint.get("response_ids") or [])
        old_segment = self._split_tool_call_segment(old_response_ids)
        new_segment = self._split_tool_call_segment(new_response_ids)
        replaced_tool_call = False

        if old_segment and new_segment and old_segment["call_ids"] and new_segment["call_ids"]:
            old_call_text = old_segment["call_text"]
            new_call_text = new_segment["call_text"]
            similarity = difflib.SequenceMatcher(None, old_call_text, new_call_text).ratio()
            should_replace_reasoning = similarity < self.ROLLBACK_SIMILARITY_THRESHOLD

            if should_replace_reasoning:
                self._replace_full_turn(checkpoint, old_response_ids, new_response_ids)
                if agent_data is not None:
                    agent_data.rollback_full_turn_count += 1
            else:
                old_call_len = len(old_segment["call_ids"])
                new_call_len = len(new_segment["call_ids"])
                if old_call_len:
                    checkpoint["prompt_ids"] = checkpoint["prompt_ids"][:-old_call_len]
                    checkpoint["response_mask"] = checkpoint["response_mask"][:-old_call_len]
                checkpoint["prompt_ids"].extend(new_segment["call_ids"])
                checkpoint["response_mask"].extend([1] * new_call_len)
                checkpoint["response_ids"] = old_segment["prefix_ids"] + new_segment["call_ids"]
                if agent_data is not None:
                    agent_data.rollback_tool_call_only_count += 1
            replaced_tool_call = True

        if not replaced_tool_call:
            self._replace_full_turn(checkpoint, old_response_ids, new_response_ids)
            if agent_data is not None:
                agent_data.rollback_full_turn_count += 1

    def _replace_full_turn(
        self, checkpoint: dict[str, Any], old_response_ids: list[int], new_response_ids: list[int]
    ) -> None:
        old_response_len = len(old_response_ids)
        if old_response_len:
            checkpoint["prompt_ids"] = checkpoint["prompt_ids"][:-old_response_len]
            checkpoint["response_mask"] = checkpoint["response_mask"][:-old_response_len]
        checkpoint["prompt_ids"].extend(new_response_ids)
        checkpoint["response_mask"].extend([1] * len(new_response_ids))
        checkpoint["response_ids"] = list(new_response_ids)

    def _split_tool_call_segment(self, token_ids: list[int]) -> Optional[dict[str, Any]]:
        if not token_ids:
            return None
        tool_call_token_idx = self._find_tool_call_token_boundary(token_ids)
        if tool_call_token_idx is None:
            return None
        prefix_ids = token_ids[:tool_call_token_idx]
        call_ids = token_ids[tool_call_token_idx:]
        prefix_text = self._decode_response_text(prefix_ids)
        call_text = self._decode_response_text(call_ids)
        return {
            "prefix_ids": prefix_ids,
            "call_ids": call_ids,
            "prefix_text": prefix_text,
            "call_text": call_text,
        }

    def _find_tool_call_token_boundary(self, token_ids: list[int]) -> Optional[int]:
        text = self._decode_response_text(token_ids)
        marker = getattr(self.tool_parser, "tool_call_start_token", "<tool_call>")
        idx = text.find(marker)
        if idx == -1:
            return None
        prefix_text = text[:idx]
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        return len(prefix_ids)

    def _decode_response_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
