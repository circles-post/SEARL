import os
import copy
import logging
import asyncio
import torch
import itertools
import numpy as np
import ray
import zlib
from typing import Any, Optional, List, Dict, Tuple
from verl import DataProto
from tensordict import TensorDict
from time import time

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RollbackConfig:
    """Configuration for rollback mechanism"""
    def __init__(
        self,
        enable_rollback: bool = True,
        max_retries: int = 3,
        retry_on_errors: List[str] = None,
        retry_strategy: str = "batch_redo",  # "batch_redo" or "sample_level"
    ):
        self.enable_rollback = enable_rollback
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy

        # Default error patterns to trigger retry
        if retry_on_errors is None:
            self.retry_on_errors = [
                "ImportError",
                "SyntaxError",
                "worker_timeout",
                "queue_timeout",
                "internal_error",
                "NameError",
                "AttributeError",
                "TypeError",
                "ValueError",
            ]
        else:
            self.retry_on_errors = retry_on_errors

        logger.info(f"RollbackConfig initialized: enable={enable_rollback}, "
                   f"max_retries={max_retries}, strategy={retry_strategy}")


class RollbackManager:
    """Manages retry logic and checkpoint state for error recovery"""

    def __init__(self, config: RollbackConfig):
        self.config = config
        self.retry_counts: Dict[int, int] = {}  # batch_idx -> retry_count
        self.checkpoints: Dict[int, Dict[str, Any]] = {}  # batch_idx -> checkpoint
        self.error_history: Dict[int, List[Dict[str, Any]]] = {}  # batch_idx -> error list
        self.total_retries = 0
        self.successful_retries = 0

    def should_retry(self, batch_idx: int) -> bool:
        """Check if this batch should be retried"""
        if not self.config.enable_rollback:
            return False
        retry_count = self.retry_counts.get(batch_idx, 0)
        return retry_count < self.config.max_retries

    def increment_retry(self, batch_idx: int):
        """Increment retry count for a batch"""
        self.retry_counts[batch_idx] = self.retry_counts.get(batch_idx, 0) + 1
        self.total_retries += 1
        logger.info(f"Batch {batch_idx} retry count: {self.retry_counts[batch_idx]}/{self.config.max_retries}")

    def record_success(self, batch_idx: int):
        """Record successful retry"""
        if batch_idx in self.retry_counts and self.retry_counts[batch_idx] > 0:
            self.successful_retries += 1
            logger.info(f"Batch {batch_idx} succeeded after {self.retry_counts[batch_idx]} retries")

    def record_error(self, batch_idx: int, error_info: Dict[str, Any]):
        """Record error for a batch"""
        if batch_idx not in self.error_history:
            self.error_history[batch_idx] = []
        self.error_history[batch_idx].append(error_info)
        logger.warning(f"Batch {batch_idx} error recorded: {error_info.get('error_type', 'unknown')}")

    def create_checkpoint(self, batch_idx: int, state: Dict[str, Any]):
        """Create a checkpoint for rollback"""
        self.checkpoints[batch_idx] = copy.deepcopy(state)
        logger.debug(f"Checkpoint created for batch {batch_idx}")

    def restore_checkpoint(self, batch_idx: int) -> Optional[Dict[str, Any]]:
        """Restore state from checkpoint"""
        if batch_idx in self.checkpoints:
            logger.info(f"Restoring checkpoint for batch {batch_idx}")
            return copy.deepcopy(self.checkpoints[batch_idx])
        return None

    def clear_checkpoint(self, batch_idx: int):
        """Clear checkpoint after successful completion"""
        if batch_idx in self.checkpoints:
            del self.checkpoints[batch_idx]
        if batch_idx in self.retry_counts:
            del self.retry_counts[batch_idx]

    def detect_error_from_tool_result(self, tool_result: Dict[str, Any]) -> Tuple[bool, Optional[str], str]:
        """
        Detect if tool execution resulted in an error

        Args:
            tool_result: Dictionary returned from tool execution

        Returns:
            Tuple of (is_error, error_type, reason)
        """
        # Check run_success or success field
        run_success = tool_result.get('run_success', tool_result.get('success', True))

        if not run_success:
            error_type = tool_result.get('error_type', 'unknown_error')
            reason = tool_result.get('reason', tool_result.get('stderr', 'No reason provided'))

            # Check if this error type should trigger retry
            should_retry = any(pattern in error_type for pattern in self.config.retry_on_errors)

            return True, error_type if should_retry else None, reason

        return False, None, ""

    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics"""
        return {
            'total_retries': self.total_retries,
            'successful_retries': self.successful_retries,
            'active_checkpoints': len(self.checkpoints),
            'retry_success_rate': self.successful_retries / self.total_retries if self.total_retries > 0 else 0.0
        }


def pad_matrix_to_shape(data, target_shape, pad_value=np.nan):
    """
    将不规则的嵌套列表pad为指定形状的三维矩阵

    Args:
        data: 嵌套列表数据
        target_shape: 目标形状 (n, m, k)
        pad_value: 用于填充的值，默认为np.nan

    Returns:
        填充后的numpy数组
    """
    n, m, k = target_shape

    # 创建目标形状的数组，初始化为pad_value
    padded = np.full((n, m, k), pad_value, dtype=object)

    # 遍历原始数据并填充
    for i in range(min(n, len(data))):
        if i < len(data):
            for j in range(min(m, len(data[i]))):
                if j < len(data[i]):
                    for l in range(min(k, len(data[i][j]))):
                        if l < len(data[i][j]):
                            padded[i, j, l] = data[i][j][l]

    return padded


def auto_pad_matrix(data, pad_value=np.nan):
    """
    自动分析数据维度并pad为规则形状
    """
    if not data:
        return np.array([])

    # 找到最大维度，限制max_dim3不超过10
    max_dim1 = len(data)
    max_dim2 = max(len(item) for item in data) if data else 0
    max_dim3 = 10  # 强制限制为10

    # 截断超过10长度的数据以防止报错
    truncated_data = []
    for item in data:
        if item is None:
            truncated_data.append([])
            continue
        truncated_item = []
        for subitem in item:
            if isinstance(subitem, list):
                # 截断超过10长度的子列表
                truncated_item.append(subitem[:10] if len(subitem) > 10 else subitem)
            else:
                # 非列表元素保持不变
                truncated_item.append(subitem)
        truncated_data.append(truncated_item)

    return pad_matrix_to_shape(truncated_data, (max_dim1, max_dim2, max_dim3), pad_value)


class ToolUtilsRollback:
    """Extended ToolUtils with rollback mechanism, workflow and MCP support"""

    def __init__(
        self,
        tokenizer,
        meta_info,
        config,
        env_object,
        rollback_config: Optional[RollbackConfig] = None,
        is_validate: bool = False
    ):
        self.tokenizer = tokenizer
        self.final_str = config.stop[-1] if config.stop else ''
        self.config_prompt_length = config.prompt_length
        self.config_response_length = config.response_length
        self.stop_id = self.tokenizer.encode(config.stop[0], add_special_tokens=False)[0]
        self.max_turns = config.max_turns
        self.max_prompt_length = config.prompt_length
        self.max_tool_response_length = config.tool_response_length
        self.rollout_num = config.n
        self.is_validate = is_validate

        pad_token_id = meta_info.get('pad_token_id')
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        else:
            eos_token_id = meta_info.get('eos_token_id')
            if isinstance(eos_token_id, (list, tuple)):
                self.pad_token_id = eos_token_id[-1]
            else:
                self.pad_token_id = eos_token_id

        eos_token_id = meta_info.get('eos_token_id')
        if isinstance(eos_token_id, (list, tuple)):
            self.eos_token_id = eos_token_id[0]
        else:
            self.eos_token_id = eos_token_id

        self.meta_info = meta_info
        self.loop_cnt = 0
        self.env_object = env_object

        # Initialize rollback manager
        if rollback_config is None:
            rollback_config = RollbackConfig()
        self.rollback_manager = RollbackManager(rollback_config)

        # Additional state for rollback
        self.pending_retries: List[int] = []  # Batch indices that need retry
        self.batch_error_flags: Dict[int, bool] = {}  # Track errors per batch

        logger.info("ToolUtilsRollback initialized with rollback support, workflow and MCP")

    def _create_checkpoint_state(self, batch_idx: int) -> Dict[str, Any]:
        """Create checkpoint state for a batch"""
        return {
            'loop_responses_token': copy.deepcopy(self.loop_responses_token[batch_idx]),
            'tool_use': copy.deepcopy(self.tool_use[batch_idx]),
            'mcp_ids': copy.deepcopy(self.mcp_ids[batch_idx]) if hasattr(self, 'mcp_ids') else [],
            'loop_cnt': self.loop_cnt,
        }

    def _restore_checkpoint_state(self, batch_idx: int, checkpoint: Dict[str, Any]):
        """Restore state from checkpoint"""
        self.loop_responses_token[batch_idx] = checkpoint['loop_responses_token']
        self.tool_use[batch_idx] = checkpoint['tool_use']
        if hasattr(self, 'mcp_ids') and 'mcp_ids' in checkpoint:
            self.mcp_ids[batch_idx] = checkpoint['mcp_ids']
        # Note: loop_cnt is global, handled separately

    def _detect_errors_in_batch(
        self,
        infos_str: List[str],
        batch_idxs: List[int],
        tool_results: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        """
        Detect errors in batch responses

        Args:
            infos_str: Tool response strings
            batch_idxs: Batch indices
            tool_results: Optional raw tool result dictionaries

        Returns:
            Tuple of (error_batch_indices, error_info_dict)
        """
        error_indices = []
        error_info_dict = {}

        for idx, batch_idx in enumerate(batch_idxs):
            # Method 1: Check tool_results if available
            if tool_results and idx < len(tool_results):
                is_error, error_type, reason = self.rollback_manager.detect_error_from_tool_result(tool_results[idx])

                if is_error and error_type:  # error_type is None if shouldn't retry
                    error_indices.append(batch_idx)
                    error_info = {
                        'error_type': error_type,
                        'reason': reason,
                        'batch_idx': batch_idx,
                        'loop_cnt': self.loop_cnt,
                        'detection_method': 'tool_result'
                    }
                    error_info_dict[batch_idx] = error_info
                    self.rollback_manager.record_error(batch_idx, error_info)
                    continue

            # Method 2: Fallback to text-based detection
            info_text = infos_str[idx] if idx < len(infos_str) else ""

            # Check for error patterns in text
            for error_pattern in self.rollback_manager.config.retry_on_errors:
                if error_pattern.lower() in info_text.lower():
                    error_indices.append(batch_idx)
                    error_info = {
                        'error_type': error_pattern,
                        'reason': info_text[:200],  # First 200 chars
                        'batch_idx': batch_idx,
                        'loop_cnt': self.loop_cnt,
                        'detection_method': 'text_pattern'
                    }
                    error_info_dict[batch_idx] = error_info
                    self.rollback_manager.record_error(batch_idx, error_info)
                    break

        return error_indices, error_info_dict

    def find_step_id_by_mcp_name(self, mcp_name_list, mcp_dict=None):
        """
        将工具名称列表转换为ID列表
        修复版：正确处理各种类型的工具名称，使用哈希保证相同名称有相同ID

        Args:
            mcp_name_list: List[str] - 工具名称列表，例如 ["search_tool", "calculator"]
            mcp_dict: dict - 已注册的MCP工具字典（可选，用于警告）

        Returns:
            List[int] - 工具ID列表（使用CRC32哈希）
        """
        if mcp_dict is None:
            mcp_dict = {}

        # 类型检查
        if not isinstance(mcp_name_list, (list, tuple)):
            logger.warning(f"find_step_id_by_mcp_name expected list/tuple, got {type(mcp_name_list)}")
            return [np.nan]

        tool_id_list = []
        for item in mcp_name_list:
            # 处理非字符串类型
            if not isinstance(item, str):
                logger.warning(f"Non-string tool name: {item} (type: {type(item)})")
                tool_id_list.append(np.nan)
                continue

            # 计算哈希ID（所有字符串统一处理）
            # 使用CRC32确保相同名称产生相同ID，便于分组
            if item in ["search-search", "execute_python_code-execute_python_code"]:
                tool_id = np.nan
            else:
                tool_id = zlib.crc32(item.encode('utf-8'))

            # 检查是否已注册（仅用于调试信息）
            if item not in mcp_dict:
                # 特殊标记不需要警告
                if item not in ["plan", "plan failed", "<error>", "<empty>", "None", "search-search", "execute_python_code-execute_python_code"]:
                    logger.info(f"Tool '{item}' not found in registered MCPs (using hash ID {tool_id})")

            tool_id_list.append(tool_id)

        return tool_id_list

    async def _process_single_response_async(self, idx, responses, optimal_idx, mcp_box):
        """
        异步处理单个轨迹（responses），与 mcp_box actor 进行交互。
        修复版：仅收集规划信息，MCP注册将在批量处理阶段完成。
        """
        temp_dict = {}
        temp_prompts_list = []
        loss_mask = []

        is_optimal_trajectory = (idx == optimal_idx)
        input_ids_list = list(itertools.chain.from_iterable(responses))

        # 收集规划信息
        planning_info = None

        for turn_idx, response_turn in enumerate(responses):
            length = len(response_turn)

            if is_optimal_trajectory:  # 只在最优轨迹上收集规划信息
                if not self.is_validate:  # Test的时候就不更新MCP Box了
                    if turn_idx == 0:  # 规划阶段
                        plan_str = self.tokenizer.decode(response_turn, skip_special_tokens=True)
                        planning_info = {
                            'trajectory_idx': idx,
                            'plan_str': plan_str,
                            'max_turns': self.max_turns
                        }
                    # 注意：MCP注册将在批量处理阶段完成，这里不再收集注册任务

            loss_mask.extend([(turn_idx + 1) % 2] * length)

        # 返回处理结果和规划信息
        return {
            'input_ids_list': input_ids_list,
            'loss_mask': loss_mask,
            'total_length': len(input_ids_list),
            'planning_info': planning_info,
            'trajectory_idx': idx
        }

    def _handle_batch_redo(
        self,
        output: DataProto,
        error_indices: List[int],
        batch_idxs: List[int]
    ) -> Optional[DataProto]:
        """
        Handle batch redo strategy: if any sample fails, redo entire batch

        Returns:
            DataProto for retry if needed, None otherwise
        """
        if not error_indices:
            return None

        # Check if any batch can be retried
        can_retry = any(self.rollback_manager.should_retry(idx) for idx in error_indices)

        if not can_retry:
            logger.warning(f"Max retries reached for batch indices: {error_indices}")
            return None

        logger.info(f"Batch redo triggered. Errors in {len(error_indices)} samples: {error_indices}")

        # Restore all batches to checkpoint
        restored_count = 0
        for batch_idx in batch_idxs:
            checkpoint = self.rollback_manager.restore_checkpoint(batch_idx)
            if checkpoint:
                self._restore_checkpoint_state(batch_idx, checkpoint)
                restored_count += 1

        if restored_count == 0:
            logger.error("No checkpoints available for rollback")
            return None

        logger.info(f"Restored {restored_count} batch checkpoints")

        # Increment retry counts for error samples
        for batch_idx in error_indices:
            self.rollback_manager.increment_retry(batch_idx)

        # Reconstruct input for retry - use same logic as postprocess_output
        # but with restored state
        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []

        for batch_idx in batch_idxs:
            prompt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
            next_prompt_token.append(prompt_token)
            next_prompt_length.append(len(prompt_token))
            next_sample_idx.append(batch_idx)

        # Left pad
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask

        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:],
                'position_ids': position_ids[:, -max_len:],
                'attention_mask': next_attention_mask[:, -max_len:]
            },
            batch_size=next_input_ids.shape[0]
        )

        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        retry_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        retry_data.meta_info.update(self.meta_info)
        retry_data.meta_info['index'] = next_sample_idx
        retry_data.meta_info['do_sample'] = False
        retry_data.meta_info['is_retry'] = True

        logger.info(f"Prepared retry batch with {len(next_sample_idx)} samples")
        return retry_data

    def _handle_sample_level_redo(
        self,
        infos_str: List[str],
        dones: List[bool],
        step_scores: List[float],
        tool_names: List[Any],
        error_indices: List[int],
        batch_idxs: List[int],
    ) -> Optional[DataProto]:
        """
        Handle sample-level retry: only rollback/retry failed samples while
        successful samples continue to next state.

        Returns:
            DataProto for next step (mixed normal + retry samples), None if finished.
        """
        if not error_indices:
            return None

        retryable_error_indices = [idx for idx in error_indices if self.rollback_manager.should_retry(idx)]
        if not retryable_error_indices:
            logger.warning(f"No retryable samples in errors={error_indices}; all reached max retries")
            return None

        logger.info(
            "Sample-level redo triggered. retryable=%s, all_errors=%s",
            retryable_error_indices,
            error_indices,
        )

        retryable_set = set(retryable_error_indices)
        info_tokens = self.tokenizer(infos_str, truncation=True, max_length=self.max_tool_response_length).input_ids

        # Rollback only retryable samples to their checkpoints
        for batch_idx in retryable_error_indices:
            checkpoint = self.rollback_manager.restore_checkpoint(batch_idx)
            if checkpoint is not None:
                self._restore_checkpoint_state(batch_idx, checkpoint)
                self.rollback_manager.increment_retry(batch_idx)
            else:
                logger.warning(f"No checkpoint for retryable sample {batch_idx}; skip rollback for this sample")

        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []

        # Build next prompts for all unfinished samples:
        # - retryable error samples: use restored checkpoint state (no progress this turn)
        # - others: consume env info and progress normally
        for idx, batch_idx in enumerate(batch_idxs):
            if dones[idx]:
                self.rollback_manager.clear_checkpoint(batch_idx)
                continue

            if batch_idx not in retryable_set:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)
                self.tool_use[batch_idx].append(step_scores[idx])
                self.mcp_ids[batch_idx].append(tool_names[idx])

                # Update checkpoint for progressed samples that are not in current error set.
                if batch_idx not in error_indices and self.rollback_manager.config.enable_rollback:
                    checkpoint_state = self._create_checkpoint_state(batch_idx)
                    self.rollback_manager.create_checkpoint(batch_idx, checkpoint_state)
                    self.rollback_manager.record_success(batch_idx)

            prompt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
            next_prompt_token.append(prompt_token)
            next_prompt_length.append(len(prompt_token))
            next_sample_idx.append(batch_idx)

        if len(next_prompt_token) == 0:
            stats = self.rollback_manager.get_statistics()
            logger.info(f"Rollback statistics: {stats}")
            return None

        # Left pad
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask

        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                "input_ids": next_input_ids[:, -max_len:],
                "position_ids": position_ids[:, -max_len:],
                "attention_mask": next_attention_mask[:, -max_len:],
            },
            batch_size=next_input_ids.shape[0],
        )

        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={"raw_prompt_ids": raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info["index"] = next_sample_idx
        next_data.meta_info["do_sample"] = False
        next_data.meta_info["is_retry"] = True
        next_data.meta_info["retry_indices"] = retryable_error_indices
        return next_data

    def postprocess_output(
        self,
        output: DataProto,
        step: int,
        questions: list = None,
        mcp_worthiness: float = None,
        tool_call_count: int = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
        is_validation: bool = False
    ):
        """
        Extended postprocess_output with rollback support, workflow and MCP tracking

        Args:
            output: DataProto from rollout
            step: Current step number
            questions: Optional question list
            mcp_worthiness: Optional MCP worthiness scores
            tool_call_count: Optional tool call count
            tool_results: Optional raw tool result dictionaries for error detection
            is_validation: Whether this is a validation run
        """
        start_time = time()

        # 使用类属性 is_validate（优先级更高），如果没有则使用参数
        is_validation = self.is_validate if hasattr(self, 'is_validate') else is_validation

        logger.info(f"[TOOL_UTILS_ROLLBACK] postprocess_output 开始处理 step {step}, loop_cnt {self.loop_cnt}, is_validation={is_validation}")
        logger.info(f"[TOOL_UTILS_ROLLBACK] 输入数据大小: batch_size={output.batch.batch_size[0] if hasattr(output.batch, 'batch_size') else 'unknown'}")

        # Initialize on first loop
        if self.loop_cnt == 0:
            self.batch_size = output.batch.batch_size[0]
            self.loop_responses_token = [[] for _ in range(self.batch_size)]
            self.init_prompt_token = output.batch.get('prompts')
            self.tool_use = [[] for _ in range(self.batch_size)]
            self.mcp_ids = [[] for _ in range(self.batch_size)]
            self.tool_call_count = tool_call_count
            prompt_length = self.init_prompt_token.shape[-1]
            self.init_attention_mask = output.batch.get('attention_mask')[:, :prompt_length]

            batch_idxs = list(range(self.batch_size))
            for idx in range(self.batch_size):
                prompt_token = self.init_prompt_token[idx]
                prompt_token_list = prompt_token[prompt_token != self.pad_token_id].tolist()
                self.loop_responses_token[idx].append(prompt_token_list)

                # Create initial checkpoint
                if self.rollback_manager.config.enable_rollback:
                    checkpoint_state = self._create_checkpoint_state(idx)
                    self.rollback_manager.create_checkpoint(idx, checkpoint_state)
        else:
            batch_idxs = output.meta_info['index']

        # Process responses
        responses = output.batch.get('responses')
        process_response = []
        for idx, batch_idx in enumerate(batch_idxs):
            response_token = responses[idx]
            response_token_list = response_token[response_token != self.pad_token_id].tolist()
            if self.env_object.use_process_reward:
                # Ensure last token is stop token
                if response_token_list[-1] != self.stop_id:
                    if len(response_token_list) != self.config_response_length:
                        response_token_list.append(self.stop_id)
                    else:
                        response_token_list[-1] = self.stop_id
            self.loop_responses_token[batch_idx].append(response_token_list)
            process_response.append(response_token_list)

        # Decode responses for env step
        decode_start = time()
        responses_str = self.tokenizer.batch_decode(
            process_response,
            skip_special_tokens=False,
        )
        decode_end = time()
        logger.info(f"[TOOL_UTILS_ROLLBACK] Token decode 耗时: {decode_end - decode_start:.3f}秒")

        # Call env.step
        step_start_time = time()
        try:
            infos_str, dones, valid_actions, _, tool_names = self.env_object.step(
                responses=responses_str,
                tokenizer=self.tokenizer,
                loop_cnt=self.loop_cnt,
                questions=questions,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation,
                batch_indices=batch_idxs
            )
        except TypeError:
            infos_str, dones, valid_actions, _, tool_names = self.env_object.step(
                responses=responses_str,
                tokenizer=self.tokenizer,
                loop_cnt=self.loop_cnt,
                questions=questions,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation
            )
        step_end_time = time()
        logger.info(f"[TOOL_UTILS_ROLLBACK] Env step 耗时: {step_end_time - step_start_time:.3f}秒")

        logger.info(f"dones number: {sum(dones)}")

        # Get step rewards
        if self.env_object.use_process_reward:
            step_scores = self.env_object.get_step_reward(responses=responses_str, mcp_worthiness=mcp_worthiness)
        else:
            step_scores = [0] * len(responses_str)

        # Error detection
        error_indices, error_info_dict = self._detect_errors_in_batch(
            infos_str, batch_idxs, tool_results
        )

        # Handle rollback if errors detected
        if error_indices and self.rollback_manager.config.enable_rollback:
            logger.warning(f"Errors detected in batch: {error_indices}")

            retry_strategy = self.rollback_manager.config.retry_strategy
            if retry_strategy == "sample_level" and self.loop_cnt == 0:
                # loop_cnt==0 is planning phase in MCP env. Mixing progressed and retried
                # samples is unsafe here, so fallback to batch_redo.
                logger.warning("sample_level retry at loop_cnt=0 is unsafe; fallback to batch_redo")
                retry_data = self._handle_batch_redo(output, error_indices, batch_idxs)
            elif retry_strategy == "sample_level":
                retry_data = self._handle_sample_level_redo(
                    infos_str=infos_str,
                    dones=dones,
                    step_scores=step_scores,
                    tool_names=tool_names,
                    error_indices=error_indices,
                    batch_idxs=batch_idxs,
                )
            else:
                # default: batch_redo
                retry_data = self._handle_batch_redo(output, error_indices, batch_idxs)

            if retry_data is not None:
                if retry_strategy == "sample_level" and self.loop_cnt > 0:
                    # sample-level branch already prepared next prompts.
                    self.loop_cnt += 1
                logger.info(f"Returning retry data for strategy={retry_strategy}")
                return retry_data
            else:
                logger.warning("Cannot retry, proceeding with errors")

        # Record successes and update checkpoints
        for batch_idx in batch_idxs:
            if batch_idx not in error_indices:
                self.rollback_manager.record_success(batch_idx)
                # Update checkpoint for next turn
                if self.rollback_manager.config.enable_rollback and not dones[batch_idxs.index(batch_idx)]:
                    checkpoint_state = self._create_checkpoint_state(batch_idx)
                    self.rollback_manager.create_checkpoint(batch_idx, checkpoint_state)

        # Encode infos for next prompt
        encode_start = time()
        info_tokens = self.tokenizer(infos_str, truncation=True, max_length=self.max_tool_response_length).input_ids
        encode_end = time()
        logger.info(f"[TOOL_UTILS_ROLLBACK] Info encode 耗时: {encode_end - encode_start:.3f}秒")

        next_prompt_token = []
        next_prompt_length = []
        next_sample_idx = []

        for idx, batch_idx in enumerate(batch_idxs):
            if not dones[idx]:
                info_token_list = info_tokens[idx]
                self.loop_responses_token[batch_idx].append(info_token_list)
                next_sample_idx.append(batch_idx)
                prompt_token = list(itertools.chain.from_iterable(self.loop_responses_token[batch_idx]))
                next_prompt_token.append(prompt_token)
                next_prompt_length.append(len(prompt_token))
                # Get process reward
                self.tool_use[batch_idx].append(step_scores[idx])
                self.mcp_ids[batch_idx].append(tool_names[idx])
            else:
                # Clear checkpoint for completed samples
                self.rollback_manager.clear_checkpoint(batch_idx)

        if len(next_prompt_token) == 0:
            # Log statistics before returning
            stats = self.rollback_manager.get_statistics()
            logger.info(f"Rollback statistics: {stats}")
            return

        # Left pad
        pad_start = time()
        max_len = max(max(next_prompt_length), self.config_prompt_length)
        next_prompt_token_pad = []
        for prompt_token in next_prompt_token:
            token = [self.pad_token_id] * (max_len - len(prompt_token)) + prompt_token
            next_prompt_token_pad.append(token)
        pad_end = time()
        logger.info(f"[TOOL_UTILS_ROLLBACK] Token padding 耗时: {pad_end - pad_start:.3f}秒")

        next_input_ids = torch.tensor(next_prompt_token_pad, dtype=torch.int64)
        next_attention_mask = next_input_ids != self.pad_token_id
        position_ids = torch.clip(torch.cumsum(next_attention_mask, dim=-1) - 1, min=0, max=None) * next_attention_mask

        max_len = self.config_prompt_length
        next_batch = TensorDict(
            {
                'input_ids': next_input_ids[:, -max_len:],
                'position_ids': position_ids[:, -max_len:],
                'attention_mask': next_attention_mask[:, -max_len:]
            },
            batch_size=next_input_ids.shape[0]
        )

        raw_prompt_ids = np.empty(len(next_prompt_token), dtype=object)
        raw_prompt_ids[:] = [x[-max_len:] for x in next_prompt_token]

        next_data = DataProto(batch=next_batch, non_tensor_batch={'raw_prompt_ids': raw_prompt_ids})
        next_data.meta_info.update(self.meta_info)
        next_data.meta_info['index'] = next_sample_idx
        next_data.meta_info['do_sample'] = False
        self.loop_cnt += 1

        end_time = time()
        total_time = end_time - start_time
        logger.info(f"[TOOL_UTILS_ROLLBACK] postprocess_output 总耗时: {total_time:.3f}秒")

        return next_data

    def compose_final_output(self, step, group) -> DataProto:
        """Compose final generation output."""
        start_time = time()
        input_ids_list = []
        loss_mask_list = []
        length_list = []
        print("Start to compose final output")
        # centralized_actor_handle 本身已是 Ray ActorHandle，不需要再 .remote()
        mcp_box_start = time()
        mcp_box = getattr(self.env_object, "centralized_actor_handle", None)
        if mcp_box is None and getattr(self.env_object, "tool_manager", None) is not None:
            mcp_box = getattr(self.env_object.tool_manager, "centralized_actor_handle", None)
        if mcp_box is None:
            raise AttributeError("centralized_actor_handle 未初始化，请确保 env_object/tool_manager 正确注入 Ray actor 句柄。")
        print(f"获取 mcp_box 耗时: {time() - mcp_box_start:.4f}s")

        # 使用累积的 step 奖励选择组内最优轨迹；若缺失奖励则默认 0
        reward_calc_start = time()
        total_rewards = []
        for tool_use in self.tool_use:
            if tool_use:
                total_rewards.append(float(np.nansum(tool_use)))
            else:
                total_rewards.append(0.0)
        optimal_indices = {}
        for start in range(0, len(self.loop_responses_token), self.rollout_num):
            end = start + self.rollout_num
            group_rewards = total_rewards[start:end]
            if not group_rewards:
                continue
            local_best = int(np.argmax(group_rewards))
            for i in range(len(group_rewards)):
                optimal_indices[start + i] = start + local_best
        print(f"计算奖励和最优索引耗时: {time() - reward_calc_start:.4f}s")

        tasks = [
            self._process_single_response_async(idx, responses[1:], optimal_indices.get(idx), mcp_box)
            for idx, responses in enumerate(self.loop_responses_token)
        ]

        async def _run_tasks(ts):
            return await asyncio.gather(*ts)

        # 从同步上下文收集异步结果和批量任务
        async_exec_start = time()
        try:
            gathered_results = asyncio.run(_run_tasks(tasks))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                gathered_results = loop.run_until_complete(_run_tasks(tasks))
            finally:
                loop.close()
        print(f"异步任务执行耗时: {time() - async_exec_start:.4f}s")

        # 批量处理规划和MCP注册任务 - 修复版：正确的两阶段处理
        async def _batch_process_async():
            from collections import defaultdict

            # 收集所有规划任务
            all_planning_tasks = []
            trajectory_results = {}

            # 找到所有最优轨迹（去重）
            optimal_trajectory_set = set(optimal_indices.values())
            print(f"最优轨迹集合: {optimal_trajectory_set}")

            for result in gathered_results:
                if result:
                    trajectory_idx = result["trajectory_idx"]
                    trajectory_results[trajectory_idx] = result

                    # 收集规划任务
                    if result.get("planning_info"):
                        all_planning_tasks.append(result["planning_info"])

            # === 优化1：批量规划任务 - 使用单次 Ray 调用 ===
            planning_results = {}
            if all_planning_tasks:
                print(f"开始批量处理 {len(all_planning_tasks)} 个规划任务...")
                planning_batch_start = time()

                try:
                    # 将所有规划文本打包成单个列表，减少远程调用次数
                    plan_texts = [p["plan_str"] for p in all_planning_tasks]
                    traj_indices = [p["trajectory_idx"] for p in all_planning_tasks]
                    max_turns_list = [p["max_turns"] for p in all_planning_tasks]

                    # 批量调用 - 添加超时保护 (90秒,适应大批次场景)
                    try:
                        future = mcp_box.batch_extract_dag_and_parallel.remote(plan_texts, traj_indices, max_turns_list)
                        batch_results = await asyncio.wait_for(future, timeout=90.0)

                        # 解析批量结果
                        for traj_idx, result in zip(traj_indices, batch_results):
                            planning_results[traj_idx] = result

                        planning_batch_duration = time() - planning_batch_start
                        print(f"批量规划处理耗时: {planning_batch_duration:.3f}s (平均 {planning_batch_duration / len(all_planning_tasks):.3f}s/个)")
                    except asyncio.TimeoutError:
                        print("Warning: batch_extract_dag_and_parallel timeout after 90s, falling back to sequential planning")
                        # 降级：对所有任务使用顺序规划
                        for planning_info in all_planning_tasks:
                            try:
                                fallback_future = mcp_box.sequential_planning.remote(planning_info["max_turns"])
                                fallback_result = await asyncio.wait_for(fallback_future, timeout=5.0)
                                planning_results[planning_info["trajectory_idx"]] = fallback_result
                            except Exception as e:
                                logger.error(f"Sequential planning also failed for trajectory {planning_info['trajectory_idx']}: {e}")
                                planning_results[planning_info["trajectory_idx"]] = (None, None, None)
                except AttributeError:
                    # 如果 batch_extract_dag_and_parallel 不存在，回退到逐个调用（保持向后兼容）
                    print("Warning: batch_extract_dag_and_parallel not found, falling back to sequential processing")
                    planning_futures = []
                    for planning_info in all_planning_tasks:
                        try:
                            future = mcp_box.extract_dag_and_parallel.remote(planning_info["plan_str"])
                            planning_futures.append((planning_info["trajectory_idx"], future, planning_info))
                        except Exception as e:
                            logger.error(f"Error preparing DAG extraction for trajectory {planning_info['trajectory_idx']}: {e}")
                            future = mcp_box.sequential_planning.remote(planning_info["max_turns"])
                            planning_futures.append((planning_info["trajectory_idx"], future, planning_info))

                    # 添加超时保护
                    planning_completed = await asyncio.gather(
                        *[asyncio.wait_for(future, timeout=10.0) for _, future, _ in planning_futures],
                        return_exceptions=True,
                    )

                    for (traj_idx, _, planning_info), result in zip(planning_futures, planning_completed):
                        if isinstance(result, (Exception, asyncio.TimeoutError)):
                            logger.error(f"Planning failed for trajectory {traj_idx}: {result}")
                            try:
                                fallback_future = mcp_box.sequential_planning.remote(planning_info["max_turns"])
                                fallback_result = await asyncio.wait_for(fallback_future, timeout=5.0)
                                planning_results[traj_idx] = fallback_result
                            except Exception as fallback_e:
                                logger.error(f"Fallback planning also failed for trajectory {traj_idx}: {fallback_e}")
                                planning_results[traj_idx] = (None, None, None)
                        else:
                            planning_results[traj_idx] = result
                except Exception as e:
                    logger.error(f"批量规划处理失败: {e}")
                    # 降级：对所有任务返回空结果
                    for planning_info in all_planning_tasks:
                        planning_results[planning_info["trajectory_idx"]] = (None, None, None)

            # === 优化2：批量MCP注册任务 - 修复版：根据规划结果收集注册任务 ===
            print("\n=== 阶段2: 根据规划结果收集MCP注册任务 ===")
            registration_by_trajectory = defaultdict(list)

            # 遍历所有轨迹，根据规划结果收集工具调用
            for traj_idx, result in trajectory_results.items():
                # 只处理最优轨迹
                if traj_idx not in optimal_trajectory_set:
                    continue

                # 检查是否有规划结果
                if traj_idx not in planning_results:
                    print(f"Warning: Trajectory {traj_idx} 没有规划结果，跳过MCP注册")
                    continue

                _, task_id, subtask_dict = planning_results[traj_idx]

                # 只有规划成功才注册MCP
                if task_id is None:
                    print(f"Warning: Trajectory {traj_idx} 规划失败 (task_id=None)，跳过MCP注册")
                    continue

                print(f"Processing trajectory {traj_idx}: task_id={task_id}")

                # 获取该轨迹的所有响应
                responses = self.loop_responses_token[traj_idx]

                # 遍历所有工具调用 turn (奇数索引: 1, 3, 5, ...)
                # turn_idx=0 是规划，turn_idx=1,3,5,... 是工具调用
                for turn_idx in range(1, len(responses), 2):
                    if turn_idx >= len(responses):
                        break

                    # 解码工具调用
                    tool_call_str = self.tokenizer.decode(responses[turn_idx], skip_special_tokens=True)

                    # 添加到注册任务列表
                    registration_by_trajectory[traj_idx].append(
                        {
                            "tool_call_str": tool_call_str,
                            "task_id": task_id,
                            "subtask_dict": subtask_dict,
                            "turn_idx": turn_idx,
                        }
                    )

                print(f"Trajectory {traj_idx}: 收集了 {len(registration_by_trajectory[traj_idx])} 个工具调用")

            # 批量注册MCP
            if registration_by_trajectory:
                print(f"\n开始批量处理 {sum(len(v) for v in registration_by_trajectory.values())} 个MCP注册任务...")
                registration_batch_start = time()

                try:
                    registration_futures = []
                    for traj_idx, tasks in registration_by_trajectory.items():
                        # 提取该轨迹的所有工具调用字符串
                        tool_call_strs = [t["tool_call_str"] for t in tasks]
                        task_id = tasks[0]["task_id"]  # 同一轨迹的task_id相同
                        subtask_dict = tasks[0]["subtask_dict"]

                        try:
                            # 批量注册同一轨迹的所有MCP，添加超时保护 (60秒)
                            future = mcp_box.batch_register_mcps.remote(tool_call_strs, task_id, subtask_dict)
                            registration_futures.append((traj_idx, len(tasks), future))
                            print(f"提交批量注册任务: trajectory {traj_idx}, {len(tasks)} 个工具")
                        except AttributeError:
                            # 回退到逐个注册
                            print("Warning: batch_register_mcps 不可用，回退到逐个注册")
                            for task in tasks:
                                future = mcp_box.register_mcp.remote(task["tool_call_str"], task["task_id"], task["subtask_dict"])
                                registration_futures.append((traj_idx, 1, future))

                    if registration_futures:
                        # 为每个future添加超时保护
                        registration_results = await asyncio.gather(
                            *[asyncio.wait_for(future, timeout=60.0) for _, _, future in registration_futures],
                            return_exceptions=True,
                        )

                        success_count = 0
                        error_count = 0
                        for (traj_idx, count, _), result in zip(registration_futures, registration_results):
                            if isinstance(result, (Exception, asyncio.TimeoutError)):
                                error_count += count
                                logger.error(f"MCP registration failed for trajectory {traj_idx}: {result}")
                            else:
                                success_count += count

                        registration_batch_duration = time() - registration_batch_start
                        avg_time = registration_batch_duration / max(1, success_count + error_count)
                        print(
                            f"MCP批量注册耗时: {registration_batch_duration:.3f}s | "
                            f"成功: {success_count}, 失败: {error_count} | 平均: {avg_time:.3f}s/个"
                        )

                except Exception as e:
                    logger.error(f"批量MCP注册处理失败: {e}")
            else:
                print("Warning: 没有收集到任何MCP注册任务")

            # 返回处理后的结果
            return trajectory_results, planning_results

        # 在同步上下文中运行异步批处理
        try:
            trajectory_results, planning_results = asyncio.run(_batch_process_async())
        except RuntimeError:
            # 处理事件循环已在运行的情况
            async def run_in_existing_loop():
                return await _batch_process_async()

            loop = asyncio.new_event_loop()
            try:
                trajectory_results, planning_results = loop.run_until_complete(run_in_existing_loop())
            finally:
                loop.close()

        # 整理最终结果
        result_process_start = time()
        for traj_idx, result in trajectory_results.items():
            input_ids_list.append(result["input_ids_list"])
            loss_mask_list.append(result["loss_mask"])
            length_list.append(result["total_length"])

        result_process_duration = time() - result_process_start
        print(f"结果整理耗时: {result_process_duration:.3f}s")
        print("End to compose final output")
        # max_len = max(max(length_list), self.config_response_length)
        reduce_start = time()
        max_response_length = torch.tensor([max(length_list)], device=torch.cuda.current_device())
        # because only tp=0 will exec postprocess_output and compose_final_output
        # so we exec all_reduce in specified group(dp group)
        torch.distributed.all_reduce(max_response_length, op=torch.distributed.ReduceOp.MAX, group=group)
        max_len = int(max_response_length)
        print(f"计算最大响应长度和all_reduce耗时: {time() - reduce_start:.4f}s")
        print("End to reduce max_response_length")
        # right pad
        padding_start = time()
        input_ids = []
        loss_mask = []
        for idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids + [self.pad_token_id] * (max_len - len(input_ids))
            loss_mask = loss_mask_list[idx] + [0] * (max_len - len(loss_mask_list[idx]))
            input_ids_list[idx] = input_ids
            loss_mask_list[idx] = loss_mask[0:max_len]
        print(f"填充input_ids和loss_mask耗时: {time() - padding_start:.4f}s")

        tensor_creation_start = time()
        response_token = torch.tensor(input_ids_list, dtype=torch.int64)[:, :max_len]
        response_loss_mask = torch.tensor(loss_mask_list, dtype=torch.float32)
        response_attention_mask = (response_token != self.pad_token_id).long()
        print(f"创建响应张量耗时: {time() - tensor_creation_start:.4f}s")

        # get the max length of the process rewards
        tool_process_start = time()
        max_tool_use_len = self.max_turns
        for tool_use_item in self.tool_use:
            max_tool_use_len = max(max_tool_use_len, len(tool_use_item))

        tool_use_tensor = []
        mcp_id_tensor = []
        candidate_mcp_dict = ray.get(mcp_box.get_mcps.remote())

        print("\n=== 工具ID收集调试信息 ===")
        print(f"已注册的MCP工具数量: {len(candidate_mcp_dict)}")
        if candidate_mcp_dict:
            print(f"已注册的MCP工具示例 (前5个): {list(candidate_mcp_dict.keys())[:5]}")

        # Pad tool_use to have consistent dimensions
        for idx in range(len(self.tool_use)):
            # 处理 tool_use（奖励分数）
            if not self.tool_use[idx]:
                padded_tool_use = [torch.nan] * max_tool_use_len
            else:
                padded_tool_use = self.tool_use[idx] + [torch.nan] * (max_tool_use_len - len(self.tool_use[idx]))
            # 处理 mcp_ids（工具名称 -> 工具ID）
            if not self.mcp_ids[idx]:
                padded_tool_id = [[np.nan]] * max_tool_use_len
            else:
                tool_id_list = []
                for turn_idx, turn_tools in enumerate(self.mcp_ids[idx]):
                    # turn_tools 可能是:
                    # 1. List[str] - 工具名称列表，例如 ["search_tool", "calculator"]
                    # 2. str - 单个字符串，例如 "plan", "plan failed"
                    if isinstance(turn_tools, list):
                        # 情况1: 列表形式
                        tool_ids = self.find_step_id_by_mcp_name(turn_tools, candidate_mcp_dict)
                        tool_id_list.append(tool_ids)
                    elif isinstance(turn_tools, str):
                        # 情况2: 字符串形式（特殊标记或单个工具）
                        # 统一转换为列表处理
                        tool_ids = self.find_step_id_by_mcp_name([turn_tools], candidate_mcp_dict)
                        tool_id_list.append(tool_ids)
                    else:
                        # 未预期的类型
                        print(
                            f"Warning: Trajectory {idx}, Turn {turn_idx}: "
                            f"Unexpected tool type {type(turn_tools)}, value: {turn_tools}"
                        )
                        tool_id_list.append([np.nan])

                # 填充到最大长度
                padded_tool_id = tool_id_list + [[np.nan]] * (max_tool_use_len - len(tool_id_list))

            mcp_id_tensor.append(padded_tool_id)
            tool_use_tensor.append(padded_tool_use)

        # 调试信息：检查NaN数量
        total_tool_ids = sum(len(traj) for traj in mcp_id_tensor)
        nan_count = sum(
            sum(1 for tool_id_list in traj for tool_id in tool_id_list if isinstance(tool_id, float) and np.isnan(tool_id))
            for traj in mcp_id_tensor
        )
        print(f"工具ID统计: 总数={total_tool_ids}, NaN数量={nan_count}, NaN比例={nan_count / max(1, total_tool_ids) * 100:.1f}%")
        print(f"处理tool_use和mcp_ids耗时: {time() - tool_process_start:.4f}s")

        final_batch_creation_start = time()
        tool_use_score = torch.tensor(tool_use_tensor)
        step_rewards_tensor = torch.nan_to_num(tool_use_score, nan=0.0)

        anchor_obs = auto_pad_matrix(mcp_id_tensor)

        traj_index = np.arange(len(mcp_id_tensor))
        input_ids = torch.cat([self.init_prompt_token, response_token], dim=-1)
        attention_mask = torch.cat([self.init_attention_mask, response_attention_mask], dim=-1)
        # position_ids = torch.cumsum(attention_mask, dim=1, dtype=torch.long) - 1
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None) * attention_mask
        loss_mask = torch.cat([torch.zeros_like(self.init_attention_mask, dtype=torch.float32), response_loss_mask], dim=-1)
        final_batch = TensorDict(
            {
                "prompts": self.init_prompt_token,
                "responses": response_token,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "tool_use_scores": tool_use_score,
                "step_rewards": step_rewards_tensor,
            },
            batch_size=self.batch_size,
        )
        if self.tool_call_count is None:
            self.tool_call_count = np.array([0] * self.batch_size, dtype=object)
        final_output = DataProto(
            batch=final_batch,
            non_tensor_batch={
                "mcp_name": anchor_obs,
                "anchor_obs": anchor_obs,
                "traj_index": traj_index,
                "tool_call_count": self.tool_call_count,
            },
        )
        print(f"创建最终batch和output耗时: {time() - final_batch_creation_start:.4f}s")
        print(f"compose_final_output 总耗时: {time() - start_time:.4f}s")
        return final_output

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback statistics"""
        return self.rollback_manager.get_statistics()
