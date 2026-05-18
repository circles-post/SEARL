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

import torch
import asyncio
import numpy as np
from verl import DataProto
from envs.base import Env
from verl.workers.reward_manager import register


@register("parallel")
class AsyncRewardManager:
    """The reward manager with async processing.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.env_object = None
        self.reward_rollout_wg = None
        self.reward_tokenizer = None
        self.reward_fn_key = reward_fn_key
        self.if_val = False
        self.use_process_reward = False
        self.stop_token = None
    
    def set_env_object(self, env_object):
        self.env_object: Env = env_object
    
    def set_reward_rollout_wg(self, reward_rollout_wg):
        self.reward_rollout_wg = reward_rollout_wg
    
    def set_reward_tokenizer(self, reward_tokenizer):
        self.reward_tokenizer = reward_tokenizer
    
    def verify(self, data):
        device = data[0].batch['prompts']
        
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.env_object.use_verify_tool:
            data = self._get_verified_results(data)
        
        # 初始化step_mask
        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        for i in range(len(data)):
            data_item = data[i]
            # 获取prompts的长度
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            # 获取response部分的attention_mask
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            # 找到最后一个为1的位置
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1]
            # 在最后一个为1的位置设置为1
            step_mask[i, last_one_idx] = 1

        # 将step_mask添加到data中
        data.batch['step_mask'] = step_mask

        # 提取 tool_call_count 列表（每个样本一个值）
        tool_call_count_list = data.non_tensor_batch.get('tool_call_count', None)
        assert tool_call_count_list is not None, "tool_call_count_list is None"
        scores = self.env_object.compute_score(
            self.reward_rollout_wg,
            self.reward_tokenizer,
            self.tokenizer,
            data,
            if_val=self.if_val,
            tool_call_count=tool_call_count_list
        )
        
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=device)

        return scores
    

    def find_tool_end_positions(self, data: DataProto, tokenizer):
        """Find positions of all stop_tokens in responses, marking the end of tool calls.
        
        Args:
            data (DataProto): The data containing responses
            tokenizer: The tokenizer to decode token IDs
            
        Returns:
            list: A list of dictionaries, each containing positions of stop_token for one sample
        """
        tool_end_positions = []
        
        # Get the token ID for stop_token
        im_end_token_id = tokenizer.encode(self.stop_token, add_special_tokens=False)
        if isinstance(im_end_token_id, list) and len(im_end_token_id) > 0:
            im_end_token_id = im_end_token_id[0]
        
        for i in range(len(data)):
            data_item = data[i]
            
            # Get the entire response tokens
            response_ids = data_item.batch['responses']
            
            # Find all occurrences of im_end_token_id in response_ids
            positions = []
            
            # Convert to numpy for easier processing if it's a tensor
            if isinstance(response_ids, torch.Tensor):
                response_ids_np = response_ids.cpu().numpy()
            else:
                response_ids_np = response_ids
                
            # Find all positions where the token ID matches im_end_token_id
            for idx, token_id in enumerate(response_ids_np):
                if token_id == im_end_token_id:
                    positions.append(idx)
            
            tool_end_positions.append({
                'response_length': len(response_ids_np),
                'im_end_positions': positions,  # Positions in the response_ids array
                'im_end_count': len(positions)
            })
        
        return tool_end_positions
    
    def get_step_mask(self, data: DataProto):

        # 初始化step_mask
        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)

        #找到所有stop token的位置
        tool_end_positions = self.find_tool_end_positions(data,self.tokenizer)

        for i in range(len(data)):
            data_item = data[i]
            # 获取prompts的长度
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1].item()  # 转换为 Python int

            step_index=[]
            #记录每个<tool_call></tool_call>以及<answer></answer>导致的停止位置
            #
            # 问题分析：
            # 1. 正常情况下，im_end_count 应该是奇数（tool_call 对出现，最后一个 answer）
            # 2. 当 im_end_count 是偶数时，说明最后一个 answer 被截断或缺失
            # 3. 在偶数情况下，last_one_idx 可能和最后一个 im_end 重复，导致去重后位置减少
            if tool_end_positions[i]['im_end_count']%2 == 1:
                # 奇数情况：正常的对话流程
                for j in range((tool_end_positions[i]['im_end_count'] // 2) + 1):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])
            else:
                # 偶数情况：可能是 answer 被截断
                for j in range((tool_end_positions[i]['im_end_count'] // 2)):
                    step_index.append(tool_end_positions[i]['im_end_positions'][2 * j])

                # 只在 last_one_idx 不在已有位置中时才添加
                # 避免重复导致去重后位置减少
                # last_one_idx 已经是 Python int，可以直接比较
                if last_one_idx not in step_index:
                    step_index.append(last_one_idx)

            # 使用 set 去重并排序，确保没有重复值
            step_index = sorted(set(step_index))

            #根据是否使用过程奖励以及if_val来定义step_mask
            if self.use_process_reward and not self.if_val:
                for idx in step_index:
                    step_mask[i, idx] = 1
            else:
                step_mask[i, last_one_idx] = 1

        # 返回 step_mask 和 tool_end_positions，避免重复计算
        return step_mask, tool_end_positions

    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if self.env_object.use_verify_tool:
            data = self._get_verified_results(data)
        
        #if use process reward
        self.use_process_reward = self.env_object.use_process_reward
        
        #获取step_mask 和 tool_end_positions
        step_mask, tool_end_positions = self.get_step_mask(data)

        # 将step_mask添加到data中
        data.batch['step_mask'] = step_mask

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 提取 mcp_worthiness 和 tool_call_count 列表（每个样本一个值）
        mcp_worthiness_list = data.non_tensor_batch.get('mcp_worthiness', None)
        tool_call_count_list = data.non_tensor_batch.get('tool_call_count', None)
        # assert tool_call_count_list is not None, "tool_call_count_list is None"
        scores = self.env_object.compute_score(self.reward_rollout_wg, self.reward_tokenizer, self.tokenizer, data, if_val=self.if_val, use_process_reward=self.use_process_reward, mcp_worthiness=mcp_worthiness_list, tool_call_count=tool_call_count_list)

        # 调整 step_mask 以匹配 scores 的数量
        # 这是必要的，因为 tool_use_scores 的非 NaN 数量可能和 im_end 数量不一致
        # 传递 tool_end_positions 以便智能地选择添加位置
        step_mask = self._adjust_step_mask_to_scores(step_mask, scores, data, tool_end_positions)
        data.batch['step_mask'] = step_mask  # 更新调整后的 step_mask

        reward_tensor = self._set_reward_tensor(scores, data)

        acc_list = []
        for s in scores:
            if isinstance(s, (list, tuple, np.ndarray)):
                # 如果是序列，取最后一个元素 (Final Reward)
                acc_list.append(s[-1] if len(s) > 0 else 0.0)
            else:
                # 如果是纯数字 (标量)，直接用
                acc_list.append(s)

        # 4. 转 numpy (现在它是纯粹的一维数组了，不会报错)
        acc_arr = np.array(acc_list, dtype=np.float32)
        # ==================== 修改结束 / FIX END ====================

        try:
            data.batch["acc"] = torch.tensor(acc_arr, dtype=torch.float32, device=reward_tensor.device)
        except Exception:
            data.batch["acc"] = torch.tensor(acc_arr, dtype=torch.float32)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "acc": acc_arr.tolist(),
                },
            }
        else:
            return reward_tensor
    
    def _adjust_step_mask_to_scores(self, step_mask, scores, data: DataProto, tool_end_positions):
        """
        智能调整 step_mask 以匹配 scores，确保 reward 对齐的正确性。

        核心改进：
        1. 移除位置时：优先保留前面重要的步骤位置
        2. 添加位置时：智能选择真实的 im_end 位置，保持顺序
        3. 记录对齐质量，警告可能的 reward 失调

        Args:
            step_mask: 原始的 step_mask
            scores: 计算得到的分数列表（每个样本一个列表）
            data: DataProto 对象
            tool_end_positions: 已计算的 tool_end_positions 列表

        Returns:
            调整后的 step_mask
        """
        adjusted_mask = step_mask.clone()
        adjustment_made = False
        alignment_quality = {'perfect': 0, 'good_remove': 0, 'good_add': 0, 'poor_add': 0}

        for i in range(len(data)):
            cur_scores = scores[i]
            mask_indices = torch.where(adjusted_mask[i] == 1)[0]
            mask_count = len(mask_indices)
            scores_count = len(cur_scores)

            if mask_count == scores_count:
                # 数量匹配，无需调整
                alignment_quality['perfect'] += 1
                continue

            adjustment_made = True
            print(f"\n{'='*80}")
            print(f"[智能调整] 样本 {i}")
            print(f"  原始 step_mask 位置数: {mask_count}")
            print(f"  scores 数量: {scores_count}")
            print(f"  差异: {scores_count - mask_count}")

            if mask_count < scores_count:
                # ========== 情况1：需要添加位置 ==========
                data_item = data[i]
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
                valid_positions = torch.where(response_attention_mask == 1)[0]

                needed = scores_count - mask_count
                print(f"  需要添加 {needed} 个位置")
                print(f"  当前标记的位置: {mask_indices.tolist()}")

                # 策略：在已有位置之间插入，保持顺序
                tool_end_info = tool_end_positions[i]
                all_im_end_positions = sorted(tool_end_info['im_end_positions'])

                # 找到所有未使用的 im_end 位置
                unused_im_end = [pos for pos in all_im_end_positions
                                if adjusted_mask[i, pos] == 0]

                added_positions = []
                if len(unused_im_end) >= needed:
                    # 好情况：有足够的 im_end 位置
                    # 选择策略：从已标记位置之间的 im_end 中选择
                    current_marked = mask_indices.tolist()

                    for pos in unused_im_end[:needed]:
                        adjusted_mask[i, pos] = 1
                        added_positions.append(pos)
                        print(f"  ✓ 添加 im_end 位置: {pos}")

                    alignment_quality['good_add'] += 1
                    print(f"  ✓ 对齐质量: GOOD（使用真实 im_end 位置）")

                else:
                    # 差情况：im_end 不够，需要用其他位置
                    # 先用完所有 unused_im_end
                    for pos in unused_im_end:
                        adjusted_mask[i, pos] = 1
                        added_positions.append(pos)
                        print(f"  ✓ 添加 im_end 位置: {pos}")
                        needed -= 1

                    # 剩余的用最后一个有效位置（通常是 answer 结束）
                    if needed > 0:
                        last_idx = valid_positions[-1].item()
                        if adjusted_mask[i, last_idx] == 0:
                            adjusted_mask[i, last_idx] = 1
                            added_positions.append(last_idx)
                            needed -= 1
                            print(f"  △ 添加最后位置: {last_idx}")

                    # 如果还不够（极少见），从后往前填充
                    if needed > 0:
                        for pos in reversed(valid_positions.tolist()):
                            if adjusted_mask[i, pos] == 0:
                                adjusted_mask[i, pos] = 1
                                added_positions.append(pos)
                                needed -= 1
                                print(f"  ✗ [降级] 添加普通位置: {pos}")
                                if needed == 0:
                                    break

                    alignment_quality['poor_add'] += 1
                    print(f"  ⚠️ 对齐质量: POOR（缺少 im_end 位置，reward 可能失调）")
                    import warnings
                    warnings.warn(
                        f"样本 {i}: 无法找到足够的 im_end 位置来对齐 {scores_count} 个分数。"
                        f"这会导致部分 step reward 对齐不准确！"
                    )

                print(f"  添加的位置: {sorted(added_positions)}")
                print(f"  调整后位置: {torch.where(adjusted_mask[i] == 1)[0].tolist()}")

            else:
                # ========== 情况2：需要移除位置 ==========
                excess = mask_count - scores_count
                print(f"  需要移除 {excess} 个位置")
                print(f"  当前标记的位置: {mask_indices.tolist()}")

                # 策略：保留前 scores_count 个位置（最重要的步骤）
                # 移除最后的位置（通常是 less important 或被 rollback 的）
                kept_indices = mask_indices[:scores_count]
                removed_indices = mask_indices[scores_count:]

                # 重置并保留
                adjusted_mask[i] = 0
                for idx in kept_indices:
                    adjusted_mask[i, idx] = 1

                alignment_quality['good_remove'] += 1
                print(f"  ✓ 保留前 {scores_count} 个位置: {kept_indices.tolist()}")
                print(f"  ✓ 移除后 {excess} 个位置: {removed_indices.tolist()}")
                print(f"  ✓ 对齐质量: GOOD（保留重要步骤）")

            print(f"{'='*80}\n")

        # 统计和警告
        if adjustment_made:
            total = len(data)
            poor_rate = alignment_quality['poor_add'] / total if total > 0 else 0

            print(f"\n{'='*80}")
            print(f"[对齐质量统计]")
            print(f"  完美匹配: {alignment_quality['perfect']}/{total} ({alignment_quality['perfect']/total*100:.1f}%)")
            print(f"  良好添加: {alignment_quality['good_add']}/{total}")
            print(f"  良好移除: {alignment_quality['good_remove']}/{total}")
            print(f"  差劲添加: {alignment_quality['poor_add']}/{total}")

            if poor_rate > 0.05:  # 5%
                print(f"\n⚠️⚠️⚠️ 严重警告：{poor_rate*100:.1f}% 的样本对齐质量差！")
                print(f"这会导致 Process Reward 失调，影响训练效果！")
                print(f"建议：")
                print(f"  1. 检查 tool_use 和 response 生成是否同步")
                print(f"  2. 减少 rollback 触发频率（应用配置优化）")
                print(f"  3. 考虑禁用过程奖励，只用最终奖励")
            elif poor_rate > 0:
                print(f"\n⚠️ 警告：{poor_rate*100:.1f}% 的样本对齐质量差，请关注")

            print(f"{'='*80}\n")

        return adjusted_mask

    def _set_reward_tensor(self, scores, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_mask = data.batch['step_mask']
        for i in range(len(data)):
            cur_step_mask, cur_scores = step_mask[i], scores[i]
            mask_indices = torch.where(cur_step_mask == 1)[0]
            # torch.save(data, "outputs/debug/data.pth")
            assert cur_step_mask.sum() == len(cur_scores), f"cur_step_mask.sum(): {cur_step_mask.sum()}, len(cur_scores): {len(cur_scores)}"

            for j, idx in enumerate(mask_indices):
                reward_tensor[i, idx] = cur_scores[j]
            '''
            if self.use_process_reward:
                for j, idx in enumerate(mask_indices):
                    reward_tensor[i, idx] = cur_scores[j]
            else:
                 for j, idx in enumerate(mask_indices):
                    if j == len(mask_indices)-1:
                        reward_tensor[i, idx] = cur_scores[-1]
                        #reward_tensor[i,idx]=cur_scores.sum()
            '''

        return reward_tensor
    
    def _get_verified_results(self, data: DataProto):
        async def _get_single_result(data_source, solution_str, ground_truth, extra_info):
            # 使用asyncio.to_thread将同步函数转换为异步操作
            return await asyncio.to_thread(
                self.env_object.verify_tool, 
                data_source, 
                solution_str, 
                ground_truth, 
                extra_info
            )
        
        async def _process_all():
            tasks = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch['data_source']
                extra_info = data_item.non_tensor_batch.get('extra_info', None)

                # 创建异步任务
                task = _get_single_result(data_source, response_str, ground_truth, extra_info)
                tasks.append(task)

            # 并行执行所有任务
            return await asyncio.gather(*tasks)

        # 在同步函数中运行异步代码
        results = asyncio.run(_process_all())

        for i in range(len(data)):
            data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]

        return data
