import re
import json
import string
import random
import torch
from .base import Env
import numpy as np
from pathlib import Path
FENCE_RE = re.compile(r'<python>([\s\S]*?)</python>', re.S | re.IGNORECASE)
FUNC_RE  = re.compile(r'^\s*def\s+\w+\s*\([^)]*\)\s*:', re.MULTILINE)
ANSI_RE  = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
SENTINEL_RE = re.compile(r'<\|im_[^|]*\|>')

def _sanitize_code(code: str) -> str:
    code = ANSI_RE.sub('', code)
    code = SENTINEL_RE.sub('', code)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # REMOVED: The line using BACKTICK_TAIL_RE is gone.
    return code

def _parse_plan(raw_plan: str):
    text = _sanitize_code(raw_plan)
    # 1. 修改正则表达式，增加一个捕获组来专门捕获标题
    pattern = re.compile(
        r'^\s*##\s*(ST\d+):?([^\n]*)\n'  # 组1: ST编码, 组2: 标题
        r'([\s\S]*?)'                   # 组3: 内容（步骤）
        r'(?=\n\s*##\s*ST\d+\b|\Z)',     
        flags=re.MULTILINE
    )

    subtask_dict = {}

    for m in pattern.finditer(text):
        st_code, title_raw, content = m.groups()
        
        title = title_raw.strip() or st_code
        
        content_stripped = content.strip()
        
        steps = re.findall(r'^\s*\d+\.\s*.*$', content_stripped, flags=re.MULTILINE)
        steps_str = '\n'.join(s.strip() for s in steps)

        subtask_dict[st_code] = {
            "title": title,
            "steps": steps_str
        }
    return subtask_dict


class MCPBaseEnv(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False
        self.plan_success_list = [] # list of plan success
        self.plan_list = []

    def get_step_reward(self, responses, mcp_worthiness, format_score=0.1):
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(0)
            elif "plan" in temp_action:
                if temp_action == 'plan_success':
                    step_reward.append(format_score)
                else:
                    step_reward.append(-0.5 * format_score)
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)
                else:
                    fail_number = 0
                    creation_number = 0
                    for i in range(len(temp_tool_list)):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                        elif temp_tool_list[i]['name'] == 'create_and_execute_mcp':
                            creation_number += 1
                    step_rew = ((len(temp_tool_list) - 2 *fail_number + creation_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
        return step_reward

    def step(self, responses, tokenizer=None, loop_cnt=-1, questions=None, mcp_worthiness=None, is_validation=False, batch_indices=None):
        if loop_cnt != 0:
            cur_actions, tool_results, tool_lists = self.tool_manager.execute_actions(responses=responses)
        else:
            # new rollout / episode: clear previous plans
            self.plan_success_list = []
            self.plan_list = []
            cur_actions, tool_results, raw_plans, tool_lists = self.tool_manager.execute_plans(
                responses=responses,
                overall_tasks=questions,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation  # 传递 validation 标志
            )
            assert len(cur_actions) == len(raw_plans), (
                f"plan length mismatch: actions={len(cur_actions)} "
                f"raw_plans={len(raw_plans)} loop_cnt={loop_cnt}"
            )
            for action, raw_plan in zip(cur_actions, raw_plans):
                if action == 'plan_success':
                    self.plan_success_list.append(True)
                    self.plan_list.append(_parse_plan(raw_plan))
                else:
                    self.plan_success_list.append(False)
                    self.plan_list.append(raw_plan)

        # For loop_cnt>0, responses may be a subset; align plans to current batch.
        if batch_indices is not None:
            current_plan_list = [self.plan_list[i] for i in batch_indices]
        else:
            current_plan_list = self.plan_list

        if not tokenizer:
            assert len(cur_actions) == len(current_plan_list), (
                f"plan_list length mismatch: actions={len(cur_actions)} "
                f"plans={len(current_plan_list)} loop_cnt={loop_cnt}"
            )
            next_obs, dones, valid_action, is_tool, tool_names = [], [], [], [], []
            for action, tool_result, tool_list, current_plan in zip(cur_actions, tool_results, tool_lists, current_plan_list):
                if action == 'answer':
                    temp_next_obs, temp_done, temp_valid_action, temp_is_tool = '', True, 1, 0
                else:
                    temp_next_obs = tool_result
                    if action == 'error':
                        temp_done, temp_valid_action, temp_is_tool = False, 0, 0
                    elif action == 'actions':
                        temp_done, temp_valid_action, temp_is_tool = False, 1, 1
                    elif action == 'plan_success':
                        temp_done, temp_valid_action, temp_is_tool = False, 1, 0
                    elif action.startswith("plan"):
                        temp_done, temp_valid_action, temp_is_tool = False, 0, 0
                    else:
                        raise ValueError('Unexpected action: {}'.format(action))

                next_obs.append(temp_next_obs)
                dones.append(temp_done)
                valid_action.append(temp_valid_action)
                is_tool.append(temp_is_tool)
                tool_names.append(tool_list)
            return next_obs, dones, valid_action, is_tool, tool_names

        import asyncio
        assert len(cur_actions) == len(current_plan_list), (
            f"plan_list length mismatch: actions={len(cur_actions)} "
            f"plans={len(current_plan_list)} loop_cnt={loop_cnt}"
        )
        async def process_all_prompts():
            """异步处理所有 prompt 生成"""
            tasks = []

            for action, tool_result, tool_list, current_plan in zip(cur_actions, tool_results, tool_lists, current_plan_list):
                if action == 'answer':
                    async def answer_task(tl=tool_list):
                        return ('', True, 1, 0, tl)
                    tasks.append(answer_task())
                elif action == 'error':
                    mode = 'tool_call'
                    done, valid, is_t = False, 0, 0
                    tasks.append(self._process_single_prompt(tool_result, tokenizer, mode, current_plan, done, valid, is_t, tool_list))
                elif action == 'actions':
                    mode = 'tool_call'
                    done, valid, is_t = False, 1, 1
                    tasks.append(self._process_single_prompt(tool_result, tokenizer, mode, current_plan, done, valid, is_t, tool_list))
                elif action == 'plan_success':
                    mode = 'plan_call'
                    done, valid, is_t = False, 1, 0
                    tasks.append(self._process_single_prompt(tool_result, tokenizer, mode, current_plan, done, valid, is_t, tool_list))
                elif action.startswith("plan"):
                    mode = 'plan_call'
                    done, valid, is_t = False, 0, 0
                    tasks.append(self._process_single_prompt(tool_result, tokenizer, mode, current_plan, done, valid, is_t, tool_list))
                else:
                    raise ValueError('Unexpected action: {}'.format(action))

            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)

            next_obs, dones, valid_action, is_tool, tool_names = [], [], [], [], []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Error processing prompt: {result}")
                    # 降级：返回空结果
                    next_obs.append('')
                    dones.append(False)
                    valid_action.append(0)
                    is_tool.append(0)
                    tool_names.append([])
                else:
                    obs, done, valid, is_t, tool_list = result
                    next_obs.append(obs)
                    dones.append(done)
                    valid_action.append(valid)
                    is_tool.append(is_t)
                    tool_names.append(tool_list)

            return next_obs, dones, valid_action, is_tool, tool_names

        # 在同步上下文中运行异步任务
        try:
            return asyncio.run(process_all_prompts())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(process_all_prompts())
            finally:
                loop.close()

    async def _process_single_prompt(self, tool_result, tokenizer, mode, current_plan, done, valid, is_tool, tool_list):
        """异步处理单个 prompt 生成"""
        try:
            prompt = await self.tool_manager.get_prompt_async(
                input_data=tool_result,
                tokenizer=tokenizer,
                mode=mode,
                add_generation_prompt=True,
                current_plan=current_plan
            )
            return (prompt, done, valid, is_tool, tool_list)
        except Exception as e:
            print(f"Error in _process_single_prompt: {e}")
            # 降级：使用同步版本
            try:
                prompt = self.tool_manager.get_prompt(
                    input_data=tool_result,
                    tokenizer=tokenizer,
                    mode=mode,
                    add_generation_prompt=True,
                    current_plan=current_plan
                )
                return (prompt, done, valid, is_tool, tool_list)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return (tool_result, done, valid, is_tool, tool_list)


    def extract_first_final_answer(self, text: str) -> str | None:  ### 从 final_answer() 里提取出 answer_str
        """
        Extracts the content from within the very first final_answer(...) call in a string,
        handling cases with multiple calls on the same line correctly.

        Args:
            text: The string to search within.

        Returns:
            The content inside the parentheses, or None if not found.
        """
        # 修改点: 将 (.*) 改为 (.*?) 使其变为非贪婪模式
        match = re.search(r"final_answer\((.*?)\)", text, re.DOTALL)
        
        if match:
            # group(1) 返回第一个捕获组的内容
            content = match.group(1).strip()
            
            # 可选：如果内容被引号包裹，则去掉引号
            if content.startswith(('"', "'")) and content.endswith(('"', "'")):
                # 使用切片去除首尾的引号
                return content[1:-1]
            
            return content
        else:
            return " "

    def _process_data(self, data_item, tokenizer):  ### todo 继承了这个方法
        # process the data_item to the token and decode them
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch.get('extra_info', None)
        mcp_worthiness = data_item.non_tensor_batch.get('mcp_worthiness')
        tool_call_count = data_item.non_tensor_batch.get('tool_call_count')
        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info,
            'mcp_worthiness': mcp_worthiness,
            'tool_call_count': tool_call_count
        }
        
    def compute_score_single(self, data_source, solution_str, ground_truth, extra_info, mcp_worthiness, tool_call_count):
        import sys
        sys.path.append(str(Path(__file__).resolve().parent / "EED"))
        from EED import compute_score_EED
        if data_source in ["MATH-Hard", "jee"]:
            from . import math_hard_verify
            res = math_hard_verify.compute_score(solution_str, ground_truth, extra_info=extra_info)
            return res, {}  # Return (score, breakdown) tuple for consistency
        elif data_source in ["PHYBench"]:
            res = compute_score_EED(solution_str, ground_truth, extra_info)
            return res, {}  # Return (score, breakdown) tuple for consistency
        elif data_source in ['arpo']:
            from .reward_utils import compute_score
            res_dict = compute_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info, mcp_worthiness=mcp_worthiness, tool_call_count=tool_call_count)
            # Extract score and return full breakdown
            if isinstance(res_dict, dict):
                return res_dict["score"], res_dict  # Return (score, breakdown) tuple
            else:
                return res_dict, {}  # Backward compatibility
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    def _compute_score_with_rules(self, data, tokenizer, if_val=False, mcp_worthiness=None, tool_call_count=None):
        print(f'+++++[info] use reward in MathEnv [info]+++++')
        format_score = 0.0 if if_val else 0.1
        scores = []
        reward_breakdowns = []  # Store detailed reward breakdown for monitoring

        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            '''
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info,
            'mcp_worthiness': mcp_worthiness,
            'tool_call_count': tool_call_count
            '''
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']
            # 从函数参数（列表）中获取第 i 个样本对应的值
            item_mcp_worthiness = mcp_worthiness[i] if mcp_worthiness is not None else None
            item_tool_call_count = tool_call_count[i] if tool_call_count is not None else None

            # compute_score_single now returns (score, breakdown) tuple
            score, breakdown = self.compute_score_single(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth['target'],
                extra_info=extra_info,
                mcp_worthiness=item_mcp_worthiness,
                tool_call_count=item_tool_call_count
            )
            scores.append([score])
            reward_breakdowns.append(breakdown)

        # Store reward breakdown in non_tensor_batch for monitoring
        if hasattr(data, 'non_tensor_batch'):
            data.non_tensor_batch['reward_breakdown'] = np.array(reward_breakdowns, dtype=object)

        return scores



if __name__ == '__main__':
    # 调试 get_prompt：直接使用 MCPBaseEnv，本地注入最小 ToolManager
    import os

    class DummyTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "<decoded>"

    class DummyToolManager:
        def get_prompt(self, input_data, tokenizer, mode, add_generation_prompt, current_plan=None):
            return f"[prompt mode={mode} add_gen={add_generation_prompt} plan={current_plan}] {input_data}"

        def execute_plans(self, responses, overall_tasks):
            # 预设三种动作：actions / plan_success / error
            actions = ['actions', 'plan_success', 'error']
            tool_results = ['tool_result_actions', 'tool_result_plan', 'tool_result_error']
            raw_plans = [
                "## ST1: demo\n1. step one\n##DAG_LIST\n[(ST1)]",
                "## ST1: demo\n1. step one\n##DAG_LIST\n[(ST1)]",
                overall_tasks[2]
            ]
            tool_lists = [['t1'], ['t2'], ['t3']]
            return actions, tool_results, raw_plans, tool_lists

    # 最小配置：先用正常方式构造 MCPBaseEnv，然后替换内部 tool_manager 以便专注调试 get_prompt
    dummy_config = {
        "tool_manager": "adaptive",
        "model_type": "qwen3",
        "max_prompt_length": 2048,
        "use_process_reward": False,
    }
    env = MCPBaseEnv(config=dummy_config, centralized_actor=None)
    env.tool_manager = DummyToolManager()
    env.plan_success_list = []
    env.plan_list = []

    responses = ["resp1", "resp2", "resp3"]
    questions = ["q1", "q2", "q3"]
    tokenizer = DummyTokenizer()

    next_obs, dones, valid_action, is_tool, tool_names = env.step(
        responses=responses, tokenizer=tokenizer, loop_cnt=0, questions=questions
    )

    print("next_obs:", next_obs)
    print("dones:", dones)
    print("valid_action:", valid_action)
    print("is_tool:", is_tool)
    print("tool_names:", tool_names)
