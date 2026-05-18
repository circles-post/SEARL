import re
import json
import string
import random
import torch
from .base import Env
from pathlib import Path

class BaselineEnv(Env):
    def __init__(self, config, centralized_actor=None):
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False

    def get_step_reward(self, responses, format_score=0.1):
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(torch.nan)
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)
                else:
                    fail_number = 0
                    for i in range(len(temp_tool_list )):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    step_rew = ((len(temp_tool_list) - 2 *fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
       

        return step_reward

    def _process_data(self, data_item, tokenizer):
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
        
        return {
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info
        }

    def compute_score_single(self, data_source, solution_str, ground_truth, extra_info):
        import sys
        sys.path.append(str(Path(__file__).resolve().parent / "EED"))
        from EED import compute_score_EED
        if data_source in ["MATH-Hard", "jee"]:
            from . import math_hard_verify
            res = math_hard_verify.compute_score(solution_str, ground_truth, extra_info=extra_info)
        elif data_source in ["PHYBench"]:
            res = compute_score_EED(solution_str, ground_truth, extra_info)
        elif data_source in ['arpo']:
            from .reward_utils import compute_score
            res_dict = compute_score(solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
            # Extract score for backward compatibility (compute_score now returns dict)
            res = res_dict["score"] if isinstance(res_dict, dict) else res_dict
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
        return res

    # NOTE: Add your reward calculation rules here!
    def _compute_score_with_rules(self, data, tokenizer, if_val=False):
        print(f'+++++[info] use reward in BaselineEnv [info]+++++')
        format_score = 0.0 if if_val else 0.1
        scores = []
        for i in range(len(data)):
            data_item = data[i]
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            '''
            'prompt_str': prompt_str,
            'response_str': response_str,
            'ground_truth': ground_truth,
            'data_source': data_source,
            'extra_info': extra_info
            '''
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']
            
            score = self.compute_score_single(data_source=data_source, solution_str=response_str, ground_truth=ground_truth['target'], extra_info=extra_info)
            scores.append([score])
        
        return scores
