# RL中的奖赏计算
- 这一部分的核心文件在`envs`文件夹下，本文档主要是代码拆解与逻辑介绍，如果想知道如何快速使用SEARL开始训练，请参考[`docs/rl_factory/main_tutorial.md`](../main_tutorial.md)。本文档的主要内容包含：并行奖赏计算（可包含工具调用）、数据预处理、基于规则 & Model Judge的奖赏计算。

## 并行奖赏计算
+ 这一部分的核心文件在[`verl/workers/reward_manager/parallel.py`](../../../verl/workers/reward_manager/parallel.py)下，其主要功能是实现异步并行的奖赏计算，主要入口为`reward_fn(...)`（即`__call__`，`verify()`的功能与之相似）。
+ **步骤1**：如果`env_object`被设置为需要调用`verify_tool`(例如需要通过代码解释器工具来验证生成代码的正确性)，则通过`_get_verified_results`方法实现异步的工具调用，并将工具调用结果添加到`data`中：
    ```python
    for i in range(len(data)):
        data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]
    ```
+ **步骤2**：计算`step_mask`以获得score在reward_tensor中的赋值位置，在目前的版本中仍使用结果奖赏计算，即仅在最后一个token处提供reward而其余置零
    ```python
        step_mask = torch.zeros_like(data.batch['responses'], dtype=torch.long)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_attention_mask = data_item.batch['attention_mask'][prompt_length:]
            last_one_idx = torch.where(response_attention_mask == 1)[0][-1]
            step_mask[i, last_one_idx] = 1
    ```
+ **步骤3**：调用`env_object.compute_score()`并行计算batch reward，其分为1）使用规则计算，以及2）使用模型评判两种方式（两种方式细节的请见后文）
    ```python
    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto):
        if reward_rollout_wg is not None:
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            scores = self._compute_score_with_rules(data, tokenizer)
        
        return scores
    ```
+ **步骤4**：根据计算得到的`scores`为`reward_tensor`赋值，其与`data.batch['responses']`的维度是保持一致的
    ```python
    def _set_reward_tensor(self, scores, data: DataProto):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_mask = data.batch['step_mask']
        for i in range(len(data)):
            cur_step_mask, cur_scores = step_mask[i], scores[i]
            assert cur_step_mask.sum() == len(cur_scores), "step_mask and score length mismatch"
            mask_indices = torch.where(cur_step_mask == 1)[0]
            for j, idx in enumerate(mask_indices):
                reward_tensor[i, idx] = cur_scores[j]

        return reward_tensor
    ```
    值得注意的是，尽管目前框架仅支持结果奖赏设置，但我们为保留后续实现过程奖赏时的拓展性，`scores`实际上是一个是`list[list[float]]`，其中的第一层代表当前`batch`的大小，第二层则代表每个`batch`中`reward`的数量（当前仅为1），过程奖赏的`reward_tensor`将由`step_mask`的计算逻辑提供具体的赋值索引。

## 基于规则的奖赏计算
+ 基于规则的方式是在verl的基础上基于asyncio实现了异步的 **并行reward计算** ，使用者仅需修改`_compute_score_with_rules`方法，与`verl`中一样定义奖赏函数即可，例如可以直接将[Search-R1](https://github.com/PeterGriffinJin/Search-R1)中的奖赏函数[`qa_em.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py)迁移过来
    ```python
    def _compute_score_with_rules(self, data, tokenizer):
        def normalize_answer(s):
            ...

        def em_check(prediction, golden_answers):
            ...

        def extract_solution(solution_str):
            """Extract the equation from the solution string."""
            ...

        def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.):
            """The scoring function for exact match (EM)."""
            ...
        
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            # process the data_item to the token and decode them
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

            # reserved for compatibility
            prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

            score = compute_score_em(response_str, ground_truth)
            scores.append([score])

        return scores
    ```

## 基于Model Judge的奖赏计算
+ 基于Model Judge的奖赏计算额外创建一个`reward_rollout_wg`的worker group，并为其新建了**额外的资源池**，从而使用vllm实现对QwQ等Judge Model进行分布式推理。在`Env`中使用Judge Model生成**模型的Judge结果**，实现对训练模型生成的response的自动评估。
+ RL-factory集成了对Judge LLM的额外资源池申请和vllm初始化，使用时在程序入口处只需调整参数：
  ```python
  reward_rollout.if_use_reward_rollout=True
  ```
+ 即可自动创建reward judge model的资源池，其中模型路径参数示例：
    ```python
    reward_rollout.rollout.model_name=huggingface.co/Qwen/QwQ-32B
    ```
    其他配置参数详见`reward_rollout.rollout.`参数配置。

+ 在程序运行时，会调用环境Env类下的新评判函数_compute_score_with_reward_rollout_wg，如下所示。该函数的核心功能是基于actor生成的回答和真实标签的回答，构造出传递给judge model的prompt，然后返回judge model对本轮次actor生成回答的打分scores，并作为训练的奖励函数。

    ```python
  def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data: DataProto):
        # 基于actor rollout的回答和真实答案构造judge model的prompts
        reward_prompt_strs = self.get_prompt_for_reward(reward_tokenizer, data)
        
        # 将reward_prompt_strs转化为DataProto格式，准备好judge model的输入prompt数据data_proto
        ...
        
        # judge model生成responses
        responses_data = reward_rollout_wg.generate_sequences(data_proto)
        
        # 计算每个response的分数
        flat_scores = []
        for i, temp_response_data in enumerate(responses_data):
            # 获取每个judge model输出的response_str
            ...
            response_str = reward_tokenizer.decode(temp_valid_response_ids, skip_special_tokens=True)

            # 根据真实训练标签data_item.non_tensor_batch['reward_model']['ground_truth']和judge model生成的response_str计算打分
            score = self._compute_single_score_with_reward_rollout_wg(
                data_item.non_tensor_batch['data_source'], 
                response_str, 
                data_item.non_tensor_batch['reward_model']['ground_truth'], 
                data_item.non_tensor_batch.get('extra_info', None)
            )
            flat_scores.append(score)
        
        # 将flat_scores整理为scores
        ...
        return scores
    ```
+ 其中，以search-r1的搜索工具调用场景为例，judge model获取的prompt和response示例如下所示：

    ```python
    judge_model_prompt = """
    You are the assistant to an intelligent search helper, responsible for evaluating the quality of search results. These evaluations will be used to adjust search queries and to better respond to users using the relevant information obtained from searches.
    Below are the outputs from a fine-tuned LLM and the corresponding ground truth labels. The model outputs are enclosed in <output></output> tags, while the ground truth labels are enclosed in <gt></gt> tags. Please assess the quality of the model's output, assign a score, and present the score within <score></score> tags.

    Model outputs: <output></output> 
    Ground Truth: <gt>['Selena Gomez']</gt>
    """

    judge_model_response = """
    <think>Okay, let's see. The user wants me to evaluate the model's output against the ground truth. The model's output is empty, just "<output></output>", and the ground truth is ['Selena Gomez']. Hmm, so the model didn't provide any answer here. The correct answer is Selena Gomez, but the model's response is blank. That means it completely missed the target. I need to assign a score. Since there's no correct information provided, the score should be the lowest possible. If it's out of 5, maybe a 0 or 1? But the instructions didn't specify the scale. Wait, the example probably uses a 0-5 scale commonly used in such evaluations. A completely incorrect or missing response would get a 0. So I'll go with 0 here. The model didn't even attempt an answer, so no partial credit.
    </think>
    <score>0.0</score>  
    """
    ```

+ `get_prompt_for_reward()`函数定义如下，用于根据actor的输出逐个构造judge model的prompt。

    ```python
    def get_prompt_for_reward(self, reward_tokenizer, data: DataProto):
        # 根据actor的输出逐个构造judge model的prompt        
        for i in range(len(data)):
            # 获取有效的prompts，并通过定义mask_indices控制每个step的奖励位置
            ...
            
            for end_idx in mask_indices:
                # 截取当前response并生成prompt
                ...
            
                reward_prompt_str = self._get_single_prompt_str(
                    data_source=non_tensor_data['data_source'],
                    solution_str=current_response_str,
                    ground_truth=non_tensor_data['ground_truth'],
                    extra_info=non_tensor_data['extra_info'],
                    reward_tokenizer=reward_tokenizer
                )
                ...

        # 收集整理judge model的prompt并存储到reward_prompt_strs列表
        ...
                
        return reward_prompt_strs
    
    ```
+ 通过对`get_prompt_for_reward()`函数内调用的`_get_single_prompt_str()`函数中`system_content`和`user_content`的修改，实现judge model的prompts自定义。

    ```py
    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
            system_content = """
    You are the assistant to an intelligent search helper, responsible for evaluating the quality of search results. These evaluations will be used to adjust search queries and to better respond to users using the relevant information obtained from searches.
    Below are the outputs from a fine-tuned LLM and the corresponding ground truth labels. The model outputs are enclosed in <output></output> tags, while the ground truth labels are enclosed in <gt></gt> tags. Please assess the quality of the model's output, assign a score, and present the score within <score></score> tags.
    """
            user_content = """
    Model outputs: <output>{}</output>
    Ground Truth: <gt>{}</gt>
    """
            # 生成judge model的prompt模版
            result = reward_tokenizer.apply_chat_template([
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content.format(solution_str, ground_truth['target'])}
                    ], add_generation_prompt=True, tokenize=False)
            return result
    ```

+ 最后，通过`_compute_single_score_with_reward_rollout_wg()`函数提取judge model的单个打分并返回分数值，可以根据具体需求调整分数的提取规则。
    ```py
    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        pattern = r'<score>(.*?)</score>'
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return float(match.group(1).strip())
        else:
            return 0.0
    ```

## 使用工具验证辅助奖赏计算
+ 我们在`Env`中额外引入了`verify_tool`的方法，其能够在计算奖赏的时候被调用来评估原始response。以NL2SQL任务为例，模型最终输出的是一段代码，SEARL可以执行这一SQL语句来**判断程序的准确性**，我们同样采用异步并行的方式**提高调用效率**
+ 举例来说，verify_tool也可以调用API来实现Model Judge的功能
    + 首先将环境设置为需要使用`verify_tool`
        ```python
        self.use_verify_tool = True
        ```
    + 定义`verify_tool`实现调用API
        ```python
        def verify_tool(self, data_source, solution_str, ground_truth, extra_info):
            system_content = """
        You are the assistant to an intelligent search helper, responsible for evaluating the quality of search results. These evaluations will be used to adjust search queries and to better respond to users using the relevant information obtained from searches.
        Below are the outputs from a fine-tuned LLM and the corresponding ground truth labels. The model outputs are enclosed in <output></output> tags, while the ground truth labels are enclosed in <gt></gt> tags. Please assess the quality of the model's output, assign a score, and present the score within <score></score> tags.
        """
            user_content = """
        Model outputs: <output>{}</output>
        Ground Truth: <gt>{}</gt>
        """
            from openai import OpenAI
            client = OpenAI()

            completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content.format(solution_str, ground_truth['target'])}
                ]
            )

            return completion.choices[0].message.content
        ```
    + `verify_tool`在[`verl/workers/reward_manager/parallel.py`](../../../verl/workers/reward_manager/parallel.py)使用，把上述运行结果赋值到`data`中
        ```python
        for i in range(len(data)):
            data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]
        ```
    + 最后，在`compute_scores`中根据`verified_results`等信息计算`reward`
        ```python
        def _compute_score_with_rules(self, data, tokenizer):
            scores = []
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem
                
                # process the data_item to the token and decode them
                processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
                ground_truth, response_str = processed_data['ground_truth'], processed_data['response_str']

                # reserved for compatibility
                prompt_str, data_source, extra_info = processed_data['prompt_str'], processed_data['data_source'], processed_data['extra_info']

                verified_results = ground_truth['verified_results']
                pattern = r'<score>(.*?)</score>'
                match = re.search(pattern, verified_results, re.DOTALL)
                score = float(match.group(1).strip()) if match else 0.0

                scores.append([score])

            return scores
        ```