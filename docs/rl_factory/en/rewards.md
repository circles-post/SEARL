# Reward Calculation in RL
- The core files for this section are located in the `envs` folder. This document mainly explains the code structure and logic. If you want to quickly learn how to use SEARL for training, please refer to [`docs/rl_factory/main_tutorial.md`](../main_tutorial.md). The main contents of this document include: parallel reward calculation (including tool calls), data preprocessing, and reward calculation based on rules & Model Judge.

## Parallel Reward Calculation
+ The core file for this part is [`verl/workers/reward_manager/parallel.py`](../../../verl/workers/reward_manager/parallel.py), whose main function is to implement asynchronous parallel reward calculation. The main entry point is `reward_fn(...)` (i.e., `__call__`, and `verify()` has similar functionality).
+ **Step 1**: If `env_object` is set to require `verify_tool` (e.g., to verify code correctness using a code interpreter tool), it will asynchronously call the tool via the `_get_verified_results` method and add the tool call results to `data`:
    ```python
    for i in range(len(data)):
        data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]
    ```
+ **Step 2**: Calculate the `step_mask` to determine the assignment position of the score in `reward_tensor`. In the current version, result reward calculation is still used, i.e., reward is only provided at the last token, with zeros elsewhere.
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
+ **Step 3**: Call `env_object.compute_score()` to calculate batch rewards in parallel, which can be done in two ways: 1) rule-based calculation and 2) model-judged calculation (see below for details on both).
    ```python
    def compute_score(self, reward_rollout_wg, reward_tokenizer, tokenizer, data: DataProto):
        if reward_rollout_wg is not None:
            scores = self._compute_score_with_reward_rollout_wg(reward_rollout_wg, reward_tokenizer, data)
        else:
            scores = self._compute_score_with_rules(data, tokenizer)
        
        return scores
    ```
+ **Step 4**: Assign the computed `scores` to the `reward_tensor`, keeping the same dimensions as `data.batch['responses']`.
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
    Note: Although the current framework only supports result-based rewards, for future extensibility to process-based rewards, `scores` is actually a `list[list[float]]`, where the first level represents the current batch size, and the second level represents the number of `reward`s in each batch (currently only 1). For process-based rewards, the assignment index in `reward_tensor` will be provided by the `step_mask` calculation logic.

## Rule-based Reward Calculation
+ The rule-based approach implements asynchronous **parallel reward calculation** based on asyncio in verl. Users only need to modify the `_compute_score_with_rules` method and define the reward function as in verl. For example, you can directly migrate the reward function [`qa_em.py`](https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py) from [Search-R1](https://github.com/PeterGriffinJin/Search-R1):
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

## Model Judge-based Reward Calculation
+ Model Judge-based reward calculation creates an additional `reward_rollout_wg` worker group and a dedicated resource pool for it, enabling distributed inference of Judge Models such as QwQ using vllm. In the `Env`, the Judge Model generates **model judgment results** to automatically evaluate the responses generated by the training model.
+ SEARL integrates resource pool allocation and vllm initialization for Judge LLMs. To use it, simply adjust the entry parameters:
  ```python
  reward_rollout.if_use_reward_rollout=True
  ```
+ This will automatically create a resource pool for the reward judge model. Example for model path parameter:
    ```python
    reward_rollout.rollout.model_name=huggingface.co/Qwen/QwQ-32B
    ```
    For other configuration parameters, see the `reward_rollout.rollout.` parameter settings.
+ During program execution, the new evaluation function `_compute_score_with_reward_rollout_wg` in the Env class will be called as shown below. The core functionality of this function is to construct prompts for the judge model based on the actor's generated response and the ground truth, then return the judge model's scores for each round of actor responses, which are used as training rewards.
    ```python
    def _compute_score_with_reward_rollout_wg(self, reward_rollout_wg, reward_tokenizer, data: DataProto):
        # Construct judge model prompts based on actor rollout responses and ground truth
        reward_prompt_strs = self.get_prompt_for_reward(reward_tokenizer, data)
        
        # Convert reward_prompt_strs to DataProto format, preparing input prompt data for the judge model
        ...
        
        # Judge model generates responses
        responses_data = reward_rollout_wg.generate_sequences(data_proto)
        
        # Calculate the score for each response
        flat_scores = []
        for i, temp_response_data in enumerate(responses_data):
            # Get the response_str output by each judge model
            ...
            response_str = reward_tokenizer.decode(temp_valid_response_ids, skip_special_tokens=True)

            # Calculate the score based on the ground truth label and the judge model's response_str
            score = self._compute_single_score_with_reward_rollout_wg(
                data_item.non_tensor_batch['data_source'], 
                response_str, 
                data_item.non_tensor_batch['reward_model']['ground_truth'], 
                data_item.non_tensor_batch.get('extra_info', None)
            )
            flat_scores.append(score)
        
        # Organize flat_scores into scores
        ...
        return scores
    ```
+ For example, in the search tool scenario of search-r1, the prompt and response for the judge model are as follows:
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
+ The `get_prompt_for_reward()` function is defined as follows, used to construct the judge model's prompt for each actor output:
    ```python
    def get_prompt_for_reward(self, reward_tokenizer, data: DataProto):
        # Construct judge model prompts for each actor output        
        for i in range(len(data)):
            # Get valid prompts and use mask_indices to control the reward position for each step
            ...
            
            for end_idx in mask_indices:
                # Truncate the current response and generate the prompt
                ...
            
                reward_prompt_str = self._get_single_prompt_str(
                    data_source=non_tensor_data['data_source'],
                    solution_str=current_response_str,
                    ground_truth=non_tensor_data['ground_truth'],
                    extra_info=non_tensor_data['extra_info'],
                    reward_tokenizer=reward_tokenizer
                )
                ...

        # Collect and organize judge model prompts into the reward_prompt_strs list
        ...
                
        return reward_prompt_strs
    
    ```
+ By modifying `system_content` and `user_content` in the `_get_single_prompt_str()` function called inside `get_prompt_for_reward()`, you can customize the prompts for the judge model.
    ```python
    def _get_single_prompt_str(self, data_source, solution_str, ground_truth, extra_info, reward_tokenizer):
            system_content = """
    You are the assistant to an intelligent search helper, responsible for evaluating the quality of search results. These evaluations will be used to adjust search queries and to better respond to users using the relevant information obtained from searches.
    Below are the outputs from a fine-tuned LLM and the corresponding ground truth labels. The model outputs are enclosed in <output></output> tags, while the ground truth labels are enclosed in <gt></gt> tags. Please assess the quality of the model's output, assign a score, and present the score within <score></score> tags.
    """
            user_content = """
    Model outputs: <output>{}</output>
    Ground Truth: <gt>{}</gt>
    """
            # Generate the prompt template for the judge model
            result = reward_tokenizer.apply_chat_template([
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content.format(solution_str, ground_truth['target'])}
                    ], add_generation_prompt=True, tokenize=False)
            return result
    ```
+ Finally, use the `_compute_single_score_with_reward_rollout_wg()` function to extract the single score from the judge model and return the score value. You can adjust the extraction rules as needed.
    ```python
    def _compute_single_score_with_reward_rollout_wg(self, data_source, solution_str, ground_truth, extra_info):
        pattern = r'<score>(.*?)</score>'
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return float(match.group(1).strip())
        else:
            return 0.0
    ```

## Using Tool Verification to Assist Reward Calculation
+ We have introduced the `verify_tool` method in the `Env`, which can be called during reward calculation to evaluate the original response. For example, in the NL2SQL task, the model's final output is a piece of code. SEARL can execute this SQL statement to **determine the correctness of the program**, and we also use an asynchronous parallel approach to **improve efficiency**.
+ For example, `verify_tool` can also call an API to implement the Model Judge function:
    + First, set the environment to require `verify_tool`:
        ```python
        self.use_verify_tool = True
        ```
    + Define the `verify_tool` method to call the API:
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
    + `verify_tool` is used in [`verl/workers/reward_manager/parallel.py`](../../../verl/workers/reward_manager/parallel.py), assigning the above results to `data`:
        ```python
        for i in range(len(data)):
            data[i].non_tensor_batch['reward_model']['ground_truth']['verified_results'] = results[i]
        ```
    + Finally, in `compute_scores`, calculate the `reward` based on `verified_results` and other information:
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