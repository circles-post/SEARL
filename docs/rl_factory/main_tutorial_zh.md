# 最小实现教程
+ 本文档以复现[Search-R1](https://github.com/PeterGriffinJin/Search-R1)为例，说明了如何使用我们的SEARL实现你自己的Multi-turn tool-use RL post-training。总体而言，你只需要完成“**工具定义**”和“**奖赏函数定义**”即可开始训练！
+ **步骤1 - 工具定义**：以[qwen_agent](https://github.com/QwenLM/Qwen-Agent)为基础，支持自定义工具（继承自`BaseTool`）、MCP工具集以及`qwen_agent`自带工具（例如`code_interpreter`）。推荐定义在tools文件夹下以保证项目的整洁，我们提供了一个search的实例，其参考自[search-r1](https://github.com/PeterGriffinJin/Search-R1): 
  + **准备RAG Server** - 参考自[search-r1](https://github.com/PeterGriffinJin/Search-R1)，其构建了一个离线的语料库，使用RAG的形式搭建wiki并暴露了一个搜索接口
    + **下载语料库**
        ```bash
        save_path=/the/path/to/save
        python rag_server/download.py --save_path $save_path
        cat $save_path/part_* > $save_path/e5_Flat.index
        gzip -d $save_path/wiki-18.jsonl.gz
        ```
    + **处理数据集**
        ```bash
        python rag_server/data_process/nq_search.py
        # or
        tar -zxvf rag_server/nq_search.tar.gz
        mv nq_search/ data/
        ```
    + **运行RAG Server**：运行bash文件前需要修改launch.sh中的参数（file_path是语料库的存储位置，retriever是模型intfloat/e5-base-v2的本地路径）
        ```bash
        bash rag_server/launch.sh
        ```
  + **准备MCP启动文件**：在`envs/tools/search.py`中实现搜索工具的MCP格式注册
    ```python
    from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
    
    mcp = FastMCP("LocalServer")
    
    @mcp.tool()
    def query_rag(query: str, topk: int = 3) -> str:
        ...
    ```
  + **准备MCP配置文件**：在`envs/configs`中新建MCP配置文件`mcp_tools.pydata`
    ```json
    [
        {'mcpServers': {
            'time': {
                'command': 'python',
                'args': ['envs/tools/search.py']
            }
        }}
    ]
    ```
    在`main_grpo.sh`中将`actor_rollout_ref.env.config_path`的参数设置为该配置文件的路径。
+ **步骤2 - 奖赏函数定义**：该奖赏函数主要逻辑参考自[search-r1](https://github.com/PeterGriffinJin/Search-R1) - `verl/utils/reward_score/qa_em.py`
  + **新建**一个环境文件`envs/search.py`
    ```python
    import re
    import string
    import random
    from .base import Env

    class SearchEnv(Env):
        def __init__(self, config):
            super().__init__(config)
            self.use_verify_tool = False

        # NOTE: Add your reward calculation rules here!
        def _compute_score_with_rules(self, data, tokenizer):
            ...
    ```
  + 在`envs/__init__.py`中**注册**该环境文件为`search`
    ```python
    from .base import Env as BaseEnv
    from .search import SearchEnv

    __all__ = ['BaseEnv', 'SearchEnv','...']

    TOOL_ENV_REGISTRY = {
        'base': BaseEnv,
        'search': SearchEnv,
    }
    ```
  + **基于规则**的形式定义奖赏函数
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
    上述程序的关键核心在于`compute_score_em(...)`，其他的处理程序可以不用修改
+ **步骤3 - 开始训练！**
  + 修改脚本文件`main_grpo.sh`
    ```bash
    set -e -x

    export MODEL_PATH=/the/path/to/model
    export REWARD_MODEL_PATH=/the/path/to/reward_rollout_model

    python3 -m verl.trainer.main_ppo\
    ...
    actor_rollout_ref.env.name=search\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=/the/path/to/mcp_tools.pydata\
    ...
    ```
  + 运行训练程序
    ```bash
    bash main_grpo.sh
    ```
+ **训练结束后 - 查看实验结果**
    ```bash
    tensorboard --logdir=./
    ```