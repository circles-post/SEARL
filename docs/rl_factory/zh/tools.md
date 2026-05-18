# 工具调用环境
+ 这一部分的核心文件在`envs`文件夹下，本文档主要是代码拆解与逻辑介绍，如果想知道如何快速使用SEARL开始训练，请参考[`docs/rl_factory/main_tutorial.md`](../main_tutorial.md)。主要文档的主要内容包含：工具定义、工具解析、工具调用和训练。

## 工具定义
+ 以[`qwen_agent`](https://github.com/QwenLM/Qwen-Agent)为基础，支持自定义工具（继承自`BaseTool`）、MCP工具集以及[`qwen_agent`](https://github.com/QwenLM/Qwen-Agent)自带工具（例如`code_interpreter`）。推荐定义在tools文件夹下以保证项目的整洁，我们提供了一个search的实例，其参考自[search-r1](https://github.com/PeterGriffinJin/Search-R1): 
  + 下载语料库和数据集、处理数据集为parquet格式请参考[search-r1](https://github.com/PeterGriffinJin/Search-R1)
  + 运行bash文件前需要修改[`launch.sh`](../../../rag_server/launch.sh)中的参数（file_path是语料库的存储位置，retriever是模型intfloat/e5-base-v2的本地路径）
    ```bash
    # 运行bash文件以启动rag server
    bash rag_server/launch.sh
    ```
  + 我们在[`envs/tools/search.py`](../../../envs/tools/search.py)中实现了搜索工具的MCP格式注册
    ```python
    from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
    
    mcp = FastMCP("LocalServer")
    
    @mcp.tool()
    def query_rag(query: str, topk: int = 3) -> str:
        ...
    ```
  + 在`envs/configs`中新建MCP的配置文件[`mcp_tools.pydata`](../../../envs/configs/mcp_tools.pydata)
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
  并将`actor_rollout_ref.env.config_path`的参数设置为该配置文件的路径。

## 工具解析
+ 在`tool_manager`文件夹下注册你自己的`manager`，用于实现你自己的工具解析逻辑，当然你也可以直接用我们的[`qwen3_manager.py`](../../../envs/tool_manager/qwen3_manager.py)，利用`qwen_agent`的相关方法实现了以上三类工具的解析、启动和灵活的调用。

### 工具初始化
+ `QwenManager`继承自`ToolManager`并通过定义的`_build_tools`方法初始化工具，其中使用的`_init_tool`方法是参考自`qwen_agent`以支持三类工具的解析，并将工具添加到`self.tool_map`和`self.functions`中
    ```python
    def _init_tool(self, tool: Union[str, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                self.tool_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f'Tool {tool_name} is not registered.')

            self.tool_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)
    ```

### 回复解析
+ 使用`parse_response`解析模型的输出，并通过`parse_end_flag`解析输出中是否包含`<answer></answer>`，解析成功则得到答案并结束整个序列
+ 通过`parse_tools`方法将模型的输出解析为待执行的工具，针对Qwen系列模型的对话模版提取回复中的`<tool_call></tool_call>`，遵循以下逻辑：
    1. 如果没有成功提取到`<tool_call>`，则直接返回输入进来的`response`，并在`parse_response`中认定其是"answer"而结束整个序列
    2. 对于每一个被`<tool_call>`分割出的字符串，如果提取到`<tool_call>`但没提取到`</tool_call>`，则称这个工具为`empty`：
        ```json
        {
            "name": "<empty>", 
            "args": "# Extract the tool name failed"
        }
        ```
    3. 如果成功提取则尝试解析json并验证其中是否包含`name`和`arguments`的字段，否则仍将工具视为`empty`

### 工具执行
+ 在`execute_actions`方法中通过`asyncio`实现异步的并行调用。这一方法支持输入多个生成的`responses`，其主要存在两层结构
+ **第一层**：针对所有的`response`进行异步并行
    ```python
    def execute_actions(self, responses: List[str]):
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            tools.append(temp_tool_list)

        # 使用asyncio.run同步运行异步函数
        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools))
        except RuntimeError:
            # 如果事件循环已经在运行，则获取当前循环
            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools))
        
        return actions, tool_results
    ```
+ **第二层**：则在`execute_all_tools`内部实现了针对单个`response`内部的多工具执行进行异步并行
    ```python
    async def execute_all_tools(self, actions, tool_list):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            
        Returns:
            所有工具执行结果的列表
        """
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))
        
        results = await asyncio.gather(*tasks)
        
        return results
    ```
+ 执行工具的功能是在`_execute_tool_batch`中实现的，按照`action`选择执行逻辑：其中的`answer`和`error`的选项主要是为了保证框架兼容性，为`actions`时则调用全部解析出的工具并记录结果。

## 工具调用
+ 在`envs`文件夹下注册你自己的`Env`，用于实现你整体的**工具调用逻辑**以及**奖赏计算逻辑**。其中的step函数提供了默认的工具调用逻辑，输入一个batch的responses，输出工具调用的结果，真正地把**工具调用**当成了与LLM**解耦**的环境！
  + 在`Env`的`step`方法中，通过上面定义`execute_actions`返回工具调用结果
    ```python
    cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses)
    ```
  + 根据`action`判断"环境"返回状态和终止状态的逻辑，其中的一些变量主要目的是为了保证框架在后续的扩展性，并不影响核心功能。在下列程序中，推荐将`add_generation_prompt`设置为`True`以提示模型进行回复：
    ```python
    temp_next_obs = self.tool_manager.get_prompt(
        input_data=tool_result, 
        tokenizer=tokenizer,
        mode='tool_call',
        dd_generation_prompt=True
    )
    ```

## 多轮工具调用及训练
+ 在[`envs/utils/tool_utils.py`](../../../envs/utils/tool_utils.py)中实现多轮工具调用的prompt拼接、`loss_mask`定义及计算，**准确无误**地实现基于LLM的多轮工具调用！
    
    + 当设置 config.actor_rollout_ref.rollout.max_turns 时，框架将开启多轮工具调用训练模式，该模式当前是一种同步 rollout 的实现，同时支持 sglang 和 vllm 作为 rollout 引擎。多轮工具调用的入口如下，
        ```python
        def generate_sequences_loop(self, prompts: DataProto):
        ```
    + 在该函数内实现了多轮对话、工具调用的 prompt拼接，多轮对话的提前终止等逻辑，同时生成了 loss_mask，以屏蔽训练 actor 时，tool_call 部分的信息对loss的贡献。
        ```python
        prompt + rollout1 + tool_call1 + rollout2 + tool_call2 + ... + rolloutn + tool_calln
        ```