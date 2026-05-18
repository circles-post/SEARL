# Tool-Calling Environment
+ The core files for this section are located in the `envs` folder. This document mainly explains the code structure and logic. If you want to quickly learn how to use SEARL for training, please refer to [`docs/rl_factory/main_tutorial.md`](../main_tutorial.md). The main contents of this document include: tool definition, tool parsing, tool calling, and training.

## Tool Definition
+ Based on [`qwen_agent`](https://github.com/QwenLM/Qwen-Agent), custom tools (inheriting from `BaseTool`), MCP toolsets, and built-in tools from [`qwen_agent`](https://github.com/QwenLM/Qwen-Agent) (such as `code_interpreter`) are supported. It is recommended to define your tools under the `tools` folder to keep the project organized. We provide an example of a search tool, referring to [search-r1](https://github.com/PeterGriffinJin/Search-R1):
  + For downloading the corpus and dataset, and processing the dataset into parquet format, please refer to [search-r1](https://github.com/PeterGriffinJin/Search-R1).
  + Before running the bash file, you need to modify the parameters in [`launch.sh`](../../../rag_server/launch.sh) (`file_path` is the storage location of the corpus, `retriever` is the local path of the model intfloat/e5-base-v2)
    ```bash
    # Run the bash file to start the rag server
    bash rag_server/launch.sh
    ```
  + We implement MCP-format registration for the search tool in [`envs/tools/search.py`](../../../envs/tools/search.py)
    ```python
    from mcp.server.fastmcp import FastMCP  # Assuming you have this base library
    
    mcp = FastMCP("LocalServer")
    
    @mcp.tool()
    def query_rag(query: str, topk: int = 3) -> str:
        ...
    ```
  + Create a new MCP config file [`mcp_tools.pydata`](../../../envs/configs/mcp_tools.pydata) in `envs/configs`
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
  Then set the parameter `actor_rollout_ref.env.config_path` to the path of this config file.

## Tool Parsing
+ Register your own `manager` under the `tool_manager` folder to implement your own tool parsing logic. Of course, you can also directly use our [`qwen3_manager.py`](../../../envs/tool_manager/qwen3_manager.py), which uses the related methods of `qwen_agent` to implement parsing, launching, and flexible calling for all three types of tools.

### Tool Initialization
+ `QwenManager` inherits from `ToolManager` and initializes tools via the defined `_build_tools` method. The `_init_tool` method, referenced from `qwen_agent`, supports parsing all three types of tools and adds them to `self.tool_map` and `self.functions`.
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

### Response Parsing
+ Use `parse_response` to parse the model’s output, and use `parse_end_flag` to check whether the output contains `<answer></answer>`. If parsing is successful, the answer is obtained and the entire sequence ends.
+ The `parse_tools` method parses the model output into tools to be executed. For Qwen series models, the dialogue template extracts `<tool_call></tool_call>` from the reply, following this logic:
    1. If `<tool_call>` is not successfully extracted, directly return the input `response`, and determine it as "answer" in `parse_response`, ending the sequence.
    2. For each string split by `<tool_call>`, if `<tool_call>` is extracted but `</tool_call>` is not, this tool is called `empty`:
        ```json
        {
            "name": "<empty>", 
            "args": "# Extract the tool name failed"
        }
        ```
    3. If parsing is successful, try to parse the JSON and verify whether it contains `name` and `arguments` fields; otherwise, the tool is still regarded as `empty`.

### Tool Execution
+ The `execute_actions` method uses `asyncio` to implement asynchronous parallel execution. This method supports inputting multiple generated `responses` and mainly has two layers of structure.
+ **First Layer**: Asynchronous parallel execution for all `responses`:
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

        # Use asyncio.run to synchronously run the async function
        try:
            tool_results = asyncio.run(self.execute_all_tools(actions, tools))
        except RuntimeError:
            # If the event loop is already running, get the current loop
            loop = asyncio.get_event_loop()
            tool_results = loop.run_until_complete(self.execute_all_tools(actions, tools))
        
        return actions, tool_results
    ```
+ **Second Layer**: In `execute_all_tools`, asynchronous parallel execution is implemented for multiple tools within a single `response`:
    ```python
    async def execute_all_tools(self, actions, tool_list):
        """Asynchronously and in parallel execute all tool lists
        
        Args:
            tool_list: List of lists of tools
            
        Returns:
            List of results from all tool executions
        """
        
        # Parallel execution for each tool list
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list))
        
        results = await asyncio.gather(*tasks)
        
        return results
    ```
+ The tool execution logic is implemented in `_execute_tool_batch`. The `answer` and `error` options are mainly for framework compatibility. For `actions`, all parsed tools are called, and results are recorded.

## Tool Calling
+ Register your own `Env` in the `envs` folder to implement your overall **tool-calling logic** and **reward calculation logic**. The `step` function provides the default tool-calling logic: input a batch of responses and output the tool-calling results, truly decoupling **tool calling** from the LLM as an **environment**!
  + In the `step` method of `Env`, use the above-defined `execute_actions` to return tool-calling results:
    ```python
    cur_actions, tool_results = self.tool_manager.execute_actions(responses=responses)
    ```
  + Use `action` to determine the logic for the environment's return status and termination status. Some variables here are mainly to ensure extensibility of the framework and do not affect core functions. In the following code, it is recommended to set `add_generation_prompt=True` to prompt the model to reply:
    ```python
    temp_next_obs = self.tool_manager.get_prompt(
        input_data=tool_result, 
        tokenizer=tokenizer,
        mode='tool_call',
        dd_generation_prompt=True
    )
    ```

## Multi-turn Tool Calling and Training
+ In [`envs/utils/tool_utils.py`](../../../envs/utils/tool_utils.py), implement prompt concatenation, `loss_mask` definition and calculation for multi-turn tool calling, **accurately** enabling multi-turn tool calling based on LLM!
    
    + When `config.actor_rollout_ref.rollout.max_turns` is set, the framework will enable multi-turn tool-calling training mode. This mode is currently a synchronous rollout implementation, supporting both sglang and vllm as rollout engines. The entry for multi-turn tool calling is as follows:
        ```python
        def generate_sequences_loop(self, prompts: DataProto):
        ```
    + This function implements multi-turn dialogue, prompt concatenation for tool calling, early termination logic for multi-turn dialogue, and generates a loss_mask to mask out the contribution of tool_call parts to the loss when training the actor.
        ```python
        prompt + rollout1 + tool_call1 + rollout2 + tool_call2 + ... + rolloutn + tool_calln
        ```