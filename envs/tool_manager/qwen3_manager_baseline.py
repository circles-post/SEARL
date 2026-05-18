import re
import copy
import json
import json5
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
import traceback
import time
import os
from pathlib import Path
from ast import literal_eval
from omegaconf import OmegaConf
from envs.tool_manager.base_manager import ToolManager
# from envs.utils.mcp_manager import MCPManager, BaseTool
from typing import Union, List, Tuple, Optional, Any, Dict
from envs.utils.util import ToolServiceError, DocParserError
from envs.utils.mcp_manager import MCPManager as SSEMCPManager
from qwen_agent.tools import TOOL_REGISTRY, MCPManager, BaseTool
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
from envs.utils.concurrency_limiter import ConcurrencyLimiter
from envs.utils.async_mcp_manager import AsyncMCPManager


def parse_mcp_tools_config(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # 支持在 .pydata 中使用 {project_name} 占位符，便于多项目隔离
        project_name = os.environ.get("PROJECT_NAME")
        if "{project_name}" in content:
            content = content.replace("{project_name}", project_name or "default_project")

        # 使用 literal_eval 安全地解析 Python 字面量
        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"解析错误: {e}")
        return None


class QwenManagerBaseline(ToolManager):    
    def __init__(self, verl_config):
        if isinstance(verl_config, dict):
            verl_config = OmegaConf.create(verl_config)
        super().__init__(verl_config)
        # 工具超时时间，防止单次调用挂死
        self.tool_timeout = getattr(verl_config, "tool_timeout", 20)
        
        # 创建并发限制器
        if self.verl_config.enable_limiter:
            global_limit = getattr(verl_config, 'max_concurrency', 32)
            self._limiter = ConcurrencyLimiter(global_limit=global_limit)
        else:
            self._limiter = None
            
        # 初始化线程池用于隔离 Search 和其他工具
        # search 工具往往并发高且IO密集，给予独立的线程池
        max_workers = getattr(verl_config, 'max_concurrency', 50)
        self._search_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SearchWorker")
        # 其他工具使用默认线程池，可以设置稍小的 worker 数，或者与 search 分开
        self._default_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DefaultToolWorker")

        # 初始化search缓存
        self._search_cache = {}
        self._search_cache_lock = None # 将在异步执行时动态创建
        self._in_flight_requests = {}
        self._in_flight_lock = None # 将在异步执行时动态创建
        self._lock_loop = None  # 记录当前锁所属的事件循环
        self._search_cache_refresh_interval = 15
        self._search_cache_mod_time = 0.0
        self._search_cache_last_check = 0.0
        self._search_cache_path = os.environ.get(
            "RL_FACTORY_SEARCH_CACHE_DIR",
            str(Path.home() / ".cache" / "rl_factory" / "search"),
        )
        self._debug_verbose = os.getenv("VERL_TOOL_DEBUG", "0") == "1"
        
        # 使用 getattr 提供默认值，防止配置缺失导致错误
        self.local_search = getattr(self.verl_config, 'local_search', False)
        print(f"Search Cache Config: local_search={self.local_search}")
        
        if self.local_search:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "local_bing_search_cache.jsonl"))
        else:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "bing_search_cache.jsonl"))
        
        print(f"Search Cache File: {self._search_cache_file}")
        self._load_search_cache()

    def get_tool(self, name_or_short_name: str):
        """通过名称或简写获取工具
        
        Args:
            name_or_short_name: 工具名称或简写
            
        Returns:
            找到的工具，如果没找到则返回None
        """
        name_or_short_name = str(name_or_short_name)
        return self.tool_map.get(name_or_short_name, None)
    
    @property
    def all_tools(self):
        """获取所有工具

        Returns:
            所有工具的列表
        """
        return self.tool_map

    def _load_search_cache(self) -> None:
        """加载搜索缓存"""
        if not self._search_cache_file.exists():
            return

        print(f"Loading search cache from {self._search_cache_file}...")
        loaded_cache = {}
        try:
            self._search_cache_mod_time = os.path.getmtime(self._search_cache_file)
            with open(self._search_cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        if "query" in entry and "response" in entry:
                            loaded_cache[entry["query"]] = entry["response"]
                    except (json.JSONDecodeError, KeyError):
                        print(f"Warning: Skipping malformed cache line: {line}")

            self._search_cache = loaded_cache
            self._search_cache_last_check = time.time()
            print(f"Loaded {len(self._search_cache)} search cache entries.")
        except Exception as e:
            print(f"Failed to load search cache file: {str(e)}")
            self._search_cache = {}

    def _check_search_cache_update(self) -> bool:
        """检查搜索缓存文件是否更新"""
        now = time.time()
        if now - self._search_cache_last_check < self._search_cache_refresh_interval:
            return False

        self._search_cache_last_check = now

        if not self._search_cache_file.exists():
            return False

        try:
            current_mod_time = os.path.getmtime(self._search_cache_file)
            if current_mod_time > self._search_cache_mod_time:
                print("Search cache file update detected, reloading")
                self._load_search_cache()
                return True
        except Exception as e:
            print(f"Failed to check search cache file updates: {str(e)}")

        return False

    def _ensure_async_structs(self):
        loop = asyncio.get_running_loop()
        if self._lock_loop is not loop or self._search_cache_lock is None or self._in_flight_lock is None:
            self._search_cache_lock = asyncio.Lock()
            self._in_flight_lock = asyncio.Lock()
            self._in_flight_requests = {}
            self._lock_loop = loop

    def _save_search_cache_sync(self, query: str, response: str) -> None:
        """同步保存搜索缓存到文件"""
        # 确保缓存目录存在
        os.makedirs(self._search_cache_path, exist_ok=True)

        new_entry = {"query": query, "response": response}
        try:
            print(f"DEBUG: Attempting to save search cache to {self._search_cache_file}")
            with open(self._search_cache_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(f"DEBUG: Successfully saved search cache entry. File size: {os.path.getsize(self._search_cache_file)}")
        except Exception as e:
            print(f"保存搜索缓存文件时失败: {str(e)}")
            traceback.print_exc()

    def _build_tools(self):
        config_path = self.verl_config.config_path
        if config_path is not None:
            function_list = parse_mcp_tools_config(config_path)

            if function_list:
                for tool in function_list:
                    self._init_tool(tool)
            
            self.functions = [func.function for func in self.tool_map.values()]
        else:
            print("The config_path is None!")
            self.functions = []

    async def execute_all_tools_with_limiter(self, actions, tools, temp_dict=None):
        """并行执行工具调用"""
        # 确保锁与当前事件循环匹配；同一 loop 内复用以启用 in-flight 去重
        self._ensure_async_structs()

        async def execute_single_tool(tool):
            """执行单个工具调用"""
            try:
                tool_name = tool.get("name", "default")
                future = None
                query_str = None
                if self._debug_verbose:
                    print("TOOL start", tool_name, "args_type", type(tool.get("args", "")))

                # Add MCP registration logic
                if tool_name == 'create_and_execute_mcp' and temp_dict is not None:
                    try:
                        args_content = tool.get("args", "")
                        if isinstance(args_content, str):
                            args = json.loads(args_content)
                        elif isinstance(args_content, dict):
                            args = args_content
                        else:
                            args = {}
                            
                        if 'name' in args and 'code' in args:
                            temp_dict[args['name']] = {
                                "mcp_name": args['name'],
                                "description": args.get('description', ''),
                                "arguments": args.get('arguments', ''),
                                "returns": args.get('returns', ''),
                                "code": args.get('code', ''),
                                "step_id": len(temp_dict) + 2 # Simple step_id logic
                            }
                            print(f"Agent Log: MCP {args['name']} registered in temp_dict")
                    except Exception as e:
                        print(f"Error registering MCP tool in temp_dict: {e}")

                # 如果是search工具，检查缓存和In-flight
                if tool_name == "search" or tool_name == "search-search":
                    query = tool.get("args", "")
                    if isinstance(query, dict):
                        query_str = json.dumps(query, sort_keys=True)
                    else:
                        query_str = str(query)

                    # 1. 检查缓存更新
                    self._check_search_cache_update()

                    # 2. 检查持久化缓存
                    async with self._search_cache_lock:
                        if query_str in self._search_cache:
                            print(f"Search cache hit: {query_str}")
                            if self._debug_verbose:
                                print("TOOL cache-hit", tool_name)
                            return self._search_cache[query_str]

                    # 3. In-flight Request 去重
                    async with self._in_flight_lock:
                        if query_str in self._in_flight_requests:
                            print(f"Cache hit (in-flight): Waiting for another task for '{query_str}'")
                            future = self._in_flight_requests[query_str]
                            try:
                                return await asyncio.wait_for(future, timeout=self.tool_timeout)
                            except asyncio.TimeoutError:
                                # 清理卡住的任务，允许后续请求重试
                                self._in_flight_requests.pop(query_str, None)
                                return f"# Tool calling exceeds timeout: {tool_name}, exceeded {self.tool_timeout}s"
                        
                        # 创建新的 Future 并注册
                        future = asyncio.get_running_loop().create_future()
                        self._in_flight_requests[query_str] = future
                        if self._debug_verbose:
                            print("TOOL done", tool_name)

                # 增加专门针对 python 代码执行工具的短超时配置
                # 虽然全局超时已经设置（例如60s），但 python 代码执行可能因为死循环等原因需要更严格的控制
                specific_timeout = self.tool_timeout
                if tool_name == "execute_python_code" or tool_name == "execute_python_code-execute_python_code":
                    # 尝试从 args 中读取 timeout 参数，如果存在的话
                    try:
                        if isinstance(tool.get("args", ""), str):
                            tool_args = json.loads(tool.get("args", ""))
                            if "timeout" in tool_args:
                                specific_timeout = float(tool_args["timeout"])
                        elif isinstance(tool.get("args", ""), dict):
                            if "timeout" in tool.get("args", ""):
                                specific_timeout = float(tool.get("args", "")["timeout"])
                    except:
                        pass
                    
                    # 默认给Python工具设置一个更短的超时时间，例如5秒
                    if specific_timeout == self.tool_timeout:
                         specific_timeout = 5.0

                async with self._limiter.limit(tool_name):
                    try:
                        if tool_name == "search" or tool_name == "search-search":
                            target_executor = self._search_executor
                        else:
                            target_executor = self._default_executor
                            
                        loop = asyncio.get_running_loop()
                        # run_in_executor returns a Future, not a coroutine, so we cannot use create_task directly on it
                        # We can await the Future directly or use wait_for on it
                        task = loop.run_in_executor(
                            target_executor, 
                            self._call_tool, 
                            tool_name, 
                            tool.get("args", "")
                        )
                        try:
                            result = await asyncio.wait_for(task, timeout=specific_timeout)
                        except asyncio.TimeoutError:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task
                            raise
                        except Exception:
                            # 其他异常直接传递
                            raise
                        if self._debug_verbose:
                            print("TOOL done", tool_name)

                        # 如果是search工具，保存到缓存并通知 waiters
                        if tool_name == "search" or tool_name == "search-search":
                            await asyncio.to_thread(self._save_search_cache_sync, query_str, result)
                            async with self._search_cache_lock:
                                self._search_cache[query_str] = result
                            
                            # 设置 In-flight 结果
                            if not future.done():
                                future.set_result(result)
                        
                        return result
                    except asyncio.TimeoutError as e:
                        # search 的 in-flight 等待者需要收到异常
                        if (tool_name == "search" or tool_name == "search-search") and not future.done():
                            future.set_exception(e)
                        return f"# Tool Calling Exceeds Timeout: {tool_name}, Exceeded {self.tool_timeout}s"
                    except Exception as e:
                        # 如果是 search 工具，处理 In-flight 异常
                        if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                            future.set_exception(e)
                        raise e
                    finally:
                        # 清理 In-flight 记录
                        if tool_name == "search" or tool_name == "search-search":
                            async with self._in_flight_lock:
                                if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                    del self._in_flight_requests[query_str]

            except Exception as e:
                print(f"工具调用失败: {e}")
                if self._debug_verbose:
                    print("TOOL error", tool.get("name", "default"))
                return f"# Tool Calling Failed: {str(e)}"
        
        async def process_batch_item(action, tool_list):
            if action == 'answer':
                # 'tools' is a str (the answer)
                return {'role': 'assitant', 'content': tool_list}
            elif action == 'error':
                # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
                return {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tool_list)}
            elif action == 'actions':
                # 'tools' is the list of the 'Tool' instances
                item_tasks = [execute_single_tool(temp_tool) for temp_tool in tool_list]
                if not item_tasks:
                    return []
                item_results = await asyncio.gather(*item_tasks)
                return [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in item_results]
            else:
                raise ValueError('Unexpected action: {}'.format(action))

        tasks = [process_batch_item(a, t) for a, t in zip(actions, tools)]
        results = await asyncio.gather(*tasks)
        return results
    
    async def execute_all_tools(self, actions, tool_list, temp_dict=None):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            temp_dict: 用于存储动态创建的MCP工具的字典
            
        Returns:
            所有工具执行结果的列表
        """
        # 确保锁与当前事件循环匹配；同一 loop 内复用以启用 in-flight 去重
        self._ensure_async_structs()
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_dict))
        
        results = await asyncio.gather(*tasks)
        
        return results
        
    async def _execute_tool_batch(self, action, tools, temp_dict=None):
        """异步并行执行一批工具
        
        Args:
            tools: 工具列表
            temp_dict: 用于存储动态创建的MCP工具的字典
            
        Returns:
            工具执行结果的列表
        """        
        async def execute_single_tool(tool):
            try:
                tool_name = tool.get("name", "default")
                future = None
                query_str = None
                
                # Add MCP registration logic
                if tool_name == 'create_and_execute_mcp' and temp_dict is not None:
                    try:
                        args_content = tool.get("args", "")
                        if isinstance(args_content, str):
                            args = json.loads(args_content)
                        elif isinstance(args_content, dict):
                            args = args_content
                        else:
                            args = {}
                            
                        if 'name' in args and 'code' in args:
                            temp_dict[args['name']] = {
                                "mcp_name": args['name'],
                                "description": args.get('description', ''),
                                "arguments": args.get('arguments', ''),
                                "returns": args.get('returns', ''),
                                "code": args.get('code', ''),
                                "step_id": len(temp_dict) + 2
                            }
                            print(f"Agent Log: MCP {args['name']} registered in temp_dict")
                    except Exception as e:
                        print(f"Error registering MCP tool in temp_dict: {e}")

                # 如果是search工具，检查缓存和In-flight
                if tool_name == "search" or tool_name == "search-search":
                    query = tool.get("args", "")
                    if isinstance(query, dict):
                        query_str = json.dumps(query, sort_keys=True)
                    else:
                        query_str = str(query)

                    # 检查缓存更新
                    self._check_search_cache_update()

                    # 检查内存缓存
                    async with self._search_cache_lock:
                        if query_str in self._search_cache:
                            print(f"Search cache hit: {query_str}")
                            return self._search_cache[query_str]

                    # In-flight Request 去重
                    async with self._in_flight_lock:
                        if query_str in self._in_flight_requests:
                            print(f"Cache hit (in-flight): Waiting for another task for '{query_str}'")
                            future = self._in_flight_requests[query_str]
                            try:
                                return await asyncio.wait_for(
                                    future,
                                    timeout=self.tool_timeout
                                )
                            except asyncio.TimeoutError:
                                self._in_flight_requests.pop(query_str, None)
                                return f"# Tool Calling Exceeds Timeout: {tool_name}, Exceeded {self.tool_timeout}s"
                        
                        future = asyncio.get_running_loop().create_future()
                        self._in_flight_requests[query_str] = future

                tool_instance = self.get_tool(tool_name)
                args = tool.get("args", "")
                if tool_instance is not None:
                    try:
                        args = json.loads(args)
                    except Exception as e:
                        pass

                    if type(args) is dict:
                        try:
                            # 使用asyncio.to_thread包装self._call_tool以保持异步特性，并添加超时
                            print(f"DEBUG: Executing tool {tool_name}...")
                            exec_task = asyncio.create_task(
                                asyncio.to_thread(
                                    self._call_tool, 
                                    tool_name, json.dumps(args, ensure_ascii=False, indent=4)
                                )
                            )
                            try:
                                tool_result = await asyncio.wait_for(
                                    exec_task,
                                    timeout=self.tool_timeout
                                )
                            except asyncio.TimeoutError:
                                exec_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError):
                                    await exec_task
                                raise
                            except Exception:
                                raise
                            print(f"DEBUG: Tool execution finished. Result length: {len(str(tool_result))}")
                            
                            # 如果是search工具，保存到缓存并设置 Future 结果
                            if tool_name == "search" or tool_name == "search-search":
                                print("DEBUG: Triggering search cache save...")
                                if isinstance(tool.get("args", ""), dict):
                                    query_str = json.dumps(tool.get("args", ""), sort_keys=True)
                                else:
                                    query_str = str(tool.get("args", ""))
                                    
                                await asyncio.to_thread(self._save_search_cache_sync, query_str, tool_result)
                                async with self._search_cache_lock:
                                    self._search_cache[query_str] = tool_result
                                
                                if not future.done():
                                    future.set_result(tool_result) # 这里应该返回 tool_result 还是格式化后的 result? 
                                print("DEBUG: Search cache save task completed.")

                            result = """# Execute the tool {} successed
  - The result is:
{}""".format(tool_name, tool_result)


                            
                        except asyncio.TimeoutError as e:
                            # 处理 In-flight 超时
                            if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                                future.set_exception(e)
                            result = """# Execute the tool {} timeout
  - Timeout (s): {}
""".format(tool_name, self.tool_timeout)
                        except Exception as e:
                            # 处理 In-flight 异常
                            if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                                future.set_exception(e)
                                
                            result = """# Execute the tool {} failed
  - Error message:
{}""".format(tool_name, str(e))
                        finally:
                            # 清理 In-flight
                            if tool_name == "search" or tool_name == "search-search":
                                async with self._in_flight_lock:
                                    if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                        del self._in_flight_requests[query_str]

                    elif type(args) is str:
                        # Json decode error: xxx
                        result = 'parse json failed, argument is: {}'.format(args)
                        if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                            future.set_result(result)
                    else:
                        result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
                        if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                            future.set_result(result)
                else:
                    if tool_name == '<empty>':
                        result = 'toolname is empty, argument is: '.format(args)
                    else:
                        result = "# Failed to find the tool {} in the tool map".format(tool_name)
                    if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                        future.set_result(result)

                return result
            except Exception as e:
                print(f"工具调用失败: {e}")
                if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                    future.set_exception(e)
                return f"# Tool Calling Failed: {str(e)}"
        
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*tasks)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results

    def _init_tool(self, tool: Union[str, BaseTool]):
        print(f'tool: {tool}')
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            self.tool_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            print(f'MCP is using {self.verl_config.mcp_mode} mode')
            if self.verl_config.mcp_mode == 'sse':
                if self.verl_config.parallel_sse_tool_call.is_enabled:
                    tools = AsyncMCPManager(num_instances=self.verl_config.parallel_sse_tool_call.num_instances).initConfig(tool)
                else:
                    tools = SSEMCPManager().initConfig(tool)
            elif self.verl_config.mcp_mode == 'stdio':
                tools = MCPManager().initConfig(tool)
            else:
                raise ValueError(f"Unexpected mcp mode: {self.verl_config.mcp_mode}")
            
            for tool in tools:
                tool_name = tool.name
                self.tool_map[tool_name] = tool
                print(f'register tool: {tool_name} --> {tool}')
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

        # select used tools within one sse link before tool learning
        if self.verl_config.mcp_mode == 'sse' and len(self.verl_config.tool_name_selected) != 0:
            try:
                self.tool_map = {each_tool_name: self.tool_map[each_tool_name] for each_tool_name in self.verl_config.tool_name_selected}
            except:
                raise ValueError('Selected tool names are not valid or available sse tool list error. Available tool names: {}'.format(self.tool_map.keys()))

    def execute_actions(self, responses: List[str], temp_dict: Optional[Dict] = None):
        actions, tools = [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            temp_tool_list = self.full_name(temp_tool_list)
            tools.append(temp_tool_list)

        # 使用asyncio.run同步运行异步函数
        try:
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = asyncio.run(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
            else:
                tool_results = asyncio.run(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
        except RuntimeError:
            # 如果事件循环已经在运行，则获取当前循环
            loop = asyncio.get_event_loop()
            if self.verl_config.enable_limiter:
                assert self._limiter is not None, "Limiter is not enabled"
                tool_results = loop.run_until_complete(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
            else:
                tool_results = loop.run_until_complete(self.execute_all_tools_with_limiter(actions, tools, temp_dict))
        
        return actions, tool_results
    
    def parse_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """执行动作
        
        Args:
            response_content: 响应文本
            
        Returns:
            解析后的动作信息，包含name和args（如果存在）
            如果没有动作标签或格式不正确，返回None
        """
        # 提取answers
        if_answer, answer = self.parse_end_flag(response_content)
        if if_answer:
            return 'answer', answer
        
        # 提取tools
        tools = self.parse_tools(response_content)
        if type(tools) == list:
            return 'actions', tools
        else:
            assert type(tools) == str
            # if the response is not a tool call, it is an answer
            return 'answer', tools
    
    def parse_end_flag(self, response_content: str) -> tuple[bool, str]:
        """解析答案标签，支持 \boxed{} 格式和其他答案格式"""
        answer = None
        # 匹配 <answer> 标签内的内容（包括嵌套标签）
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_match = re.search(answer_pattern, response_content, re.DOTALL)

        if answer_match:
            # 提取标签内的内容
            answer_content = answer_match.group(1).strip()

            # 如果内容包含 \boxed{}，直接返回
            if r'\boxed{' in answer_content:
                return True, answer_content

            # 尝试提取 \boxed{} 内部的内容
            boxed_pattern = r'\\boxed\{([^}]+)\}'
            boxed_match = re.search(boxed_pattern, answer_content)
            if boxed_match:
                return True, boxed_match.group(1)

            # 如果没有 \boxed{} 格式，直接返回内容
            return True, answer_content

        return False, None
    
    def parse_tools(self, response: str):
        parsed_tools = []
        i = response.find('<tool_call>')
        # If no function call:
        if i < 0:
            j = response.find('</tool_call>')
            if j < 0:
                return response
            else:
                parsed_tools.append({
                        "name": "<error>",
                        "args": "# Extract the tool name failed"
                    })
                return parsed_tools

        # split tool-call to separate assistant msg
        tool_call_list = response.split('<tool_call>')
        pre_thought = tool_call_list[0].strip()
        for txt in tool_call_list[1:]:
            if not txt.strip():
                continue

            if '</tool_call>' not in txt:
                # incomplete </tool_call>: This is to better represent incomplete tool calls in streaming output
                fn_name = '<empty>'
                fn_args = """# Extract the tool name failed"""
                parsed_tools.append(
                    {
                        "name": fn_name,
                        "args": fn_args,
                    }
                )
            else:
                one_tool_call_txt = txt.split('</tool_call>')
                try:
                    # 检查分割后是否有有效内容
                    if not one_tool_call_txt[0].strip():
                        raise ValueError("Empty tool call content")
                        
                    # 尝试解析JSON
                    fn = json5.loads(one_tool_call_txt[0].strip())
                    
                    # 检查必须字段是否存在
                    if type(fn) is not dict or 'name' not in fn or 'arguments' not in fn:
                        raise KeyError("Missing required fields")
                    
                    # 解析成功的情况
                    parsed_tools.append({
                        "name": fn['name'],
                        "args": json.dumps(fn['arguments'], ensure_ascii=False, indent=4),
                    })
                
                except (IndexError, KeyError, ValueError) as e:
                    # 所有可能的错误类型处理
                    parsed_tools.append({
                        "name": "<empty>",
                        "args": "# Extract the tool name failed"
                    })
        
        if len(parsed_tools) == 0 :
            # <tool_call> is last token
            fn_name = '<empty>'
            fn_args = """# Extract the tool name failed"""
            parsed_tools.append(
                {
                    "name": fn_name,
                    "args": fn_args,
                })

        return parsed_tools
    
    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, tools=self.functions, 
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = [{'role': role, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))
            
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=self.functions, 
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        
        return prompt_with_chat_template


# 定义全局变量

if __name__ == "__main__":
    import os
    import sys

    def test_qwen_manager():
        """测试 QwenManager 的基本功能"""
        print("🚀 开始测试 QwenManager...")

        # 创建模拟配置
        test_config = {
            'config_path': 'envs/configs/mcp_tools_local.pydata',
            'mcp_mode': 'stdio',  # 或 'sse'
            'enable_limiter': False,
            'tool_name_selected': [],  # 空列表表示使用所有工具
            'use_storage_manager': False,
            'enable_thinking': False,
            'local_search': True
        }

        try:
            print("1. 初始化 QwenManager...")
            manager = QwenManagerBaseline(test_config)
            print("✅ QwenManager 初始化成功")

            print("\n2. 检查工具映射...")
            print(f"   可用工具数量: {len(manager.tool_map)}")
            for tool_name, tool in manager.tool_map.items():
                print(f"   - {tool_name}: {type(tool).__name__}")
            print("✅ 工具映射检查完成")

            print("\n3. 测试获取工具...")
            if manager.tool_map:
                first_tool_name = list(manager.tool_map.keys())[0]
                tool = manager.get_tool(first_tool_name)
                if tool:
                    print(f"✅ 成功获取工具: {first_tool_name}")
                else:
                    print(f"❌ 获取工具失败: {first_tool_name}")
            else:
                print("⚠️ 没有可用工具")

            print("\n4. 测试工具执行...")
            # 创建一个简单的测试响应，使用时间戳确保不命中缓存
            import time
            unique_query = f"你是谁"
            test_response = f"""<tool_call>
{{
    "name": "search",
    "arguments": {{
        "query": "{unique_query}",
        "topk": 2
    }}
}}
</tool_call>"""

            try:
                action, tools = manager.parse_response(test_response)
                print(f"   解析结果 - 动作: {action}")
                if action == 'actions':
                    print(f"   工具数量: {len(tools)}")
                    for i, tool in enumerate(tools):
                        print(f"   工具{i+1}: {tool}")

                    # 尝试执行工具（如果有的话）
                    if tools:
                        print("   尝试执行工具...")
                        actions_list, tool_results = manager.execute_actions([test_response])
                        print("   执行结果:")
                        for i, result in enumerate(tool_results):
                            print(f"   结果{i+1}: {result[:200]}..." if len(str(result)) > 200 else f"   结果{i+1}: {result}")
                else:
                    print(f"   响应内容: {tools}")
                print("✅ 工具执行测试完成")

            except Exception as e:
                print(f"❌ 工具执行测试失败: {str(e)}")
                traceback.print_exc()

            print("\n5. 测试 Python 工具...")
            # 测试 Python 代码执行工具
            python_test_response = """<tool_call>
{"name": "create_and_execute_mcp-create_and_execute_mcp", "arguments": {"name": "find_min_xyz", "description": "Find the minimum value of the product xyz where x, y, z are odd primes satisfying x divides (y^5 + 1), y divides (z^5 + 1), and z divides (x^5 + 1)", "arguments": "x, y, z (int)", "returns": "product (int)", "code": "def find_min_xyz(x, y, z):\n    # Check if x divides (y^5 + 1)\n    if (y**5 + 1) % x != 0:\n        return None\n    # Check if y divides (z^5 + 1)\n    if (z**5 + 1) % y != 0:\n        return None\n    # Check if z divides (x^5 + 1)\n    if (x**5 + 1) % z != 0:\n        return None\n    # If all conditions are satisfied, return the product\n    return x * y * z\n\n# Try small odd primes\nprimes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]\nfor x in primes:\n    for y in primes:\n        for z in primes:\n            result = find_min_xyz(x, y, z)\n            if result is not None:\n                return result", "inputs": {"x": 3, "y": 3, "z": 3}, "timeout": 15.0}}
</tool_call>"""

            try:
                action, tools = manager.parse_response(python_test_response)
                print(f"   Python 工具解析结果 - 动作: {action}")
                if action == 'actions' and tools:
                    print("   尝试执行 Python 工具...")
                    actions_list, tool_results = manager.execute_actions([python_test_response])
                    print("   Python 工具执行结果:")
                    for i, result in enumerate(tool_results):
                        print(f"   结果{i+1}: {result[:300]}..." if len(str(result)) > 300 else f"   结果{i+1}: {result}")

                    # 检查结果是否包含预期的输出
                    result_str = str(tool_results[0]) if tool_results else ""
                    if "Hello from test!" in result_str or "2 + 3 = 5" in result_str:
                        print("✅ Python 工具执行结果正确")
                    else:
                        print("⚠️ Python 工具执行结果可能不正确")
                else:
                    print("⚠️ Python 工具解析失败")
                print("✅ Python 工具测试完成")
            except Exception as e:
                print(f"❌ Python 工具测试失败: {str(e)}")
                traceback.print_exc()

            print("\n6. 测试答案解析...")
            test_answers = [
                "<answer>\\boxed{[2, 1]}</answer>",
                "<answer>42</answer>",
                "<answer>\\boxed{42}</answer>",
                "Some text <answer>\\boxed{[2, 1]}</answer> more text"
            ]

            for test_answer in test_answers:
                try:
                    found, parsed = manager.parse_end_flag(test_answer)
                    print(f"   输入: {test_answer}")
                    print(f"   解析结果: found={found}, parsed='{parsed}'")
                except Exception as e:
                    print(f"   解析失败: {str(e)}")

            print("✅ 答案解析测试完成")

            print("\n7. 测试搜索缓存...")
            if hasattr(manager, '_search_cache'):
                print(f"   缓存条目数量: {len(manager._search_cache)}")
                print("✅ 搜索缓存检查完成")
            else:
                print("⚠️ 搜索缓存未初始化")

            return True

        except Exception as e:
            print(f"❌ 测试失败: {str(e)}")
            traceback.print_exc()
            return False

    def test_config_parsing():
        """测试配置文件解析"""
        print("\n🔧 测试配置文件解析...")

        # 使用绝对路径或确保相对路径正确
        config_path = 'envs/configs/mcp_tools.pydata'
        if not os.path.exists(config_path):
            # Fallback to local path if absolute path doesn't exist (e.g. different environment)
            config_path = str(Path(__file__).resolve().parents[2] / 'envs/configs/mcp_tools.pydata')

        print(f"   Checking config path: {config_path}")
        
        if os.path.exists(config_path):
            try:
                function_list = parse_mcp_tools_config(config_path)
                if function_list is None:
                    print("❌ 配置文件解析返回 None")
                    return False
                    
                print(f"✅ 配置文件解析成功，包含 {len(function_list)} 个工具配置")
                for i, tool_config in enumerate(function_list):
                    print(f"   配置{i+1}: {tool_config}")
                return True
            except Exception as e:
                print(f"❌ 配置文件解析失败: {str(e)}")
                return False
        else:
            print(f"❌ 配置文件不存在: {config_path}")
            return False

    # 运行所有测试
    print("=" * 60)
    print("QwenManager 调试测试")
    print("=" * 60)

    # 测试配置文件解析
    config_test_passed = test_config_parsing()

    # 测试管理器功能
    manager_test_passed = test_qwen_manager()

    # 输出总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"配置文件解析测试: {'✅ 通过' if config_test_passed else '❌ 失败'}")
    print(f"管理器功能测试: {'✅ 通过' if manager_test_passed else '❌ 失败'}")

    if config_test_passed and manager_test_passed:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("⚠️ 部分测试失败，请检查相关配置和服务")
        sys.exit(1)
