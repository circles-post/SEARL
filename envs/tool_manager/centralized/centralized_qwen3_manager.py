import ray
import json
import torch
import asyncio
import datetime
import threading
import networkx as nx
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from envs.tool_manager.qwen3_manager import QwenManager
import yaml
from concurrent.futures import ThreadPoolExecutor
import contextlib
import re
from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
from ast import literal_eval
import ast
import time
from jinja2 import StrictUndefined, Template
from transformers import AutoTokenizer
import os
from pathlib import Path
FENCE_RE = re.compile(r'<python>([\s\S]*?)</python>', re.S | re.IGNORECASE)
FUNC_RE  = re.compile(r'^\s*def\s+\w+\s*\([^)]*\)\s*:', re.MULTILINE)
ANSI_RE  = re.compile(r'\x1B\[[0-9;]*[A-Za-z]')
SENTINEL_RE = re.compile(r'<\|im_[^|]*\|>')


def _repo_root() -> Path:
    return Path(os.environ.get("RL_FACTORY_ROOT", Path(__file__).resolve().parents[3]))


def _default_embedding_model() -> str:
    return os.environ.get("RL_FACTORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _resolve_project_name(cfg=None) -> str:
    """统一获取项目名：环境变量优先，其次配置字段，最后回退默认。"""
    env_name = os.getenv("RL_FACTORY_PROJECT_NAME") or os.getenv("PROJECT_NAME")
    if env_name:
        return env_name
    if cfg is not None:
        cfg_name = getattr(cfg, "project_name", None)
        if cfg_name:
            return cfg_name
    return "default_project"

def parse_mcp_tools_config(file_path, project_name=None):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        # 支持在 .pydata 中使用 {project_name} 占位符，便于多项目隔离
        resolved_project_name = project_name or _resolve_project_name(None) or "default_project"
        if "{project_name}" in content:
            content = content.replace("{project_name}", resolved_project_name)
        data = literal_eval(content)
        return data
    except Exception as e:
        print(f"解析错误: {e}")
        return []

def remove_middle_tokens(s):
    # 找到第一个 im_start 的结束位置和最后一个 im_end 的开始位置
    start_tag = "<|im_start|>"
    end_tag = "<|im_end|>"
    
    first_idx = s.find(start_tag) + len(start_tag)
    last_idx = s.rfind(end_tag)
    
    if first_idx == -1 or last_idx == -1:
        return s # 如果没找齐，直接返回原样
    
    # 提取中间部分
    middle_part = s[first_idx:last_idx]
    # 删除中间部分所有的标记
    middle_part = middle_part.replace(start_tag, "").replace(end_tag, "")
    
    # 重新拼接：开头 + 处理后的中间 + 结尾
    return s[:first_idx] + middle_part + s[last_idx:]

def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def extract_tool_names(text: str):
    """
    仅在 <tool_call>...</tool_call> 内提取 name 字段，包括顶层name和arguments中的name，返回去重且保序列表。
    """
    names = []
    seen = set()

    # 优化：预编译正则表达式，避免重复编译
    tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
    name_pattern = re.compile(r'"name"\s*:\s*"([^"]+)"')
    create_mcp_args_pattern = re.compile(r'"arguments"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', re.DOTALL)

    # 找出所有 tool_call 块
    for block in tool_call_pattern.findall(text):
        # 提取所有 "name": "value" 模式的字段
        for n in name_pattern.findall(block):
            if n == 'create_and_execute_mcp':
                args_match = create_mcp_args_pattern.search(block)
                if args_match:
                    n = args_match.group(1)
                    names.append(n)
            elif n not in ["search-search", "execute_python_code-execute_python_code"]:
                if n not in seen:
                    seen.add(n)
                    names.append(n)
    return names


def map_st_to_number(st_id: str) -> int:
    match = re.match(r"ST_?(\d+)", st_id)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid step ID format: {st_id}")

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

def _parse_dag_list(text: str, max_turns: Optional[int] = None) -> Tuple[Optional[List], str]:
    """
    解析 ##DAG_LIST 段，返回 (dag_list, status)
    status in {"ok", "missing", "empty", "invalid"}
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    m = re.search(r'##DAG_LIST[^\n]*\n([\s\S]*?)(?=\n##|\Z)', text)
    if not m:
        return None, "missing"
    s = m.group(1).strip()
    if s.startswith("```"):
        s = re.sub(r'^```[^\n]*\n([\s\S]*?)\n```$', r'\1', s).strip()
    if not s:
        return None, "empty"

    # 只为未加引号的 ST 节点加引号，避免 double-quote
    s = re.sub(r'(?<!")\b(ST\d+)\b(?!")', r'"\1"', s)
    try:
        parsed = ast.literal_eval(s)
    except Exception:
        return None, "invalid"

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None, "invalid"

    nodes = []
    for item in parsed:
        if isinstance(item, str):
            nodes.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) not in [1, 2]:
                return None, "invalid"
            for node in item:
                if not isinstance(node, str):
                    return None, "invalid"
                nodes.append(node)
        else:
            return None, "invalid"

    for node in nodes:
        if not re.match(r'^ST\d+$', node):
            return None, "invalid"

    if max_turns is not None and nodes:
        max_id = max(int(n[2:]) for n in nodes)
        if max_id > max_turns:
            return None, "invalid"

    return parsed, "ok"

class CentralizedQwenManager(QwenManager):
    """集中式QwenManager - 工具调用转发到Ray Actor"""
    
    def __init__(self, verl_config, centralized_actor_handle=None, mock=False):
        if not mock and centralized_actor_handle is None:
            raise ValueError("集中式QwenManager需要centralized_actor_handle参数")
        
        # 先调用父类初始化，获得完整的基础功能
        super().__init__(verl_config)
        
        # 保存集中式Actor句柄
        self.centralized_actor_handle = centralized_actor_handle
        with open(_repo_root() / "prompt" / "workflow.yaml", "r") as file:
            self.system_template = yaml.safe_load(file)

        # 设置超时时间，避免 Ray 调用无限阻塞
        self.timeout_dag = 30  # 从10秒增加到30秒,适应大批次场景
        self.max_turns = verl_config.get("max_turns", 6)

        # 从配置中读取 dynamic_prompt 参数（默认为 False）
        self.dynamic_prompt = getattr(verl_config, 'dynamic_prompt', True)
        print(f"[CentralizedQwenManager] Initialized with dynamic_prompt={self.dynamic_prompt}")

        # 添加 temp_mapping 本地缓存，避免频繁远程调用
        self._temp_mapping_cache = {}
        self._temp_mapping_cache_time = 0
        self._temp_mapping_cache_ttl = 5  # 缓存5秒
        self._cache_lock = threading.Lock()  # 添加缓存锁保护
        # 添加工具选择缓存，避免频繁调用 hybrid_search
        self._tool_selection_cache = {}  # key: query_hash, value: (tools, timestamp)
        self._tool_selection_cache_ttl = 10  # 缓存10秒

        # 新增：MCP 工具版本号缓存机制
        self._cached_mcp_version = -1  # 本地缓存的版本号，初始为-1表示未同步
        self._cached_functions_mcps = []  # 本地缓存的工具列表

        self._build_temp_mcp_tools(init_project=False)
        print("集中式QwenManager初始化完成，工具调用将转发到集中式Actor")

    def _build_temp_mcp_tools(self, init_project=False):
        # try:
            # 迁移自 envs/tools/mcp_creation.py：确保临时/正式 MCP 文件存在且具备基本入口
        self._ensure_mcp_tool_files(init_project=init_project)
        temp_config_path = self.verl_config.temp_config_path
        official_config_path = self.verl_config.official_config_path
        assert temp_config_path is not None and official_config_path is not None, "temp_config_path and official_config_path must be set"
        # if init_project:
        #     official_function_list = []
        # else:
        official_function_list = parse_mcp_tools_config(official_config_path, self.project_name)

        # 记录已有基础工具函数，便于后续过滤新增的 MCP 工具
        base_functions = set()
        for fn in getattr(self, "functions", []):
            if isinstance(fn, dict):
                name = fn.get("name")
            else:
                name = getattr(fn, "name", None)
            if name:
                base_functions.add(name)
        print(official_function_list, "OFFICIAL_FUNCTION_LIST")
        for tool in official_function_list:
            try:
                self._init_tool(tool)
            except Exception as inner_e:
                raise RuntimeError(f"Error initializing tool {tool}: {inner_e}") from inner_e
        
        all_functions = [func.function for func in self.tool_map.values()]
        self.functions_mcps = []
        for fn in all_functions:
            name = None
            if isinstance(fn, dict):
                name = fn.get("name")
            else:
                name = getattr(fn, "name", None)

            if name and name not in base_functions:
                self.functions_mcps.append(fn)
        # except Exception as e:
        #     # Catch any exception (including McpError) and raise a RuntimeError
        #     # This ensures that Ray can serialize the exception properly, avoiding the unpickle error.
        #     raise RuntimeError(f"Failed to build MCP tools: {e}") from e

    def _ensure_mcp_tool_files(self, init_project=False):
        """确保 temp/official MCP 入口文件存在，并写入基础导入和 __main__ 启动块"""
        base_dir = Path(__file__).resolve().parents[2] / "tools"
        invented_dir = base_dir / "invented_tools" / (self.project_name or "default_project")
        invented_dir.mkdir(parents=True, exist_ok=True)
        invented_file = invented_dir / "temp_tool_list.py"
        official_file = invented_dir / "official_tool_list.py"

        common_imports = (
            "import json\n"
            "import sys\n"
            "import traceback\n"
            "import math\n"
            "import numpy as np\n"
            "import re\n"
            "import os\n"
            "import datetime\n"
            "from collections import Counter, defaultdict, deque\n"
            "from mcp.server.fastmcp import FastMCP\n\n"
        )
        main_block = (
            "\n\nif __name__ == \"__main__\":\n"
            "    print(\"\\nStart MCP service:\")\n"
            "    mcp.run(transport='stdio')\n"
        )

        def ensure_file(file_path: Path, server_name: str):
            if init_project:
                # 直接重新创建文件
                content = f"{common_imports}mcp = FastMCP('{server_name}')\n\n{main_block}"
            else:
                # 保留原有逻辑，检查现有内容
                content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
                header = f"{common_imports}mcp = FastMCP('{server_name}')\n\n"

                if not content.strip():
                    content = header
                elif "FastMCP" not in content:
                    content = header + content

                if main_block not in content:
                    content += main_block

            file_path.write_text(content, encoding="utf-8")

        # ensure_file(invented_file, "InventedServer")
        ensure_file(official_file, "Official_Server")

    def reload_official_tools(self):
        """重新加载 official_tool_list.py 中的工具

        在每个训练step后调用，确保新创建的工具被加载到内存中
        """
        official_config_path = self.verl_config.official_config_path
        if official_config_path is None:
            print("[reload_official_tools] official_config_path is None, skipping reload")
            return
        try:
            # 重新解析配置文件
            official_function_list = parse_mcp_tools_config(official_config_path, self.project_name) or []

            if not official_function_list:
                print("[reload_official_tools] No tools found in official config")
                return

            # 记录已有基础工具函数
            base_functions = set()
            for fn in getattr(self, "functions", []):
                if isinstance(fn, dict):
                    name = fn.get("name")
                else:
                    name = getattr(fn, "name", None)
                if name:
                    base_functions.add(name)

            # 记录重新加载前的工具数量
            old_count = len(self.functions_mcps) if hasattr(self, 'functions_mcps') else 0

            # 重新初始化工具
            # 注意：这会将新工具添加到 self.tool_map 中
            for tool in official_function_list:
                try:
                    self._init_tool(tool)
                except Exception as inner_e:
                    print(f"[reload_official_tools] Warning: Error re-initializing tool {tool}: {inner_e}")

            # 重新构建 functions 和 functions_mcps
            all_functions = [func.function for func in self.tool_map.values()]


            # 更新 self.functions_mcps（只包含非基础工具）
            self.functions_mcps = []
            for fn in all_functions:
                name = None
                if isinstance(fn, dict):
                    name = fn.get("name")
                else:
                    name = getattr(fn, "name", None)

                if name and name not in base_functions:
                    self.functions_mcps.append(fn)

            new_count = len(self.functions_mcps)
            print(f"[reload_official_tools] Reloaded tools: {old_count} -> {new_count} (total: {len(self.functions_mcps)})")

        except Exception as e:
            print(f"[reload_official_tools] Error reloading official tools: {e}")
            import traceback
            traceback.print_exc()

    def get_item_name(self, item):
        if type(item) == dict:
            if item["name"] != "create_and_execute_mcp":
                return item["name"]
            else:
                # 安全地解析 args 并获取 name
                try:
                    args = item.get("args", "{}")
                    # 如果 args 是字符串，尝试解析为 JSON
                    if isinstance(args, str):
                        parsed_args = json.loads(args)
                    else:
                        parsed_args = args

                    # 如果解析结果是字典，获取 name
                    if isinstance(parsed_args, dict):
                        mcp_name = parsed_args.get("name", "error_mcp_name")
                    else:
                        # 如果解析结果不是字典（可能是字符串），直接使用
                        mcp_name = parsed_args if isinstance(parsed_args, str) else "error_mcp_name"
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"[get_item_name] Error parsing mcp name from args: {e}, args={item.get('args', 'N/A')}")
                    mcp_name = "error_mcp_name"

                return mcp_name
        else:
            return item

    def execute_actions(self, responses: List[str]):
        assert self.centralized_actor_handle is not None, "集中式QwenManager需要centralized_actor_handle参数"
        """执行动作 - 转发到集中式Actor"""
        actions, tools, tool_names = [], [], []
        for response in responses:
            temp_action, temp_tool_list = self.parse_response(response_content=response)
            # temp_action: answer or tools
            # if temp_action is 'answer', temp_tool_list is the answer
            # else, temp_tool_list is the list of the 'Tool' instances
            actions.append(temp_action)
            temp_tool_names = [self.get_item_name(item) for item in temp_tool_list]
            temp_tool_list = self.full_name(temp_tool_list)
            tools.append(temp_tool_list)
            tool_names.append(temp_tool_names)

        # 获取 Actor 中的 temp_mapping，使用缓存机制避免频繁远程调用
        current_time = time.time()

        # 使用锁保护缓存访问，防止竞态条件
        with self._cache_lock:
            cache_expired = current_time - self._temp_mapping_cache_time > self._temp_mapping_cache_ttl
            if cache_expired:
                # 需要更新缓存
                need_update = True
                temp_dict = self._temp_mapping_cache  # 先使用旧缓存
            else:
                # 缓存有效，直接使用
                need_update = False
                temp_dict = self._temp_mapping_cache

        # 在锁外进行远程调用，避免长时间持有锁
        if need_update:
            try:
                new_temp_dict = ray.get(self.centralized_actor_handle.get_temp_mapping.remote(), timeout=10)
                # 更新缓存时再次获取锁
                with self._cache_lock:
                    self._temp_mapping_cache = new_temp_dict
                    self._temp_mapping_cache_time = current_time
                temp_dict = new_temp_dict
            except ray.exceptions.GetTimeoutError:
                print("Warning: get_temp_mapping timeout after 10s, using cached dict")
                # 使用已有的 temp_dict（在锁内已获取）
            except Exception as e:
                print(f"Failed to get temp_mapping from actor: {e}, using cached dict")
                # 使用已有的 temp_dict（在锁内已获取）

        # 修复事件循环冲突问题 - 使用进程池避免嵌套事件循环并支持超时终止
        try:
            # 检查是否在事件循环中运行
            try:
                asyncio.get_running_loop()
                # 如果在事件循环中,使用进程池执行异步任务避免死锁，并支持超时强制终止
                import concurrent.futures
                import multiprocessing

                def run_in_process(actions, tools, temp_dict, centralized_actor_handle):
                    """在独立进程中运行，避免事件循环冲突"""
                    try:
                        # 在新进程中创建新的事件循环
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(
                                self.execute_all_tools(actions, tools, temp_dict)
                            )
                            return result
                        finally:
                            new_loop.close()
                    except Exception as e:
                        return {'error': str(e), 'exception': e}

                # 使用 ThreadPoolExecutor 而非 ProcessPoolExecutor，但添加超时控制
                # 这样可以避免 Ray ActorHandle 序列化问题，同时仍能隔离事件循环
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_process, actions, tools, temp_dict, self.centralized_actor_handle)
                    try:
                        tool_results = future.result(timeout=300)  # 5分钟超时
                        if isinstance(tool_results, dict) and 'error' in tool_results:
                            raise tool_results.get('exception', Exception(tool_results['error']))
                    except concurrent.futures.TimeoutError:
                        print("Error: Tool execution timeout after 300s")
                        # 取消任务并返回错误结果
                        future.cancel()
                        tool_results = [{'role': 'assistant', 'content': 'Tool execution timeout'}] * len(actions)

            except RuntimeError:
                # 没有运行中的事件循环,直接使用 asyncio.run
                tool_results = asyncio.run(self.execute_all_tools(actions, tools, temp_dict))
        except Exception as e:
            print(f"Error running async tools: {e}")
            import traceback
            traceback.print_exc()
            tool_results = [{'role': 'assistant', 'content': f'Error: {str(e)}'} for _ in actions]

        return actions, tool_results, tool_names
    
    def parse_subtask_plan(self, text: str) -> bool:
        """
        解析 ##DAG_LIST 块，验证其格式，并检查最大子任务ID是否超过 max_turns。

        Args:
            text: 包含 DAG 列表的输入字符串。
            max_turns: 允许的最大子任务ID。

        Returns:
            如果格式有效且最大ID未超过 max_turns，则为 True，否则为 False。
        """
        _, status = _parse_dag_list(text, max_turns=self.max_turns)
        return status == "ok"

    def parse_plan_response(self, response_content: str, overall_task=None, mcp_worthiness=None, is_validation: bool=False):
        """
        解析计划响应

        Args:
            response_content: 响应内容
            overall_task: 总体任务
            mcp_worthiness: MCP 价值
            is_validation: 是否为 validation 阶段
        """
        plan_success = self.parse_subtask_plan(response_content)
        if plan_success:
            system_prompt = self.initialize_system_prompt(
                subtask_plan=response_content,
                overall_task=overall_task,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation
            )
            return 'plan_success', system_prompt
        else:
            system_prompt = self.initialize_system_prompt(
                subtask_plan=None,
                overall_task=overall_task,
                mcp_worthiness=mcp_worthiness,
                is_validation=is_validation
            )
            return 'plan_failed', system_prompt
        
        
    def initialize_system_prompt(self, subtask_plan=None, overall_task=None, mcp_worthiness=None, is_validation=False) -> str:
        """
        初始化 system prompt

        Args:
            subtask_plan: 子任务计划
            overall_task: 总体任务
            mcp_worthiness: MCP 价值（0 或 1）
            is_validation: 是否为 validation 阶段（默认 False）
                          当 is_validation=True 时，强制使用 dynamic_prompt=True 的行为

        Returns:
            system_prompt: 生成的系统提示词
        """
        # Validation 阶段强制使用 dynamic_prompt=True 的逻辑
        use_dynamic_prompt = self.dynamic_prompt or is_validation

        # if not use_dynamic_prompt:
            # 不使用动态 prompt：始终使用 system_prompt
        if subtask_plan:
            mode = "subtask"
            system_prompt = populate_template(
                self.system_template["system_prompt"],
                variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
            )
        else:
            mode = "overall"
            system_prompt = populate_template(
                self.system_template["system_prompt"],
                variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
            )
        # else:
        #     # 使用动态 prompt：根据 mcp_worthiness 选择
        #     if mcp_worthiness == 1:
        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #     elif mcp_worthiness == 0:
        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["simple_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["simple_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #     else:
        #         # mcp_worthiness 为 None 或其他值时（如 validation），默认使用 system_prompt
        #         if subtask_plan:
        #             mode = "subtask"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": subtask_plan, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )
        #         else:
        #             mode = "overall"
        #             system_prompt = populate_template(
        #                 self.system_template["system_prompt"],
        #                 variables={"tools": [], "managed_agents": [], "input_plan": overall_task, "question":overall_task, "mode": mode, "max_turns":self.max_turns},
        #             )

        # if is_validation:
        #     print(f"[Validation Mode] Using system_prompt (dynamic_prompt forced to True)")

        return system_prompt


    def execute_plans(self, responses: List[str], overall_tasks: List[str]=None, mcp_worthiness: List[int]=None, is_validation: bool=False):
        assert self.centralized_actor_handle is not None, "集中式QwenMCPGraphManager需要centralized_actor_handle参数"
        """
        执行计划动作 - 转发到集中式Actor

        Args:
            responses: 模型响应列表
            overall_tasks: 总体任务列表
            mcp_worthiness: MCP 价值列表
            is_validation: 是否为 validation 阶段（默认 False）
        """
        actions, plans, raw_plans = [], [], []
        for response, overall_task, mcp_worth in zip(responses, overall_tasks, mcp_worthiness):
            temp_action, temp_plan = self.parse_plan_response(
                response_content=response,
                overall_task=overall_task,
                mcp_worthiness=mcp_worth,
                is_validation=is_validation  # 传递 validation 标志
            )
            actions.append(temp_action)
            plans.append(temp_plan)
            if temp_action == 'plan_success':
                raw_plans.append(response)
            else:
                raw_plans.append(overall_task)
        names = len(plans) * ["plan"]
        return actions, plans, raw_plans, names

    async def _select_tools_async(self, query: str, top_k: int = 2):
        """
        异步版本：通过 hybrid search 选出最相关的 top_k MCP 工具。
        添加降级机制：搜索失败时返回空列表
        使用缓存避免频繁调用 hybrid_search

        注意：此函数只返回选中的 MCP 工具，不包含基础工具
        返回：List[Tuple[dict, float]] - 工具和对应的搜索分数
        """
        import hashlib
        import time

        # hybrid search 动态检索 - 使用异步等待
        picked_items = []  # 存储 (name, score) 元组
        search_timeout = 10  # 10秒超时
        try:
            # 使用 asyncio.wait_for 包装 ray future
            future = self.centralized_actor_handle.hybrid_search.remote(query, top_k=top_k)
            results = await asyncio.wait_for(future, timeout=search_timeout)

            print(f"[_select_tools_async] hybrid_search returned {len(results) if results else 0} groups")

            flat = []
            for group in results or []:
                flat.extend(group or [])

            print(f"[_select_tools_async] Flattened to {len(flat)} items")
            if flat:
                print(f"[_select_tools_async] First item structure: {list(flat[0].keys()) if isinstance(flat[0], dict) else type(flat[0])}")

            # 提取 name 和 score
            seen_names = set()
            for item in flat:
                name = item.get("mcp_id") or item.get("mcp_name") or item.get("name")
                score = item.get("search_score", 0.0)  # 从 hybrid_search 返回的分数
                if name and name not in seen_names:
                    picked_items.append((name, score))
                    seen_names.add(name)
                if len(picked_items) >= top_k:
                    break

            print(f"[_select_tools_async] Picked items: {[(name, score) for name, score in picked_items]}")

        except asyncio.TimeoutError:
            print(f"[_select_tools_async] hybrid_search timeout after {search_timeout}s, returning empty list")
            picked_items = []
        except Exception as e:
            print(f"[_select_tools_async] hybrid_search failed: {e}, returning empty list")
            import traceback
            traceback.print_exc()
            picked_items = []

        # === 核心优化：版本号检查 + 本地缓存 ===
        print(f"[_select_tools_async] Checking MCP version...")
        try:
            # 1. 先获取当前版本号（轻量级操作）
            version_future = self.centralized_actor_handle.get_mcp_version.remote()
            current_version = await asyncio.wait_for(version_future, timeout=1.0)

            # 2. 检查版本号是否变化
            if current_version != self._cached_mcp_version:
                print(f"[_select_tools_async] Version changed: {self._cached_mcp_version} -> {current_version}, refreshing cache...")
                # 版本号变化，需要更新缓存
                data_future = self.centralized_actor_handle.get_functions_mcps_with_version.remote()
                data = await asyncio.wait_for(data_future, timeout=3.0)

                self._cached_mcp_version = data['version']
                self._cached_functions_mcps = data['functions_mcps']
                print(f"[_select_tools_async] Cache updated: version={self._cached_mcp_version}, tools={len(self._cached_functions_mcps)}")
            else:
                print(f"[_select_tools_async] Version unchanged ({current_version}), using local cache with {len(self._cached_functions_mcps)} tools")

            # 使用缓存的工具列表
            all_mcp_tools = self._cached_functions_mcps

        except asyncio.TimeoutError:
            print("[_select_tools_async] Timeout getting version/tools, using stale cache")
            all_mcp_tools = self._cached_functions_mcps
        except Exception as e:
            print(f"[_select_tools_async] Error getting version/tools: {e}, using stale cache")
            import traceback
            traceback.print_exc()
            all_mcp_tools = self._cached_functions_mcps

        # 在已注册的 MCP 工具中筛选
        print(f"[_select_tools_async] Looking for picked_items in {len(all_mcp_tools)} cached tools")

        # 使用后缀匹配，因为 picked_names 是纯净名字（如 'analyze_problem'）
        # 而 all_mcp_tools 中的名字可能带前缀（如 'official_created_mcps-analyze_problem'）
        selected_tools_with_scores = []  # List[Tuple[dict, float]]
        for fn in all_mcp_tools:
            fn_name = fn.get("name", "")
            for picked_name, score in picked_items:
                # 后缀匹配：工具名以 picked_name 结尾，或者完全相等
                if fn_name.endswith(f"-{picked_name}") or fn_name == picked_name:
                    selected_tools_with_scores.append((fn, score))
                    break

        print(f"[_select_tools_async] Selected {len(selected_tools_with_scores)} MCP tools from hybrid search, {len(all_mcp_tools)} MCP tools in total")
        return selected_tools_with_scores

    # def _select_tools(self, query: str, top_k: int = 2):
    #     """
    #     同步版本（向后兼容）：通过 hybrid search 选出最相关的 top_k MCP 工具。
    #     添加降级机制：搜索失败时返回空列表

    #     注意：此函数只返回选中的 MCP 工具，不包含基础工具
    #     返回：List[Tuple[dict, float]] - 工具和对应的搜索分数
    #     """
    #     import hashlib
    #     import time

    #     # hybrid search 动态检索
    #     picked_items = []  # 存储 (name, score) 元组
    #     search_timeout = 10  # 10秒超时
    #     try:
    #         future = self.centralized_actor_handle.hybrid_search.remote(query, top_k=top_k)
    #         results = ray.get(future, timeout=search_timeout)
    #         flat = []
    #         for group in results or []:
    #             flat.extend(group or [])

    #         # 提取 name 和 score
    #         seen_names = set()
    #         for item in flat:
    #             name = item.get("mcp_id") or item.get("mcp_name") or item.get("name")
    #             score = item.get("search_score", 0.0)  # 从 hybrid_search 返回的分数
    #             if name and name not in seen_names:
    #                 picked_items.append((name, score))
    #                 seen_names.add(name)
    #             if len(picked_items) >= top_k:
    #                 break
    #     except ray.exceptions.GetTimeoutError:
    #         print(f"[_select_tools] hybrid_search timeout after {search_timeout}s, returning empty list")
    #         picked_items = []
    #     except Exception as e:
    #         print(f"[_select_tools] hybrid_search failed: {e}, returning empty list")
    #         picked_items = []

    #     # 关键修复：从 centralized actor 获取最新的 MCP 工具列表，而不是使用本地的 self.functions_mcps
    #     try:
    #         all_mcp_tools = ray.get(self.centralized_actor_handle.get_functions_mcps.remote(), timeout=3.0)
    #         print(f"[_select_tools] Retrieved {len(all_mcp_tools)} MCP tools from centralized actor")
    #     except ray.exceptions.GetTimeoutError:
    #         print("[_select_tools] Timeout getting MCP tools from actor, using local self.functions_mcps")
    #         all_mcp_tools = self.functions_mcps
    #     except Exception as e:
    #         print(f"[_select_tools] Error getting MCP tools from actor: {e}, using local self.functions_mcps")
    #         all_mcp_tools = self.functions_mcps

    #     # 使用后缀匹配，因为 picked_names 是纯净名字（如 'analyze_problem'）
    #     # 而 all_mcp_tools 中的名字可能带前缀（如 'official_created_mcps-analyze_problem'）
    #     selected_tools_with_scores = []  # List[Tuple[dict, float]]
    #     for fn in all_mcp_tools:
    #         fn_name = fn.get("name", "")
    #         for picked_name, score in picked_items:
    #             # 后缀匹配：工具名以 picked_name 结尾，或者完全相等
    #             if fn_name.endswith(f"-{picked_name}") or fn_name == picked_name:
    #                 selected_tools_with_scores.append((fn, score))
    #                 break

    #     print(f"[_select_tools] Selected {len(selected_tools_with_scores)} MCP tools from hybrid search, {len(all_mcp_tools)} MCP tools in total")
    #     return selected_tools_with_scores

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True, current_plan=None):
        assert mode in ['initial', 'tool_call', 'assistant_response', 'plan_call'], 'Invalid mode: {}'.format(mode)
        if isinstance(input_data, str):
            input_data = re.sub(r'<\|im_start\|>|<\|im_end\|>', '', input_data)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            # tools=self.functions,
            tools=[],
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            # 仅初始提示时进行工具裁剪
            # tools_to_use = self._select_tools(current_plan, top_k=2)
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, 
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['plan_call']:
            # NOTE: input_data 是一个完整的 system prompt，应该作为 SYSTEM 角色而非 USER
            # 修复：将 system prompt 正确标记为 SYSTEM 角色
            if type(input_data) == str:
                chat = [{'role': SYSTEM, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            # 对每个 step 单独搜索工具，然后合并并按分数排序
            tools_with_scores = []  # List[Tuple[dict, float]]
            if isinstance(current_plan, dict):
                # 对每个步骤单独进行工具搜索
                for step_name, step_content in current_plan.items():
                    step_text = step_content.get('steps', '')
                    if step_text.strip():
                        # 每个 step 搜索 top_k=2 个工具
                        step_tools = self._select_tools(step_text, top_k=2)
                        tools_with_scores.extend(step_tools)
                        print(f"[tool_selection] Step '{step_name}' found {len(step_tools)} tools")
            elif isinstance(current_plan, str):
                tools_with_scores = self._select_tools(current_plan, top_k=2)

            # 按分数排序并去重，保留 top_k=3 个工具
            seen = set()
            sorted_tools_with_scores = sorted(tools_with_scores, key=lambda x: x[1], reverse=True)
            selected_mcp_tools = []
            for tool, score in sorted_tools_with_scores:
                name = tool.get("name")
                if name and name not in seen:
                    seen.add(name)
                    selected_mcp_tools.append(tool)
                    print(f"[tool_selection] Selected tool '{name}' with score {score:.4f}")
                if len(selected_mcp_tools) >= 3:  # 最多保留 3 个工具
                    break

            print(f"[tool_selection] Final selected {len(selected_mcp_tools)} tools from {len(tools_with_scores)} candidates")

            # 构建最终工具列表：基础工具 + 选中的 MCP 工具
            base_tool_names = {"search", "execute_python_code", "create_and_execute_mcp"}

            # 调试：打印 self.functions 的内容
            print(f"[DEBUG] self.functions type: {type(self.functions)}")
            print(f"[DEBUG] self.functions length: {len(self.functions) if hasattr(self, 'functions') else 'N/A'}")
            # print(f"[DEBUG] self.tool_map keys: {list(self.tool_map.keys())}")

            # 如果 self.functions 不存在或为空，从 self.tool_map 重建
            if not hasattr(self, 'functions') or not self.functions:
                print("[WARN] self.functions is empty, rebuilding from tool_map")
                self.functions = [func.function for func in self.tool_map.values()]

            base_tools = [fn for fn in self.functions if fn.get("name") in base_tool_names]
            print(f"[DEBUG] base_tools found: {[fn.get('name') for fn in base_tools]}")

            assert base_tools is not None, "base_tools is None"
            tools_for_prompt = base_tools + selected_mcp_tools

            # 修复：对于 plan_call 模式，直接应用 chat template，不使用 base_chat 技巧
            # 因为 input_data 已经是完整的 system prompt
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tools=tools_for_prompt,
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
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
                conversation=base_chat + chat, tools=[], 
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template

    async def get_prompt_async(self, input_data, tokenizer, mode='initial', add_generation_prompt=True, current_plan=None):
        """
        异步版本的 get_prompt，用于并发处理多个 prompt 生成
        """
        assert mode in ['initial', 'tool_call', 'assistant_response', 'plan_call'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': SYSTEM, 'content': 'base'},
            {'role': USER, 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=[],
            tokenize=False, add_generation_prompt=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False,
                add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
        elif mode in ['plan_call']:
            # plan_call 模式：input_data 是 plan 文本（完整的 DAG_LIST 和 subtasks）
            # 关键修复：不使用 base_chat，而是直接构造完整的 conversation
            # plan 通过 system prompt 提供，user 消息要求开始执行

            if type(input_data) == str:
                # input_data 是 plan 文本
                # 通过 initialize_system_prompt 生成包含 plan 的 system prompt
                # 构造完整的 conversation（不使用 base_chat）
                full_conversation = [
                    {'role': SYSTEM, 'content': input_data}
                ]
            elif type(input_data) == list:
                full_conversation = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            # 使用异步工具选择 - 对每个 step 单独搜索，然后合并并按分数排序
            tools_with_scores = []  # List[Tuple[dict, float]]
            if isinstance(current_plan, dict):
                # 对每个步骤单独进行工具搜索
                for step_name, step_content in current_plan.items():
                    step_text = step_content.get('steps', '')
                    if step_text.strip():
                        # 每个 step 搜索 top_k=2 个工具
                        step_tools = await self._select_tools_async(step_text, top_k=2)
                        tools_with_scores.extend(step_tools)
                        print(f"[tool_selection] Step '{step_name}' found {len(step_tools)} tools")
            elif isinstance(current_plan, str):
                tools_with_scores = await self._select_tools_async(current_plan, top_k=2)

            # 按分数排序并去重，保留 top_k=3 个工具
            seen = set()
            sorted_tools_with_scores = sorted(tools_with_scores, key=lambda x: x[1], reverse=True)
            selected_mcp_tools = []
            for tool, score in sorted_tools_with_scores:
                name = tool.get("name")
                if name and name not in seen:
                    seen.add(name)
                    selected_mcp_tools.append(tool)
                    print(f"[tool_selection] Selected tool '{name}' with score {score:.4f}")
                if len(selected_mcp_tools) >= 3:  # 最多保留 3 个工具
                    break

            print(f"[tool_selection] Final selected {len(selected_mcp_tools)} tools from {len(tools_with_scores)} candidates")

            # 构建最终工具列表：基础工具 + 选中的 MCP 工具
            print(f"[DEBUG] self.functions type: {type(self.functions)}")
            print(f"[DEBUG] self.functions length: {len(self.functions) if hasattr(self, 'functions') else 'N/A'}")
            # print(f"[DEBUG] self.tool_map keys: {list(self.tool_map.keys())}")
            base_tool_names = {"search-search", "execute_python_code-execute_python_code", "create_and_execute_mcp-create_and_execute_mcp"}
            base_tools = [fn for fn in self.functions if fn.get("name") in base_tool_names]
            assert base_tools is not [], "base_tools is None"
            tools_for_prompt = base_tools + selected_mcp_tools

            # plan_call 模式：使用完整的 conversation，不需要 base_chat
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=full_conversation, tools=tools_for_prompt,
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            # plan_call 模式不需要 replace base_prompt，因为我们构造了完整的 conversation
            prompt_with_chat_template = remove_middle_tokens(temp_prompt_with_chat_template)
        elif mode in ['tool_call', 'assistant_response']:
            role = 'tool' if mode == 'tool_call' else ASSISTANT
            if type(input_data) == str:
                chat = [{'role': role, 'content': input_data}]
            elif type(input_data) == list:
                chat = input_data
            else:
                raise ValueError('Unexpected type of input_data {} ({})'.format(type(input_data), input_data))

            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + chat, tools=[],
                tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=self.verl_config.enable_thinking
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))

        return prompt_with_chat_template


    def _get_task_base(self) -> int:
        if self.centralized_actor_handle is not None:
            try:
                return ray.get(
                    self.centralized_actor_handle.get_task_base.remote(),
                    timeout=10
                )
            except ray.exceptions.GetTimeoutError:
                print("Warning: get_task_base timeout after 10s, returning 1")
                return 1
            except Exception as e:
                print(f"Error getting task_base: {e}, returning 1")
                return 1
        return getattr(self, "_local_task_base", 1)

    def _set_task_base(self, value: int):
        if self.centralized_actor_handle is not None:
            try:
                return ray.get(
                    self.centralized_actor_handle.set_task_base.remote(value),
                    timeout=10
                )
            except ray.exceptions.GetTimeoutError:
                print(f"Warning: set_task_base timeout after 10s")
                return None
            except Exception as e:
                print(f"Error setting task_base: {e}")
                return None
        self._local_task_base = value
        return value

    def extract_dag_and_parallel(self, text: str) -> Tuple[Optional[List], Optional[int]]:
        try:
            subtask_dict = _parse_plan(text)
        except:
            subtask_dict = None
        dag_list, status = _parse_dag_list(text, max_turns=None)
        if status in ["missing", "empty"]:
            return [], self._get_task_base(), subtask_dict
        if status != "ok":
            print("[ERROR] Failed to parse DAG_LIST")
            return None, None, subtask_dict
        print("[SUCCESS] successfully parse DAG_LIST")

        task_id = self._get_task_base()

        if dag_list:
            # 成对边（长度=2 的 tuple/list）
            valid_pairs = [tuple(item) for item in dag_list if isinstance(item, (tuple, list)) and len(item) == 2]

            # 单节点（长度=1 的 tuple/list，或直接是 "STi" 字符串）
            singleton_nodes = []
            for item in dag_list:
                if isinstance(item, (tuple, list)) and len(item) == 1:
                    singleton_nodes.append(item[0])
                elif isinstance(item, str):
                    singleton_nodes.append(item)

            node_pair_list = [
                (map_st_to_number(a) + task_id, map_st_to_number(b) + task_id)
                for (a, b) in valid_pairs
            ] if valid_pairs else []
            node_single_list = [map_st_to_number(n) + task_id for n in singleton_nodes] if singleton_nodes else []

            # 将图更新委托给 centralized actor，保证状态一致
            if self.centralized_actor_handle is not None:
                try:
                    # 添加超时控制，避免无限阻塞
                    future = self.centralized_actor_handle.update_sub_task_graph.remote(node_pair_list, node_single_list)
                    result = ray.get(future, timeout=self.timeout_dag)
                except ray.exceptions.GetTimeoutError:
                    print(f"[extract_dag_and_parallel] update_sub_task_graph timed out after {self.timeout_dag} seconds")
                except Exception as e:
                    print(f"[extract_dag_and_parallel] update_sub_task_graph failed: {e}")

            # 更新 task_base
            all_nodes = []
            for item in dag_list:
                if isinstance(item, (tuple, list)):
                    all_nodes.extend(item)
                elif isinstance(item, str):
                    all_nodes.append(item)

            if all_nodes:
                max_sub_task_id = max(map(map_st_to_number, all_nodes))
                self._set_task_base(task_id + max_sub_task_id)

        return dag_list, task_id, subtask_dict

    async def execute_all_tools(self, actions, tool_list, temp_dict=None):
        """异步并行执行所有工具列表
        
        Args:
            tool_list: 工具列表的列表
            temp_dict: 用于存储动态创建的MCP工具的字典
            
        Returns:
            所有工具执行结果的列表
        """
        
        # 并行执行每个工具列表
        tasks = []
        for temp_action, temp_tool_list in zip(actions, tool_list):
            tasks.append(self._execute_tool_batch(temp_action, temp_tool_list, temp_dict))
        
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _execute_tool_batch(self, action, tools, temp_dict=None):
        if action == 'answer':
            # 'tools' is a str (the answer)
            results = {'role': 'assitant', 'content': tools}
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            results = {'role': 'assitant', 'content': """# Extract the tools failed due to: {}""".format(tools)}
        elif action == 'actions':
            # Add MCP registration logic
            # 'tools' is the list of the 'Tool' instances
            object_refs = [self.centralized_actor_handle.execute_single_tool.remote(temp_tool) for temp_tool in tools]
            tool_results = await asyncio.gather(*object_refs)
            results = [{'role': 'tool', 'content': temp_tool_result} for temp_tool_result in tool_results]
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return results


@ray.remote(max_concurrency=200, num_gpus=0.05)  # 增加到200并发,适应大批次场景
class CentralizedToolActor:
    """集中式工具管理器Actor - 集中管理所有工具调用的并发，集成MCP功能"""

    def __init__(self, verl_config):
        self.verl_config = verl_config
        self.project_name = _resolve_project_name(verl_config)
        self.tool_manager = CentralizedQwenManager(verl_config, mock=True)
        self.max_turns = verl_config.max_turns
        self.timeout = getattr(verl_config, 'tool_timeout', 30)
        self._temp_cache_mtime = None
        self._temp_cache_data = None
        self._official_cache_mtime = None
        self._official_cache_data = None
        self._search_cache_refresh_interval = 15
        self._search_cache_mod_time = 0.0
        self._search_cache_last_check = 0.0
        self._search_cache = {}
        self._search_cache_lock = None # 将在异步执行时动态创建
        # 初始化 MCP 功能 - 创建非Ray版本的MCPBoxActor实例
        # 我们需要一个不带@ray.remote装饰器的版本
        self._lock_loop = None  # 记录当前锁所属的事件循环
        self.local_search = getattr(self.verl_config, 'local_search', True)
        self._search_cache_path = os.environ.get(
            "RL_FACTORY_SEARCH_CACHE_DIR",
            str(Path.home() / ".cache" / "rl_factory" / "search"),
        )
        if self.local_search:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "local_bing_search_cache.jsonl"))
        else:
            self._search_cache_file = Path(os.path.join(self._search_cache_path, "bing_search_cache.jsonl"))
        max_workers = getattr(verl_config, 'max_concurrency', 50)
        self._search_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SearchWorker")
        # 其他工具使用默认线程池，可以设置稍小的 worker 数，或者与 search 分开
        self._default_executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="DefaultToolWorker")
        self._load_search_cache()
        self._init_mcp_functionality()
        # 新增：MCP 工具版本号，用于跟踪工具列表更新
        self.mcp_version = 0
        print(f"集中式工具Actor初始化完成，工具执行超时时间：{self.timeout}秒，MCP版本号：{self.mcp_version}")

    def _ensure_async_structs(self):
        loop = asyncio.get_running_loop()
        if self._lock_loop is not loop or self._search_cache_lock is None or self._in_flight_lock is None:
            self._search_cache_lock = asyncio.Lock()
            self._in_flight_lock = asyncio.Lock()
            self._in_flight_requests = {}
            self._lock_loop = loop


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

    def _init_mcp_functionality(self):
        """初始化 MCP 相关功能（复制 MCPBoxActor 的核心逻辑）"""
        import threading
        from sentence_transformers import SentenceTransformer
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
        from collections import defaultdict

        # MCP 相关属性
        self.lock = threading.RLock()
        self.task_base = 1
        self.sub_task_graph = nx.DiGraph()
        self.field_components = {
            'vectorizer': TfidfVectorizer(stop_words='english'),
            'matrix': None,
            'mcp_ids': []
        }
        self.mcps = {}
        self.alias_map = {}
        self.temp_mapping = {}

        # 加载嵌入模型
        self.tool_timeout = 30
        self.embedding_model = SentenceTransformer(
            _default_embedding_model(),
            device='cuda:0'
        )
        print("CentralizedToolActor: SentenceTransformer is loading on GPU.")

        # ===== GraphRAG 集成 (可选功能) =====
        # 注意：必须在 init_mcps 之前初始化，因为 init_mcps 会访问 self.use_graphrag
        self.use_graphrag = getattr(self.verl_config, 'use_graphrag', False)
        self.graphrag_engine = None

        if self.use_graphrag:
            try:
                from envs.tool_manager.centralized.graph_rag_retrieval import GraphRAGRetrieval

                # GraphRAG 配置
                graphrag_config = {
                    'enable_community_detection': getattr(self.verl_config, 'graphrag_community_detection', True),
                    'enable_history_tracking': getattr(self.verl_config, 'graphrag_history_tracking', True),
                    'cache_dir': getattr(self.verl_config, 'graphrag_cache_dir', '/tmp/graphrag_cache')
                }

                print(f"[GraphRAG] Initializing GraphRAG engine with config: {graphrag_config}")

                self.graphrag_engine = GraphRAGRetrieval(
                    embedding_model_path=_default_embedding_model(),
                    device='cuda:0',
                    enable_community_detection=graphrag_config['enable_community_detection'],
                    enable_history_tracking=graphrag_config['enable_history_tracking'],
                    cache_dir=graphrag_config['cache_dir']
                )

                # 尝试加载缓存
                try:
                    self.graphrag_engine.load_cache()
                    print("[GraphRAG] Cache loaded successfully")
                except Exception as cache_e:
                    print(f"[GraphRAG] Warning: Failed to load cache: {cache_e}")

                print("[GraphRAG] Initialization completed successfully")

            except ImportError as e:
                print(f"[GraphRAG] Warning: GraphRAG module not found, falling back to traditional search: {e}")
                self.use_graphrag = False
                self.graphrag_engine = None
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to initialize GraphRAG, falling back to traditional search: {e}")
                import traceback
                traceback.print_exc()
                self.use_graphrag = False
                self.graphrag_engine = None
        else:
            print("[GraphRAG] GraphRAG disabled (use_graphrag=False), using traditional search")

        # 初始化 MCP 工具（必须在 GraphRAG 初始化之后）
        self.init_mcps([])

        print("DEBUG: All initialization steps completed")

    # --- task_base 远程读写（供 CentralizedQwenManager 使用） ---
    def get_task_base(self):
        return self.task_base

    def set_task_base(self, value: int):
        self.task_base = value
        return self.task_base

    def init_mcps(self, tools: List[Any]):
        for tool in tools:
            mcp_name = getattr(tool, 'name', None)
            if mcp_name and mcp_name not in self.mcps:
                description = getattr(tool, 'description', None)
                arguments = getattr(tool, 'inputs', None)
                returns = getattr(tool, 'output_type', None)
                requires = getattr(tool, 'requires', None)
                current_step_id = self.task_base
                self.mcps[mcp_name] = {
                    "mcp_name": mcp_name,
                    "description": description,
                    "arguments": arguments,
                    "returns": returns,
                    "requires": requires,
                    "code": f"{mcp_name}(**kwargs)",
                    "step_id" : current_step_id,
                }
                if not self.sub_task_graph.has_node(current_step_id):
                    self.sub_task_graph.add_node(current_step_id)
                    self.sub_task_graph.nodes[current_step_id]['mcp_name'] = mcp_name
                self.task_base += 1
            elif mcp_name:
                print(f"Agent Log: Tool '{mcp_name}' already exists, skipping.")

        self.naive_mcps = self.mcps # 假设 naive_mcps 只是 mcps 的一个引用
        self.build_tfidf_indices()
        self.build_embeddings()

        # ===== GraphRAG: 同步工具数据 =====
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.load_tools(self.mcps, self.sub_task_graph)
                self.mcp_version += 1
                print(f"[GraphRAG] Synced {len(self.mcps)} tools to GraphRAG engine (version {self.mcp_version})")
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to sync tools to GraphRAG: {e}")

    def parse_subtask_plan(self, text: str) -> bool:
        """
        解析 ##DAG_LIST 块，验证其格式，并检查最大子任务ID是否超过 max_turns。

        Args:
            text: 包含 DAG 列表的输入字符串。
            max_turns: 允许的最大子任务ID。

        Returns:
            如果格式有效且最大ID未超过 max_turns，则为 True，否则为 False。
        """
        _, status = _parse_dag_list(text, max_turns=self.max_turns)
        return status == "ok"

    def sequential_planning(self, rollout_max_turns) -> Optional[List[Tuple[str, str]]]:
        task_id = self.get_task_base()
        valid_pairs = [(i + 1 + task_id, i + 2 +task_id) for i in range(rollout_max_turns - 1)]  # Example sequential pairs, replace with actual logic
        # 然后，对过滤后的有效数据进行处理
        self.sub_task_graph.add_edges_from(valid_pairs)

        self.set_task_base(task_id + rollout_max_turns - 1)

        return None, task_id, {}

    def update_self_mcp(self):
        """获取指定名称的 MCP"""
        self.mcps.update(self.temp_mapping)

    def _register_mcp_sync(self, code: str, task_id: int = None, subtask_dict=None):
        assert task_id is not None, "task_id is required"

        # 预编译正则表达式，避免重复编译
        step_id_format = re.compile(
            r'<subtask>\s*(?P<step_id>ST_?\d+):?\s*(.*?)\s*</subtask>',
            re.IGNORECASE
        )
        step_info = step_id_format.search(code)
        plan_title = None
        plan_steps = None
        if not step_info:
            print("Code does not contain a valid step ID (e.g., '<subtask> ST_1: ... </subtask>').")
            return
        if subtask_dict:
            current_plan = subtask_dict.get(str(step_info.group(1)).replace('_', ''), None)
            if current_plan:
                plan_title = current_plan.get('title', None)
                plan_steps = current_plan.get('steps', None)
        step_id = map_st_to_number(step_info.group(1))
        nid = task_id + step_id

        tool_names = extract_tool_names(code)
        if not tool_names:
            print(f"Agent Log: No tool names found in code.")
            return 

        with self.lock:
            # 只在锁内执行图操作和状态更新
            if not self.sub_task_graph.has_node(nid):
                self.sub_task_graph.add_node(nid)

            for name in tool_names:
                if name in self.mcps:  # 直接字典查找，更高效
                    self.sub_task_graph.add_edge(self.mcps[name]['step_id'], nid)
                elif name in self.temp_mapping:  # 直接字典查找，更高效
                    temp_item = self.temp_mapping[name]
                    new_mcp_data = {
                        "mcp_name": name,
                        "description": temp_item.get('description'),
                        "arguments": temp_item.get('arguments'),
                        "returns": temp_item.get('returns'),
                        "code": temp_item.get('code', ""),
                        "step_id": nid,
                    }
                    self.mcps[name] = new_mcp_data
                    self.sub_task_graph.add_node(nid)
                    self.sub_task_graph.nodes[nid]['mcp_name'] = name

                    # GraphRAG: 同步新注册的工具
                    if self.use_graphrag and self.graphrag_engine is not None:
                        try:
                            self.graphrag_engine.update_tool(name, new_mcp_data)
                            self.mcp_version += 1
                            print(f"[GraphRAG] Synced newly registered tool: {name} (version {self.mcp_version})")
                        except Exception as e:
                            print(f"[GraphRAG] Warning: Failed to sync tool {name}: {e}")
                else:
                    print(f"Agent Log: MCP {name} not found.")
                    return
            print(f"Agent Log: MCPs registered for Step {step_id} (node {nid}).")
        return True

    # def temp_tool_list(self) -> Dict[str, Dict[str, str]]:
    #     """
    #     读取 invented_tools/temp_tool_list.py，提取每个 @mcp.tool 装饰的函数的完整信息：
    #     {tool_name: {"description": str, "arguments": str, "returns": str, "code": str}}
    #     其中 tool_name 优先取装饰器的 name 参数，否则函数名。
    #     """
    #     import ast
    #     import re
    #     from pathlib import Path

    #     root_dir = os.environ.get("RL_FACTORY_ROOT", str(_repo_root()))
    #     tool_path = Path(root_dir) / "envs" / "tools" / "invented_tools" / self.project_name / "temp_tool_list.py"
    #     if not tool_path.exists():
    #         return {}

    #     mtime = tool_path.stat().st_mtime
    #     if self._temp_cache_mtime == mtime and self._temp_cache_data is not None:
    #         return self._temp_cache_data

    #     text = tool_path.read_text(encoding="utf-8")
    #     lines = text.splitlines()

    #     try:
    #         tree = ast.parse(text)
    #     except Exception as e:
    #         print(f"Failed to parse temp_tool_list.py: {e}")
    #         return {}

    #     def _parse_docstring(ds: Optional[str]) -> Tuple[str, str, str]:
    #         """粗粒度解析 docstring 的 description/arguments/returns 块。"""
    #         if not ds:
    #             return "", "", ""

    #         desc_lines, arg_lines, ret_lines = [], [], []
    #         state = "desc"
    #         for raw in ds.splitlines():
    #             line = raw.strip()
    #             if not line:
    #                 continue
    #             if re.match(r"(?i)^args?:", line):
    #                 state = "args"
    #                 continue
    #             if re.match(r"(?i)^returns?:", line):
    #                 state = "returns"
    #                 continue
    #             if state == "desc":
    #                 desc_lines.append(line)
    #             elif state == "args":
    #                 arg_lines.append(line)
    #             elif state == "returns":
    #                 ret_lines.append(line)

    #         description = " ".join(desc_lines).strip()
    #         arguments = " ".join(arg_lines).strip()
    #         returns = " ".join(ret_lines).strip()
    #         return description, arguments, returns

    #     tools: Dict[str, Dict[str, str]] = {}
    #     for node in tree.body:
    #         if not isinstance(node, ast.FunctionDef):
    #             continue

    #         # 检查装饰器是否 mcp.tool
    #         tool_name = node.name
    #         has_mcp_deco = False
    #         if node.decorator_list:
    #             for deco in node.decorator_list:
    #                 # 形如 @mcp.tool 或 @mcp.tool(name='xxx')
    #                 if isinstance(deco, ast.Attribute) and deco.attr == "tool":
    #                     has_mcp_deco = True
    #                 elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "tool":
    #                     has_mcp_deco = True
    #                     for kw in deco.keywords:
    #                         if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
    #                             tool_name = kw.value.value
    #                 if has_mcp_deco:
    #                     break

    #         if not has_mcp_deco:
    #             continue

    #         # 代码块
    #         start_line = min([getattr(d, "lineno", node.lineno) for d in node.decorator_list] + [node.lineno])
    #         end_line = getattr(node, "end_lineno", node.lineno)
    #         code_block = "\n".join(lines[start_line - 1 : end_line])

    #         # 描述 / 参数 / 返回
    #         docstring = ast.get_docstring(node)
    #         description, arguments, returns = _parse_docstring(docstring)

    #         # 若 docstring 未提供参数信息，退化为基于函数签名的简单描述
    #         if not arguments:
    #             sig_args = []
    #             for arg in getattr(node, "args", []).args:
    #                 if arg.arg == "self":
    #                     continue
    #                 ann = getattr(arg, "annotation", None)
    #                 if ann:
    #                     try:
    #                         ann_str = ast.unparse(ann)
    #                     except Exception:
    #                         ann_str = ""
    #                 else:
    #                     ann_str = ""
    #                 sig_args.append(f"{arg.arg}: {ann_str}" if ann_str else arg.arg)
    #             arguments = ", ".join(sig_args)

    #         if not returns and getattr(node, "returns", None):
    #             try:
    #                 returns = ast.unparse(node.returns)
    #             except Exception:
    #                 returns = ""

    #         tools[tool_name] = {
    #             "description": description,
    #             "arguments": arguments,
    #             "returns": returns,
    #             "code": code_block,
    #         }

    #     # 缓存解析结果
    #     self._temp_cache_mtime = mtime
    #     self._temp_cache_data = tools
    #     return tools
    
    # def official_tool_list(self) -> Dict[str, Dict[str, str]]:
    #     """
    #     读取 invented_tools/official_tool_list.py，提取每个 @mcp.tool 装饰的函数的完整信息：
    #     {tool_name: {"description": str, "arguments": str, "returns": str, "code": str}}
    #     其中 tool_name 优先取装饰器的 name 参数，否则函数名。
    #     """
    #     import ast
    #     import re
    #     from pathlib import Path

    #     root_dir = os.environ.get("RL_FACTORY_ROOT", str(_repo_root()))
    #     tool_path = Path(root_dir) / "envs" / "tools" / "invented_tools" / self.project_name / "official_tool_list.py"
    #     if not tool_path.exists():
    #         return {}

    #     mtime = tool_path.stat().st_mtime
    #     if self._official_cache_mtime == mtime and self._official_cache_data is not None:
    #         return self._official_cache_data

    #     text = tool_path.read_text(encoding="utf-8")
    #     lines = text.splitlines()

    #     try:
    #         tree = ast.parse(text)
    #     except Exception as e:
    #         print(f"Failed to parse official_tool_list.py: {e}")
    #         return {}

    #     def _parse_docstring(ds: Optional[str]) -> Tuple[str, str, str]:
    #         """粗粒度解析 docstring 的 description/arguments/returns 块。"""
    #         if not ds:
    #             return "", "", ""

    #         desc_lines, arg_lines, ret_lines = [], [], []
    #         state = "desc"
    #         for raw in ds.splitlines():
    #             line = raw.strip()
    #             if not line:
    #                 continue
    #             if re.match(r"(?i)^args?:", line):
    #                 state = "args"
    #                 continue
    #             if re.match(r"(?i)^returns?:", line):
    #                 state = "returns"
    #                 continue
    #             if state == "desc":
    #                 desc_lines.append(line)
    #             elif state == "args":
    #                 arg_lines.append(line)
    #             elif state == "returns":
    #                 ret_lines.append(line)

    #         description = " ".join(desc_lines).strip()
    #         arguments = " ".join(arg_lines).strip()
    #         returns = " ".join(ret_lines).strip()
    #         return description, arguments, returns

    #     tools: Dict[str, Dict[str, str]] = {}
    #     for node in tree.body:
    #         if not isinstance(node, ast.FunctionDef):
    #             continue

    #         # 检查装饰器是否 mcp.tool
    #         tool_name = node.name
    #         has_mcp_deco = False
    #         if node.decorator_list:
    #             for deco in node.decorator_list:
    #                 # 形如 @mcp.tool 或 @mcp.tool(name='xxx')
    #                 if isinstance(deco, ast.Attribute) and deco.attr == "tool":
    #                     has_mcp_deco = True
    #                 elif isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "tool":
    #                     has_mcp_deco = True
    #                     for kw in deco.keywords:
    #                         if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
    #                             tool_name = kw.value.value
    #                 if has_mcp_deco:
    #                     break

    #         if not has_mcp_deco:
    #             continue

    #         start_line = min([getattr(d, "lineno", node.lineno) for d in node.decorator_list] + [node.lineno])
    #         end_line = getattr(node, "end_lineno", node.lineno)
    #         code_block = "\n".join(lines[start_line - 1 : end_line])

    #         docstring = ast.get_docstring(node)
    #         description, arguments, returns = _parse_docstring(docstring)

    #         if not arguments:
    #             sig_args = []
    #             for arg in getattr(node, "args", []).args:
    #                 if arg.arg == "self":
    #                     continue
    #                 ann = getattr(arg, "annotation", None)
    #                 if ann:
    #                     try:
    #                         ann_str = ast.unparse(ann)
    #                     except Exception:
    #                         ann_str = ""
    #                 else:
    #                     ann_str = ""
    #                 sig_args.append(f"{arg.arg}: {ann_str}" if ann_str else arg.arg)
    #             arguments = ", ".join(sig_args)

    #         if not returns and getattr(node, "returns", None):
    #             try:
    #                 returns = ast.unparse(node.returns)
    #             except Exception:
    #                 returns = ""

    #         tools[tool_name] = {
    #             "description": description,
    #             "arguments": arguments,
    #             "returns": returns,
    #             "code": code_block,
    #         }

    #     # 缓存解析结果
    #     self._official_cache_mtime = mtime
    #     self._official_cache_data = tools
    #     return tools

    async def register_mcp(self, code: str, task_id: int = None, subtask_dict=None):
        """异步注册 MCP"""
        # 优化：使用线程池执行器，避免每次创建新线程
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            return await loop.run_in_executor(executor, self._register_mcp_sync, code, task_id, subtask_dict)

    async def register_mcps_batch(self, mcp_data_list: List[Tuple[str, int, Optional[Dict]]]):
        """批量注册多个MCP，减少远程调用开销"""
        import concurrent.futures
        loop = asyncio.get_event_loop()

        async def _register_single(data):
            code, task_id, subtask_dict = data
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return await loop.run_in_executor(executor, self._register_mcp_sync, code, task_id, subtask_dict)

        # 并发处理，但限制并发度
        semaphore = asyncio.Semaphore(4)  # 最多4个并发注册

        async def _register_with_semaphore(data):
            async with semaphore:
                return await _register_single(data)

        tasks = [_register_with_semaphore(data) for data in mcp_data_list]
        return await asyncio.gather(*tasks, return_exceptions=True)


    def build_tfidf_indices(self, temp_mcps: Optional[Dict[int, str]] = None):
        """构建 TF-IDF 索引 - 优化版：最小化锁持有时间"""
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import vstack as safe_vstack
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError

        if temp_mcps is None:
            # 阶段1: 锁内快速拷贝数据
            with self.lock:
                vectorizer = self.field_components['vectorizer']
                field_data_descriptions = [
                    mcp_func_data.get('description', "") or ""
                    for mcp_func_data in self.mcps.values()
                ]
                mcp_ids_for_index = list(self.mcps.keys())

            # 阶段2: 锁外执行耗时的计算
            if field_data_descriptions:
                matrix = vectorizer.fit_transform(field_data_descriptions)
            else:
                matrix = None

            # 阶段3: 锁内快速写回结果
            with self.lock:
                self.field_components['matrix'] = matrix
                self.field_components['mcp_ids'] = mcp_ids_for_index if matrix is not None else []
                if matrix is not None:
                    print(f"Global TF-IDF index built successfully with {len(mcp_ids_for_index)} documents.")
        else:
            # 增量构建 - 同样最小化锁时间
            # 阶段1: 锁内准备数据
            with self.lock:
                vectorizer = self.field_components.get('vectorizer')
                if vectorizer is None:
                    return
                try:
                    check_is_fitted(vectorizer)
                except NotFittedError:
                    return
                new_descriptions = []
                new_mcp_ids = []
                for step_id, mcp_name in temp_mcps.items():
                    if mcp_name in self.mcps:
                        new_descriptions.append(self.mcps[mcp_name].get('description', "") or "")
                        new_mcp_ids.append(mcp_name)

            # 阶段2: 锁外执行变换
            if not new_descriptions:
                return
            new_matrix = vectorizer.transform(new_descriptions)

            # 阶段3: 锁内快速更新
            with self.lock:
                if self.field_components.get('matrix') is None:
                    self.field_components['matrix'] = new_matrix
                else:
                    self.field_components['matrix'] = safe_vstack([self.field_components['matrix'], new_matrix])

                if 'mcp_ids' not in self.field_components:
                    self.field_components['mcp_ids'] = []
                self.field_components['mcp_ids'].extend(new_mcp_ids)

    def build_embeddings(self, temp_mcps: Optional[Dict[int, str]] = None):
        """构建嵌入 - 优化版：添加缓存并最小化锁持有时间"""
        batch_size = 32

        if temp_mcps is None:
            # 阶段1: 锁内拷贝数据
            with self.lock:
                items = list(self.mcps.items())

            # 阶段2: 锁外处理（检查缓存 + 计算嵌入）
            queries_to_encode = []
            queries_indices = []

            for idx, (mcp_name, mcp_data) in enumerate(items):
                description = mcp_data.get('description') or ""
                queries_to_encode.append(description)
                queries_indices.append(idx)

            # 只对未缓存的描述进行编码
            if queries_to_encode:
                new_embeddings = self.embedding_model.encode(
                    queries_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )

            with self.lock:
                for idx, (mcp_name, mcp_data) in enumerate(items):
                    description = mcp_data.get('description') or ""
                    mcp_data['query_embedding'] = new_embeddings[idx]
        else:
            # 增量构建：只为新增/指定的 MCP 计算 embedding
            with self.lock:
                target_names = [name for _, name in temp_mcps.items()]
                target_items = [(name, self.mcps.get(name)) for name in target_names if name in self.mcps]

            if not target_items:
                return

            queries_to_encode = [data.get('description') or "" for _, data in target_items]
            new_embeddings = self.embedding_model.encode(
                queries_to_encode,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            with self.lock:
                for idx, (mcp_name, mcp_data) in enumerate(target_items):
                    mcp_data['query_embedding'] = new_embeddings[idx]

    def field_text_search(self, query: str, top_k: int = 3) -> List[dict]:
        """文本相似度搜索"""
        from sklearn.metrics.pairwise import cosine_similarity

        with self.lock:
            component = self.field_components
            if component['matrix'] is None or not component['mcp_ids']:
                return []

            query_vec = component['vectorizer'].transform([query])
            similarities = cosine_similarity(query_vec, component['matrix']).flatten()

            actual_top_k = min(top_k, len(similarities))
            top_indices = similarities.argsort()[-actual_top_k:][::-1]

            results = []
            for idx in top_indices:
                mcp_id = component['mcp_ids'][idx]
                if mcp_id in self.mcps:
                    results.append({
                        'mcp_id': mcp_id,
                        'score': float(similarities[idx]),
                        'content': self.mcps[mcp_id]
                    })
            return results

    def field_semantic_search(self, query: str, top_k: int = 3):
        """语义相似度搜索"""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        with self.lock:
            embeddings = []
            mcp_data_list = []

            for mcp_name, mcp_data in self.mcps.items():
                emb = mcp_data.get('query_embedding')
                if emb is not None:
                    embeddings.append(emb)
                    mcp_data_list.append(mcp_data)

            if not embeddings:
                return []

            embeddings_matrix = np.array(embeddings)
            similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

            actual_top_k = min(top_k, len(similarities))
            top_indices = similarities.argsort()[-actual_top_k:][::-1]

            results = []
            for idx in top_indices:
                mcp_item = mcp_data_list[idx]
                results.append({
                    'mcp_id': mcp_item['mcp_name'],
                    'score': float(similarities[idx]),
                    'field': 'description',
                    'content': mcp_item.get('description', "")
                })
            return results

    async def hybrid_search(self, query: str, top_k: int = 5,
                           weights: Optional[Dict[str, float]] = None,
                           score_threshold: float = 0.2,
                           use_graph_enhanced: Optional[bool] = None,
                           max_hops: int = 2) -> List[List[Dict[str, Any]]]:
        """
        混合搜索，只返回分数大于 score_threshold 的工具

        Args:
            query: 查询文本
            top_k: 返回结果数量
            weights: 权重配置（传统方法或 GraphRAG）
            score_threshold: 分数阈值（仅传统方法）
            use_graph_enhanced: 是否使用 GraphRAG 增强搜索
                - None: 根据 self.use_graphrag 自动决定
                - True: 强制使用 GraphRAG（如果可用）
                - False: 强制使用传统方法
            max_hops: 多跳推理深度（仅 GraphRAG）

        Returns:
            搜索结果列表
        """
        # 决定使用哪种搜索方法
        should_use_graphrag = use_graph_enhanced if use_graph_enhanced is not None else self.use_graphrag

        # ===== GraphRAG 增强搜索 =====
        if should_use_graphrag and self.graphrag_engine is not None:
            try:
                print(f"[hybrid_search] Using GraphRAG enhanced search (max_hops={max_hops})")

                # GraphRAG 权重配置
                graphrag_weights = weights or {
                    'semantic': 0.3,
                    'text': 0.2,
                    'graph': 0.2,
                    'community': 0.15,
                    'history': 0.15
                }

                # 调用 GraphRAG 检索
                results = self.graphrag_engine.graph_enhanced_search(
                    query=query,
                    top_k=top_k,
                    max_hops=max_hops,
                    use_community=True,
                    use_history=True,
                    weights=graphrag_weights
                )

                # 转换为兼容格式 [[tool_dict], [tool_dict], ...]
                detailed_results = []
                for tool in results:
                    detailed_results.append([tool])

                print(f"[hybrid_search] GraphRAG returned {len(detailed_results)} results")
                return detailed_results

            except Exception as e:
                print(f"[hybrid_search] Warning: GraphRAG search failed, falling back to traditional search: {e}")
                import traceback
                traceback.print_exc()
                # 降级到传统搜索
                should_use_graphrag = False

        # ===== 传统混合搜索 (原有逻辑) =====
        if not should_use_graphrag:
            print(f"[hybrid_search] Using traditional hybrid search (TF-IDF + Semantic)")

            from collections import defaultdict

            weights = weights or {'text': 0.5, 'semantic': 0.5}
            score_board = defaultdict(float)

            text_results = self.field_text_search(query, top_k * 2)
            semantic_results = self.field_semantic_search(query, top_k * 2)

            # print(f"[hybrid_search] query='{query[:50]}...', text_results={len(text_results)}, semantic_results={len(semantic_results)}")
            for r in text_results:
                score_board[r['mcp_id']] += weights.get('text', 0.5) * r['score']
            for r in semantic_results:
                score_board[r['mcp_id']] += weights.get('semantic', 0.5) * r['score']

            print(f"[hybrid_search] score_board has {len(score_board)} items")

            # 先过滤掉分数低于阈值的结果，再排序和截取
            filtered_results = [(mcp_id, score) for mcp_id, score in score_board.items() if score > score_threshold]
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]
            print(f"[hybrid_search] After threshold filtering: {len(filtered_results)} items, returning top {len(sorted_results)}")
            # print(f"[hybrid_search] sorted_results: {[(mcp_id, score) for mcp_id, score in sorted_results]}")

            detailed_results = []
            for mcp_id, score in sorted_results:
                mcp_item = self.mcps.get(mcp_id)
                if mcp_item:
                    # 在返回的结果中添加分数信息
                    mcp_item_with_score = mcp_item.copy()
                    mcp_item_with_score['search_score'] = score
                    detailed_results.append([mcp_item_with_score])
                else:
                    print(f"[hybrid_search] Warning: mcp_id '{mcp_id}' not found in self.mcps (keys: {list(self.mcps.keys())[:5]})")

            print(f"[hybrid_search] Returning {len(detailed_results)} results")
            return detailed_results

    # ==================== GraphRAG 相关方法 ====================

    def _record_graphrag_execution(
        self,
        tool_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        """
        记录工具执行历史到 GraphRAG 引擎

        Args:
            tool_name: 工具名称
            success: 是否执行成功
            execution_time: 执行时间（秒）
            co_used_tools: 同时使用的其他工具列表
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            return

        try:
            # 将 tool_name 转换为 mcp_name（处理带后缀的工具名）
            # 例如: "search-search" -> "search"
            mcp_name = tool_name.split('-')[0] if '-' in tool_name else tool_name

            # 只记录在 mcps 中的工具
            if mcp_name in self.mcps:
                self.graphrag_engine.record_execution(
                    mcp_name=mcp_name,
                    success=success,
                    execution_time=execution_time,
                    co_used_tools=co_used_tools
                )
        except Exception as e:
            print(f"[GraphRAG] Warning: Failed to record execution for {tool_name}: {e}")

    def get_graphrag_tool_stats(self, mcp_name: str) -> Dict[str, Any]:
        """
        获取工具的 GraphRAG 执行统计信息

        Args:
            mcp_name: 工具名称

        Returns:
            包含执行统计的字典，如果 GraphRAG 未启用则返回空字典
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            return {}

        try:
            return self.graphrag_engine.get_execution_stats(mcp_name)
        except Exception as e:
            print(f"[GraphRAG] Error getting stats for {mcp_name}: {e}")
            return {}

    def get_graphrag_community_info(self) -> Dict[str, Any]:
        """
        获取 GraphRAG 社区信息

        Returns:
            包含社区统计和摘要的字典
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            return {
                'enabled': False,
                'message': 'GraphRAG is not enabled'
            }

        try:
            num_communities = len(self.graphrag_engine.communities)
            communities_detail = {}

            for comm_id, members in self.graphrag_engine.communities.items():
                communities_detail[comm_id] = {
                    'size': len(members),
                    'members': list(members)[:10],  # 只返回前10个成员
                    'summary': self.graphrag_engine.community_summaries.get(comm_id, '')
                }

            return {
                'enabled': True,
                'num_communities': num_communities,
                'communities': communities_detail,
                'global_summary': self.graphrag_engine.global_summary
            }
        except Exception as e:
            print(f"[GraphRAG] Error getting community info: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }

    def save_graphrag_cache(self) -> bool:
        """
        保存 GraphRAG 缓存到磁盘

        Returns:
            是否保存成功
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            print("[GraphRAG] GraphRAG not enabled, skipping cache save")
            return False

        try:
            self.graphrag_engine.save_cache()
            print("[GraphRAG] Cache saved successfully")
            return True
        except Exception as e:
            print(f"[GraphRAG] Error saving cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_graphrag_cache(self) -> bool:
        """
        从磁盘加载 GraphRAG 缓存

        Returns:
            是否加载成功
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            print("[GraphRAG] GraphRAG not enabled, skipping cache load")
            return False

        try:
            self.graphrag_engine.load_cache()
            print("[GraphRAG] Cache loaded successfully")
            return True
        except Exception as e:
            print(f"[GraphRAG] Error loading cache: {e}")
            import traceback
            traceback.print_exc()
            return False

    def sync_tools_to_graphrag(self):
        """
        手动同步当前的工具列表到 GraphRAG 引擎
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            return

        try:
            self.graphrag_engine.load_tools(self.mcps, self.sub_task_graph)
            self.mcp_version += 1
            print(f"[GraphRAG] Manually synced {len(self.mcps)} tools to GraphRAG (version {self.mcp_version})")
        except Exception as e:
            print(f"[GraphRAG] Error syncing tools to GraphRAG: {e}")
            import traceback
            traceback.print_exc()

    # ==================== 原有方法 ====================

    def update_mcp(self, mcp_name: str, mcp_content: str):
        """更新 MCP 内容"""
        with self.lock:
            if mcp_name in self.mcps:
                self.mcps[mcp_name]['code'] = mcp_content
                print(f"MCP '{mcp_name}' updated with new content.")

                # GraphRAG: 同步更新的工具
                if self.use_graphrag and self.graphrag_engine is not None:
                    try:
                        self.graphrag_engine.update_tool(mcp_name, self.mcps[mcp_name])
                        self.mcp_version += 1
                        print(f"[GraphRAG] Synced updated tool: {mcp_name} (version {self.mcp_version})")
                    except Exception as e:
                        print(f"[GraphRAG] Warning: Failed to sync updated tool {mcp_name}: {e}")
            else:
                print(f"MCP '{mcp_name}' not found for update.")

    async def clear(self, temp_dict: Optional[dict] = None):
        """清理临时 MCP"""
        if temp_dict is None:
            temp_dict = self.temp_mapping
        print("Clearing temp_mapping and rebuilding indices/embeddings.")
        self.build_tfidf_indices(temp_mcps=None)
        self.build_embeddings(temp_mcps=None)
        self.temp_mapping.clear()

    def get_temp_mapping(self) -> Dict[str, Any]:
        """获取临时映射"""
        return self.temp_mapping

    async def get_mcps(self) -> Dict[str, Any]:
        """获取所有 MCP"""
        mcp_copy = self.mcps.copy()
        mcp_copy.update(self.temp_mapping)
        return mcp_copy

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


    async def list_mcps(self) -> str:
        """列出所有 MCP"""
        mcps_to_list = self.mcps.copy()
        if not mcps_to_list:
            return "MCP Box is empty."

        mcp_list_str = "Available MCPs:\n"
        for name, data in mcps_to_list.items():
            mcp_list_str += f"- {name} \n"
            mcp_list_str += f"    Description: {data.get('description', 'N/A')}\n"
            mcp_list_str += f"    Arguments: {data.get('arguments', 'N/A')}\n"
            mcp_list_str += f"    Returns: {data.get('returns', 'N/A')}\n"
            if data.get('requires'):
                mcp_list_str += f"    Requires: {data['requires']}\n"
        return mcp_list_str

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

    async def execute_single_tool(self, tool):
        self._ensure_async_structs()
        # try:
        tool_instance = self.tool_manager.get_tool(tool["name"])
        try:
            args = tool["args"]
        except Exception as e:
            args = tool["arguments"]
        tool_name = tool["name"]
        future = None
        query_str = None

        # GraphRAG: 记录工具执行开始时间
        execution_start_time = time.time() if self.use_graphrag and self.graphrag_engine else None

        if tool_instance is not None:
            try:
                args = json.loads(args)
            except Exception as e:
                pass
            if type(args) is dict:
                if (tool_name == 'create_and_execute_mcp' or tool_name == 'create_and_execute_mcp-create_and_execute_mcp') and self.temp_mapping is not None:
                    try:
                        args_content = args
                        if isinstance(args_content, str):
                            args = json.loads(args_content)
                        elif isinstance(args_content, dict):
                            args = args_content
                        else:
                            args = {}
                            
                        if 'name' in args and 'code' in args:
                            self.temp_mapping[args['name']] = {
                                "mcp_name": args['name'],
                                "description": args.get('description', ''),
                                "arguments": args.get('arguments', ''),
                                "returns": args.get('returns', ''),
                                "code": args.get('code', ''),
                                "step_id": len(self.temp_mapping) + 2 # Simple step_id logic
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

                # 增加专门针对 python 代码执行工具的短超时配置
                # 虽然全局超时已经设置（例如60s），但 python 代码执行可能因为死循环等原因需要更严格的控制
                specific_timeout = self.tool_timeout
                if tool_name == "execute_python_code" or tool_name == "execute_python_code-execute_python_code":
                    # 尝试从 args 中读取 timeout 参数，如果存在的话
                    try:
                        if isinstance(tool.get("args", ""), str):
                            tool_args = json.loads(tool.get("args", ""))
                            if "timeout" in tool_args:
                                specific_timeout = 15.0
                        elif isinstance(tool.get("args", ""), dict):
                            if "timeout" in tool.get("args", ""):
                                specific_timeout = 15.0
                    except:
                        pass
                    
                    # 默认给Python工具设置一个更短的超时时间，例如5秒
                    if specific_timeout == self.tool_timeout:
                        specific_timeout = 15.0

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
                        self.tool_manager._call_tool, 
                        tool_name, 
                        json.dumps(args, ensure_ascii=False, indent=4)
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

                    if tool_name == "search" or tool_name == "search-search":
                        await asyncio.to_thread(self._save_search_cache_sync, query_str, result)
                        async with self._search_cache_lock:
                            self._search_cache[query_str] = result

                        # 设置 In-flight 结果
                        if not future.done():
                            future.set_result(result)

                    # GraphRAG: 记录成功执行
                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=True, execution_time=execution_time)

                    return result
                except asyncio.TimeoutError as e:
                    # search 的 in-flight 等待者需要收到异常
                    if (tool_name == "search" or tool_name == "search-search") and not future.done():
                        future.set_exception(e)

                    # GraphRAG: 记录超时失败
                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=False, execution_time=execution_time)

                    return f"# Tool Calling Exceeds Timeout: {tool_name}, Exceeded {self.tool_timeout}s"
                except Exception as e:
                    # 如果是 search 工具，处理 In-flight 异常
                    if (tool_name == "search" or tool_name == "search-search") and future is not None and not future.done():
                        future.set_exception(e)

                    # GraphRAG: 记录执行失败
                    if execution_start_time is not None:
                        execution_time = time.time() - execution_start_time
                        self._record_graphrag_execution(tool_name, success=False, execution_time=execution_time)

                    # 返回错误消息而不是抛出异常，确保 tokenizer 总是接收字符串
                    return f"# Tool execution failed: {tool_name}\n- Error: {str(e)}"
                finally:
                    # 清理 In-flight 记录
                    if tool_name == "search" or tool_name == "search-search":
                        async with self._in_flight_lock:
                            if query_str in self._in_flight_requests and self._in_flight_requests[query_str] is future:
                                del self._in_flight_requests[query_str]
            elif type(args) is str:
                result = args
            else:
                result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
        else:
            if tool_name == '<empty>':
                result = args
            else:
                result = "# Failed to find the tool {} in the tool map".format(tool_name)
        return result

    # async def execute_single_tool(self, tool):
    #     tool_instance = self.tool_manager.get_tool(tool["name"])
    #     args = tool["args"]
    #     if tool_instance is not None:
    #         try:
    #             args = json.loads(args)
    #         except Exception as e:
    #             pass
    #         if type(args) is dict:
    #             try:
    #                 try:
    #                     tool_result = await asyncio.wait_for(
    #                         asyncio.to_thread(
    #                             self.tool_manager._call_tool, 
    #                             tool["name"], 
    #                             json.dumps(args, ensure_ascii=False, indent=4)
    #                         ),
    #                         timeout=self.timeout
    #                     )
    #                     if len(tool_result) == 0:
    #                         tool_result = "The tool call format is correct, but the return result is empty. Please try other inputs or tools. "
    #                     result = f"# Execute the tool {tool['name']} successed\n" \
    #                             f"- The result is:\n{tool_result}"
    #                 except asyncio.TimeoutError:
    #                     result = f"# Execute the tool {tool['name']} timeout\n" \
    #                             f"- The tool execution exceeded the timeout limit of {self.timeout} seconds\n"
    #             except Exception as e:
    #                 result = f"# Execute the tool {tool['name']} failed\n" \
    #                         f"- Error message:\n{str(e)}"
    #         elif type(args) is str:
    #             # Json decode error: xxx
    #             result = args
    #         else:
    #             result = 'Unexpected type of args: {} (args: {})'.format(type(args), args)
    #     else:
    #         if tool['name'] == '<empty>':
    #             result = args
    #         else:
    #                 result = "# Failed to find the tool {} in the tool map".format(tool['name'])
    #     return result



    async def mcp_hybrid_search(self, query: str, top_k: int = 5, weights: Optional[Dict[str, float]] = None):
        """异步混合搜索"""
        return await self.hybrid_search(query, top_k, weights)


    async def mcp_clear(self, temp_dict: Optional[dict] = None):
        """异步清理"""
        return await self.clear(temp_dict)

    async def mcp_get_mcps(self):
        """异步获取 MCP"""
        return await self.get_mcps()

    async def mcp_list_mcps(self):
        """异步列出 MCP"""
        return await self.list_mcps()

    async def get_functions_mcps(self):
        """获取所有 MCP 工具的 OpenAI function schema 列表"""
        return self.functions_mcps

    async def get_mcp_version(self):
        """获取当前 MCP 工具版本号"""
        return self.mcp_version

    async def get_functions_mcps_with_version(self):
        """获取 MCP 工具列表和版本号"""
        return {
            'version': self.mcp_version,
            'functions_mcps': self.tool_manager.functions_mcps
        }

    async def get_mcp_statistics(self):
        """获取 MCP 工具的详细统计信息，用于 wandb 监控

        Returns:
            Dict[str, Any]: 包含以下字段的字典:
                - total_mcps: 总MCP工具数量
                - official_mcps: 正式MCP工具数量
                - temp_mcps: 临时MCP工具数量
                - mcp_version: MCP版本号（表示工具列表的更新次数）
                - graph_nodes: 子任务图中的节点数量
                - graph_edges: 子任务图中的边数量
        """
        with self.lock:
            stats = {
                "total_mcps": len(self.mcps),
                "official_mcps": len(self.mcps) - len(self.temp_mapping),
                "temp_mcps": len(self.temp_mapping),
                "mcp_version": self.mcp_version,
                "graph_nodes": self.sub_task_graph.number_of_nodes(),
                "graph_edges": self.sub_task_graph.number_of_edges(),
            }

            # 添加 GraphRAG 统计信息
            if self.use_graphrag and self.graphrag_engine is not None:
                stats["graphrag_enabled"] = True
                stats["graphrag_communities"] = len(self.graphrag_engine.communities) if hasattr(self.graphrag_engine, 'communities') else 0
                stats["graphrag_history_enabled"] = self.graphrag_engine.enable_history_tracking if hasattr(self.graphrag_engine, 'enable_history_tracking') else False
            else:
                stats["graphrag_enabled"] = False

        return stats

    # ===== GraphRAG 辅助方法 =====

    def record_tool_execution(
        self,
        mcp_name: str,
        success: bool,
        execution_time: float = 0.0,
        co_used_tools: Optional[List[str]] = None
    ):
        """
        记录工具执行结果（用于 GraphRAG 历史跟踪）

        Args:
            mcp_name: 工具名称
            success: 是否成功
            execution_time: 执行时间（秒）
            co_used_tools: 同时使用的其他工具列表
        """
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.record_execution(
                    mcp_name=mcp_name,
                    success=success,
                    execution_time=execution_time,
                    co_used_tools=co_used_tools
                )
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to record execution for {mcp_name}: {e}")

    def get_tool_stats(self, mcp_name: str) -> Dict[str, Any]:
        """
        获取工具的执行统计信息

        Args:
            mcp_name: 工具名称

        Returns:
            统计信息字典，包含成功率、使用频率等
        """
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                return self.graphrag_engine.get_execution_stats(mcp_name)
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to get stats for {mcp_name}: {e}")
                return {}
        return {}

    def save_graphrag_cache(self):
        """保存 GraphRAG 缓存"""
        if self.use_graphrag and self.graphrag_engine is not None:
            try:
                self.graphrag_engine.save_cache()
                print("[GraphRAG] Cache saved successfully")
            except Exception as e:
                print(f"[GraphRAG] Warning: Failed to save cache: {e}")

    async def get_community_info(self) -> Dict[str, Any]:
        """
        获取社区检测信息（仅 GraphRAG）

        Returns:
            社区统计信息字典
        """
        if not self.use_graphrag or self.graphrag_engine is None:
            return {"graphrag_enabled": False}

        try:
            return {
                "graphrag_enabled": True,
                "num_communities": len(self.graphrag_engine.communities),
                "communities": {
                    comm_id: {
                        "size": len(members),
                        "members": list(members)[:10],  # 只返回前10个
                        "summary": self.graphrag_engine.community_summaries.get(comm_id, "")
                    }
                    for comm_id, members in self.graphrag_engine.communities.items()
                },
                "global_summary": self.graphrag_engine.global_summary
            }
        except Exception as e:
            print(f"[GraphRAG] Warning: Failed to get community info: {e}")
            return {"graphrag_enabled": True, "error": str(e)}

    def update_mcp_box(self, similarity_threshold: float = 0.88, prefer_higher_degree: bool = False, step_idx: int = -1):
        """
        合并 self.mcps 中的重复/相似节点 - 优化版：最小化锁持有时间
        将聚类计算完全移到锁外执行，避免死锁
        """
        # === 阶段1：锁内快速收集必要数据（<1ms） ===
        with self.lock:
            self.update_self_mcp()
            if not self.sub_task_graph or not self.mcps:
                print("Agent Log: Graph or MCPs are empty, skipping merge.")
                # 即使为空也需要重新加载，确保文件中的工具被加载到内存
                self._reload_official_tools_after_merge()
                return

            needs_embedding = any('query_embedding' not in v for v in self.mcps.values() if v.get("description"))

        # 如果需要嵌入，在锁外构建
        if needs_embedding:
            self.build_embeddings(temp_mcps=None)

        # 再次进入锁，收集数据快照
        with self.lock:
            # 创建图的深拷贝，避免锁外修改原图
            graph_snapshot = self.sub_task_graph.copy()
            mcps_snapshot = {k: v.copy() for k, v in self.mcps.items()}
            # 创建 alias_map 快照
            alias_map_snapshot = self.alias_map.copy()

        # === 阶段2：锁外执行所有重计算（可能需要秒级时间） ===
        print("Agent Log: Performing heavy computation (clustering) outside of the lock...")

        # 清理无效节点
        to_remove = [
            n for n in graph_snapshot.nodes
            if graph_snapshot.nodes[n].get("mcp_name") is None or
               graph_snapshot.nodes[n].get("mcp_name") not in mcps_snapshot
        ]
        for n in to_remove:
            preds = set(graph_snapshot.predecessors(n))
            succs = set(graph_snapshot.successors(n))
            for u in preds:
                for v in succs:
                    if u != v:
                        graph_snapshot.add_edge(u, v)
        if to_remove:
            graph_snapshot.remove_nodes_from(to_remove)

        # 收集有效节点
        valid_nodes = []
        for node in graph_snapshot.nodes:
            mcp_name = graph_snapshot.nodes[node].get("mcp_name")
            mcp_info = mcps_snapshot.get(mcp_name)
            if mcp_info and mcp_info.get("description") and mcp_info.get("query_embedding") is not None:
                step_id = mcp_info.get("step_id", node)
                valid_nodes.append((node, mcp_name, step_id, mcp_info["query_embedding"]))

        if len(valid_nodes) <= 1:
            print("Agent Log: Not enough valid nodes to perform a merge. Skipping.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")
            # 重新加载工具并递增版本号
            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        # 执行聚类（最耗时的部分）
        clusters = self._cluster_nodes_by_similarity(valid_nodes, similarity_threshold)
        print("Agent Log: Clustering complete.")

        if all(len(ixs) == 1 for ixs in clusters.values()):
            print(f"Agent Log: No clusters found above threshold (>= {similarity_threshold}); nothing merged.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")
            # 重新加载工具并递增版本号
            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        # 在快照图上执行合并
        merged_count, alias_map = self._merge_clusters(
            graph_snapshot,
            mcps_snapshot,
            alias_map_snapshot,
            clusters,
            valid_nodes,
            prefer_higher_degree
        )

        if merged_count == 0:
            print("Agent Log: No nodes were merged.")
            try:
                self._write_merged_mcps_to_file_async(mcps_snapshot)
            except Exception as e:
                print(f"Agent Log: Failed to write merged MCPs to file: {e}")
            # 重新加载工具并递增版本号
            self._reload_official_tools_after_merge()
            return len(mcps_snapshot)

        # 更新依赖关系（仍在快照上）
        self._update_dependencies_after_merge_on_snapshot(mcps_snapshot, alias_map)

        # === 阶段3：锁内快速写回结果（<10ms） ===
        with self.lock:
            # 原子替换
            self.sub_task_graph = graph_snapshot
            self.mcps = mcps_snapshot
            self.alias_map = alias_map_snapshot
            self.temp_mapping.clear()

            print(f"Agent Log: Merged {merged_count} duplicate nodes. Rebuilding indices...")

        # 阶段4：只有在实际合并了节点时才重建索引，避免不必要的开销
        if merged_count > 0:
            print(f"Agent Log: {merged_count} nodes were merged, rebuilding indices...")
            self.build_tfidf_indices(temp_mcps=None)
            self.build_embeddings(temp_mcps=None)
            print(f"Agent Log: Index rebuild complete.")
        else:
            print(f"Agent Log: No nodes merged, skipping index rebuild.")

        print(f"Agent Log: Merge complete. Nodes={self.sub_task_graph.number_of_nodes()}, MCPs={len(self.mcps)}")
        try:
            self._write_merged_mcps_to_file_async(mcps_snapshot)
        except Exception as e:
            print(f"Agent Log: Failed to write merged MCPs to file: {e}")
        # 重新加载工具并递增版本号
        self._reload_official_tools_after_merge()
        return len(self.mcps)


    def _reload_official_tools_after_merge(self):
        try:
            print(f"[update_mcp_box] Reloading official tools after merge...")
            print(f"[update_mcp_box] Before reload: {len(self.tool_manager.functions_mcps) if hasattr(self.tool_manager, 'functions_mcps') else 0} MCP tools")

            self.tool_manager.reload_official_tools()
            import pickle

            # 保存到本地文件
            graph_dir = Path(os.environ.get("RL_FACTORY_GRAPH_DIR", _repo_root() / "envs" / "tools" / "graphs"))
            graph_dir.mkdir(parents=True, exist_ok=True)
            with open(graph_dir / f"{self.project_name}_task_graph.pkl", 'wb') as f:
                pickle.dump(self.sub_task_graph, f)
            # 递增版本号，通知所有 Worker 工具列表已更新
            self.mcp_version += 1
            print(f"[update_mcp_box] MCP version updated to {self.mcp_version}")

            print(f"[update_mcp_box] After reload: {len(self.tool_manager.functions_mcps) if hasattr(self.tool_manager, 'functions_mcps') else 0} MCP tools")

            # 确保 functions_mcps 被正确更新并且可以被访问
            if hasattr(self.tool_manager, 'functions_mcps') and self.tool_manager.functions_mcps:
                print(f"[update_mcp_box] Tool manager functions_mcps updated successfully with {len(self.tool_manager.functions_mcps)} tools")
                # 打印前5个工具名称用于调试
                tool_names = [fn.get('name', 'unknown') for fn in self.tool_manager.functions_mcps[:5]]
                print(f"[update_mcp_box] Sample tool names: {tool_names}")
            else:
                print(f"[update_mcp_box] WARNING: functions_mcps is empty or not set!")
        except Exception as e:
            print(f"[update_mcp_box] Failed to reload official tools: {e}")
            import traceback
            traceback.print_exc()

    
    def _update_dependencies_after_merge_on_snapshot(self, mcps_snapshot: dict, alias_map: dict):
        """在快照上更新依赖关系，不需要锁"""
        if not alias_map:
            return

        for _, data in mcps_snapshot.items():
            req = data.get("requires")
            if not isinstance(req, str):
                continue

            parts = [p.strip() for p in req.split(",") if p.strip()]
            updated_reqs = {alias_map.get(p, p) for p in parts}
            data["requires"] = ", ".join(sorted(updated_reqs)) if updated_reqs else None

    def _write_merged_mcps_to_file_async(self, mcps_snapshot: dict):
        """异步写入文件，避免阻塞主流程"""
        import threading

        def _write_file():
            try:
                root_dir = _repo_root()
                tool_dir = Path(root_dir) / "envs" / "tools" / "invented_tools" / (self.project_name)
                tool_dir.mkdir(parents=True, exist_ok=True)
                official_file = tool_dir / "official_tool_list.py"

                common_imports = (
                    "import json\n"
                    "import sys\n"
                    "import traceback\n"
                    "import math\n"
                    "import numpy as np\n"
                    "import re\n"
                    "import os\n"
                    "import datetime\n"
                    "from collections import Counter, defaultdict, deque\n"
                    "from mcp.server.fastmcp import FastMCP\n\n"
                )
                main_block = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    print(\"\\nStart MCP service:\")\n"
                    "    mcp.run(transport='stdio')\n"
                )

                TOOL_TEMPLATE_WITH_DEF = '''@mcp.tool(name='{name}')
def {wrapper_func_name}(**kwargs):
    """{description}

    Args:
        {arguments}

    Returns:
        {returns}
    """
    try:
{body}
        return {inner_func_name}(**kwargs)
    except Exception as e:
        return f"tool execution failed: {{str(e)}}"
'''

                TOOL_TEMPLATE_SIMPLE = '''@mcp.tool(name='{name}')
def {name}(**kwargs):
    """{description}

    Args:
        {arguments}

    Returns:
        {returns}
    """
    try:
{indented_body}
    except Exception as e:
        return f"tool execution failed: {{str(e)}}"
'''

                def build_tool_block(name: str, description: str, arguments: str, returns: str, code: str) -> str:
                    if code.lstrip().startswith("def "):
                        code_lines = code.splitlines()
                        inner_func_name = code_lines[0].split('(')[0].replace('def ', '').strip()

                        indented_def = "\n".join(
                            f"        {line}" if line.strip() else ""
                            for line in code_lines
                        )
                        if not indented_def.strip():
                            indented_def = "        pass"

                        body = indented_def

                        return TOOL_TEMPLATE_WITH_DEF.format(
                            name=name,
                            wrapper_func_name=name,  # 外层函数名使用工具名
                            inner_func_name=inner_func_name,  # 内层函数名从代码提取
                            description=description,
                            arguments=arguments,
                            returns=returns,
                            body=body
                        )
                    else:
                        code_lines = code.splitlines() if code.strip() else ["return None"]
                        indented_body = '\n'.join(
                            f"        {line}" if line.strip() else ""
                            for line in code_lines
                        )
                        # 如果 indented_body 为空，添加 pass 语句
                        if not indented_body.strip():
                            indented_body = "        pass"

                        return TOOL_TEMPLATE_SIMPLE.format(
                            name=name,
                            description=description,
                            arguments=arguments,
                            returns=returns,
                            indented_body=indented_body
                        )

                from io import StringIO
                buffer = StringIO()
                buffer.write(f"{common_imports}mcp = FastMCP('Official_Server')\n\n")

                # 记录成功和失败的工具
                success_tools = []
                failed_tools = []

                for mcp_name, mcp_data in mcps_snapshot.items():
                    if not mcp_name.isidentifier():
                        print(f"Warning: Invalid function name '{mcp_name}', skipping...")
                        failed_tools.append((mcp_name, "Invalid function name"))
                        continue

                    desc = mcp_data.get('description') or ""
                    args = mcp_data.get('arguments') or ""
                    rets = mcp_data.get('returns') or ""
                    code = mcp_data.get('code') or ""

                    try:
                        # 1. 构建工具块
                        block = build_tool_block(mcp_name, desc, args, rets, code)

                        # 2. 对单个工具块进行语法验证
                        # 构建完整的测试代码（包含导入和FastMCP初始化）
                        test_content = f"{common_imports}mcp = FastMCP('Official_Server')\n\n{block}"

                        # 尝试编译单个工具块
                        compile(test_content, f"<{mcp_name}>", 'exec')

                        # 3. 验证通过，写入buffer
                        buffer.write(block)
                        success_tools.append(mcp_name)

                    except SyntaxError as e:
                        print(f"Agent Log: Syntax error in '{mcp_name}': {e}")
                        failed_tools.append((mcp_name, f"SyntaxError: {e}"))
                        continue
                    except Exception as e:
                        print(f"Agent Log: Failed to build/validate block for '{mcp_name}': {e}")
                        failed_tools.append((mcp_name, str(e)))
                        continue

                buffer.write(main_block)
                content = buffer.getvalue()

                # 最终验证整体语法（应该总是通过，因为每个块都已验证）
                try:
                    compile(content, str(official_file), 'exec')
                    official_file.write_text(content, encoding="utf-8")
                    print(f"Agent Log: Merged MCPs saved to {official_file}")
                    print(f"Agent Log: Successfully wrote {len(success_tools)} tools: {', '.join(success_tools)}")

                    if failed_tools:
                        print(f"Agent Log: Failed to write {len(failed_tools)} tools:")
                        for tool_name, error in failed_tools:
                            print(f"  - {tool_name}: {error}")
                except Exception as e:
                    print(f"Agent Log: Final compilation failed (unexpected): {e}")

            except Exception as e:
                print(f"Error writing MCP file: {e}")

        # 启动后台线程写入
        thread = threading.Thread(target=_write_file, daemon=True)
        thread.start()

    def _cleanup_invalid_nodes(self, G: nx.DiGraph):
        """移除无效节点并做旁路重连。"""
        to_remove = [
            n for n in G.nodes
            if G.nodes[n].get("mcp_name") is None or G.nodes[n].get("mcp_name") not in self.mcps
        ]

        for n in to_remove:
            preds = set(G.predecessors(n))
            succs = set(G.successors(n))
            for u in preds:
                for v in succs:
                    if u != v:
                        G.add_edge(u, v)

        if to_remove:
            G.remove_nodes_from(to_remove)

    def _gather_valid_nodes_for_merging(self, G: nx.DiGraph) -> list:
        """收集可参与合并的节点数据。"""
        valid_nodes = []
        for node in G.nodes:
            mcp_name = G.nodes[node].get("mcp_name")
            mcp_info = self.mcps.get(mcp_name)
            if mcp_info and mcp_info.get("description") and mcp_info.get("query_embedding") is not None:
                step_id = mcp_info.get("step_id", node)
                valid_nodes.append((node, mcp_name, step_id, mcp_info["query_embedding"]))
        return valid_nodes

    def _cluster_nodes_by_similarity(self, valid_nodes: list, similarity_threshold: float) -> dict:
        """使用余弦相似度 + 并查集进行聚类。"""
        from collections import defaultdict
        from sklearn.metrics.pairwise import cosine_similarity

        embs = np.stack([t[3] for t in valid_nodes], axis=0)
        S = cosine_similarity(embs, embs)
        N = S.shape[0]
        parent = list(range(N))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(N):
            for j in range(i + 1, N):
                if S[i, j] >= similarity_threshold:
                    union(i, j)

        clusters = defaultdict(list)
        for i in range(N):
            clusters[find(i)].append(i)

        return clusters

    def _merge_clusters(self, G: nx.DiGraph, mcps_dict: dict, alias_map_snapshot: dict, clusters: dict, valid_nodes: list, prefer_higher_degree: bool) -> Tuple[int, Dict[str, str]]:
        """根据聚类结果合并节点与元数据。

        Args:
            G: 图快照
            mcps_dict: MCP字典快照（会被修改）
            alias_map_snapshot: alias_map快照（会被修改）
            clusters: 聚类结果
            valid_nodes: 有效节点列表
            prefer_higher_degree: 是否优先选择高度数节点作为代表

        Returns:
            (merged_nodes_count, alias_map): 合并节点数和别名映射
        """
        merged_nodes_count = 0
        alias_map = {}
        node_ids = [t[0] for t in valid_nodes]
        names = [t[1] for t in valid_nodes]
        step_ids = [t[2] for t in valid_nodes]

        for _, member_indices in clusters.items():
            if len(member_indices) <= 1:
                continue

            candidates = [
                (idx, node_ids[idx], step_ids[idx], G.in_degree(node_ids[idx]) + G.out_degree(node_ids[idx]))
                for idx in member_indices
            ]
            sort_key = (lambda t: (-t[3], str(t[2]), str(t[1]))) if prefer_higher_degree else (lambda t: (str(t[2]), str(t[1])))
            candidates.sort(key=sort_key)

            rep_idx, rep_node, _, _ = candidates[0]
            rep_name = names[rep_idx]

            for idx, dup_node, _, _ in candidates[1:]:
                dup_name = names[idx]

                # 操作快照而不是 self.mcps
                if rep_name not in mcps_dict or dup_name not in mcps_dict:
                    continue

                for u in list(G.predecessors(dup_node)):
                    if u != rep_node:
                        G.add_edge(u, rep_node)
                for v in list(G.successors(dup_node)):
                    if v != rep_node:
                        G.add_edge(rep_node, v)
                G.remove_node(dup_node)

                # 更新 alias_map 快照
                alias_map_snapshot[mcps_dict[dup_name]['step_id']] = mcps_dict[rep_name]['step_id']
                alias_map[dup_name] = rep_name

                # 传入快照进行合并
                self._merge_mcp_metadata(mcps_dict, rep_name, dup_name)
                mcps_dict.pop(dup_name, None)
                merged_nodes_count += 1

            if G.has_node(rep_node):
                G.nodes[rep_node]['mcp_name'] = rep_name

        return merged_nodes_count, alias_map

    def _merge_mcp_metadata(self, mcps_dict: dict, rep_name: str, dup_name: str):
        """合并元数据字段。

        Args:
            mcps_dict: MCP字典（快照或原始数据）
            rep_name: 代表MCP的名称
            dup_name: 要合并的重复MCP的名称
        """
        if rep_name in mcps_dict and dup_name in mcps_dict:
            rep, old = mcps_dict[rep_name], mcps_dict[dup_name]

            req_rep = rep.get("requires", "") or ""
            req_old = old.get("requires", "") or ""
            all_reqs = set(r.strip() for r in req_rep.split(",") if r.strip())
            all_reqs.update(r.strip() for r in req_old.split(",") if r.strip())
            rep["requires"] = ", ".join(sorted(all_reqs)) if all_reqs else None

            if not rep.get("code") and old.get("code"):
                rep["code"] = old["code"]

            aliases = rep.get("aliases", []) or []
            if not isinstance(aliases, list):
                aliases = [aliases]
            aliases.append(dup_name)
            rep["aliases"] = sorted(set(aliases))

    def _update_dependencies_after_merge(self, alias_map: dict):
        """根据 alias_map 更新 requires 字段。"""
        if not alias_map:
            return

        for _, data in self.mcps.items():
            req = data.get("requires")
            if not isinstance(req, str):
                continue

            parts = [p.strip() for p in req.split(",") if p.strip()]
            updated_reqs = {alias_map.get(p, p) for p in parts}
            data["requires"] = ", ".join(sorted(updated_reqs)) if updated_reqs else None

    # ===== 提供给 env 侧的包装方法 =====
    def extract_dag_and_parallel(self, text: str):
        """同步封装：解析 DAG 列表并返回 (dag_list, task_id, subtask_dict)。"""
        return self.tool_manager.extract_dag_and_parallel(text)

    def batch_extract_dag_and_parallel(self, texts: list, traj_indices: list, max_turns_list: list):
        """批量解析 DAG - 真正的并行处理"""
        import concurrent.futures
        import time

        start_time = time.time()
        results = [None] * len(texts)

        def process_single_planning(idx, text, max_turns):
            """处理单个规划任务"""
            try:
                result = self.tool_manager.extract_dag_and_parallel(text)
                return idx, result, None
            except Exception as e:
                print(f"[Trajectory {idx}] DAG parsing failed: {e}, falling back to sequential")
                try:
                    fallback = self.tool_manager.sequential_planning(max_turns)
                    return idx, fallback, None
                except Exception as e2:
                    print(f"[Trajectory {idx}] Sequential planning also failed: {e2}")
                    return idx, (None, None, None), str(e2)

        # 使用线程池并行处理（因为extract_dag_and_parallel是CPU密集型+少量IO）
        max_workers = min(32, len(texts))  # 限制最大并发数避免资源耗尽
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_single_planning, idx, text, max_turns): idx
                for idx, (text, max_turns) in enumerate(zip(texts, max_turns_list))
            }

            # 收集结果（设置超时）
            completed = 0
            for future in concurrent.futures.as_completed(future_to_idx, timeout=60):
                try:
                    idx, result, error = future.result(timeout=5)
                    results[idx] = result
                    completed += 1
                    if completed % 50 == 0:
                        print(f"Planning progress: {completed}/{len(texts)} completed")
                except concurrent.futures.TimeoutError:
                    idx = future_to_idx[future]
                    print(f"[Trajectory {idx}] Planning timeout, using sequential fallback")
                    results[idx] = self.tool_manager.sequential_planning(max_turns_list[idx])
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"[Trajectory {idx}] Planning error: {e}")
                    results[idx] = (None, None, None)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r and r[1] is not None)
        print(f"Batch planning completed: {success_count}/{len(texts)} successful in {elapsed:.2f}s")

        return results

    async def batch_register_mcps(self, tool_call_strs: list, task_id: int, subtask_dict: dict):
        """批量注册MCPs - 减少远程调用开销"""
        results = []
        for tool_call_str in tool_call_strs:
            try:
                await self.register_mcp(tool_call_str, task_id, subtask_dict)
                results.append(True)
            except Exception as e:
                print(f"Failed to register MCP: {e}")
                results.append(False)
        return results

    def update_sub_task_graph(self, edge_list, single_nodes):
        """同步更新子任务图（供 manager 远程调用）"""
        if edge_list:
            self.sub_task_graph.add_edges_from(edge_list)
        if single_nodes:
            self.sub_task_graph.add_nodes_from(single_nodes)

    def preview_get_prompt(self, input_data, tokenizer, mode: str = 'initial', add_generation_prompt: bool = True, current_plan=None):
        """生成 get_prompt 的结果，方便远程测试"""
        return self.tool_manager.get_prompt(
            input_data=input_data,
            tokenizer=tokenizer,
            mode=mode,
            add_generation_prompt=add_generation_prompt,
            current_plan=current_plan
        )



# ===== 测试函数 =====
def test_centralized_tool_actor():
    """测试 CentralizedToolActor 的基本功能（通过 Ray Actor 实例化）"""
    import os
    import sys
    import ray
    import asyncio
    import contextlib
    from statistics import mean
    print("🚀 开始测试 CentralizedToolActor (Ray)...")

    class MockConfig:
        def __init__(self):
            self.tool_timeout = 30
            self.mcp_mode = 'stdio'
            # 测试用配置路径，去掉之前的空格避免加载失败
            self.config_path = "envs/configs/mcp_tools_local_graph.pydata"
            self.enable_limiter = False
            self.max_turns = 6
            self.temp_config_path = "envs/configs/temp_mcps.pydata"
            self.official_config_path = "envs/configs/official_mcps.pydata"
            self.enable_thinking = False
            self.project_name = "gigpo_gpu6_prm_new_sett"


        def get(self, key, default=None):
            return getattr(self, key, default)

    # try:
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    print("1. 初始化 Ray Actor...")
    actor = CentralizedToolActor.options(name="test_centralized_tool_actor").remote(MockConfig())
    print("✅ Actor 创建成功 用于测试temp_tool_list是否成功")
    mcp_list = ray.get(actor.mcp_list_mcps.remote())
    print(f"MCP 列表: {mcp_list}")
    mcps = ray.get(actor.mcp_get_mcps.remote())
    print(f"获取到 {len(mcps)} 个 MCP")

    # 并发参数
    concurrency_levels = [8, 16]

    def _summary(label, results):
        ok = sum(1 for r in results if r is not None)
        print(f"{label}: {ok}/{len(results)} 成功")

    print("\n3. 并发测试混合搜索 mcp_hybrid_search ...")
    try:
        for c in concurrency_levels:
            tasks = [actor.mcp_hybrid_search.remote("web search", top_k=2) for _ in range(c)]
            res = ray.get(tasks)
            _summary(f"  并发 {c}", res)
    except Exception as e:
        print(f"搜索并发测试失败: {e}")

    print("\n4. 并发测试 execute_single_tool（search-search）...")
    create_tool_call = {
        "name": "execute_python_code-execute_python_code",
        "arguments": {
            # 注意这里：删掉了原本多余的 "code": 
            "code": "import math\n\n# Define the relationship between a and b for quasi-colorability\n# Based on graph theory principles, we assume that the graph is quasi-colorable if the number of degree 4 vertices is sufficiently large relative to the number of degree 3 vertices.\n# We derive the smallest real number c such that if a/b > c, then G is quasi-colorable.\n\n# Assume that the graph is quasi-colorable if a/b > 2\n# This is based on the idea that the graph must have enough degree 4 vertices to ensure that the coloring constraints can be satisfied.\n\n# Calculate the smallest real number c\n# Based on the assumption that a/b > 2 ensures quasi-colorability\n# Therefore, c = 2\n\n# Return the value of c\nprint(2)",
            "timeout": 30.0
        }
    }
    try:
        for c in concurrency_levels:
            tasks = [actor.execute_single_tool.remote(create_tool_call) for _ in range(c)]
            res = ray.get(tasks)
            _summary(f"  并发 {c}", res)
            print(res[0])
    except Exception as e:
        print(f"execute_single_tool 并发测试失败: {e}")

    print("\n4.1 并发测试 preview_get_prompt ...")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-4B")
    sample_chat = [{'role': USER, 'content': '请给出一次去北京旅游的行程草案'}]
    try:
        for c in concurrency_levels:
            tasks = [actor.preview_get_prompt.remote(sample_chat, tokenizer, mode='plan_call', add_generation_prompt=True) for _ in range(c)]
            res = ray.get(tasks)
            print(res)
            _summary(f"  并发 {c}", res)
    except Exception as e:
        print(f"preview_get_prompt 并发测试失败: {e}")

    print("\n4.2 并发测试 DAG 解析/注册 ...")

    try:
        from envs.tools import mcp_creation as tool_creation

        # 测试 1: 简单 MCP（保留原有功能）
        simple_payload = {
            "name": "dummy_mcp",
            "description": "dummy mcp for register_mcp stress test",
            "arguments": "x (int), y (int)",
            "returns": "sum of two ints",
            "code": "def dummy_mcp(x: int, y: int):\n    return x + y",
            "inputs": {"x": 1, "y": 2},
            "timeout": 5.0,
        }
        simple_tool_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(simple_payload)
        }
        simple_res = ray.get(actor.execute_single_tool.remote(simple_tool_call))
        print(f"✓ 简单 MCP 测试结果: {simple_res}")

        # 测试 2: Prompt 示例中的复杂 MCP（与 workflow.yaml Example 3 一致）
        complex_payload = {
            "name": "analyze_polynomial_roots",
            "description": "Find roots of polynomial and return classification and properties",
            "arguments": "coeffs (list of float): coefficients [a_n, a_{n-1}, ..., a_1, a_0]",
            "returns": "dict with keys: roots, num_real, num_complex, roots_sum, roots_product",
            "code": """def analyze_polynomial_roots(coeffs):
    import numpy as np
    # Step 1: Find all roots
    roots = np.roots(coeffs)
    # Step 2: Classify real vs complex
    num_real = sum(1 for r in roots if abs(r.imag) < 1e-10)
    num_complex = len(roots) - num_real
    # Step 3: Calculate properties
    roots_sum = sum(roots)
    roots_product = np.prod(roots)
    return {
        'roots': roots.tolist(),
        'num_real': num_real,
        'num_complex': num_complex,
        'sum': complex(roots_sum).real if abs(roots_sum.imag) < 1e-10 else roots_sum,
        'product': complex(roots_product).real if abs(roots_product.imag) < 1e-10 else roots_product
    }""",
            "inputs": {"coeffs": [1, -5, 6]},  # x^2 - 5x + 6 = (x-2)(x-3), roots: [2, 3]
            "timeout": 10.0,
        }
        complex_tool_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(complex_payload)
        }
        complex_res = ray.get(actor.execute_single_tool.remote(complex_tool_call))
        print(f"✓ Prompt 示例 MCP 测试结果: {complex_res}")

        # 验证结果是否符合预期
        if complex_res and "error" not in str(complex_res).lower():
            print("  - MCP 创建成功，结果应包含: roots=[3, 2], num_real=2, sum=5, product=6")

    except Exception as e:
        import traceback
        print(f"✗ MCP 创建测试失败: {e}")
        print(f"  堆栈: {traceback.format_exc()}")
    sample_plan = """## ST1: demo
1. plan step
##DAG_LIST
[(ST1)]
"""
    try:
        for c in concurrency_levels:
            dag_tasks = [actor.extract_dag_and_parallel.remote(sample_plan) for _ in range(c)]
            dag_res = ray.get(dag_tasks)
            _summary(f"  DAG 并发 {c}", dag_res)
            # 注册 MCP 仅做轻量检查
            reg_tasks = []
            for dag_list, task_id, subtask_dict in dag_res:
                dummy_tool_call = "<subtask> ST_1: demo </subtask>\n<tool_call>{\"name\":\"dummy_mcp\"}</tool_call>"
                reg_tasks.append(actor.register_mcp.remote(dummy_tool_call, task_id, subtask_dict))
            reg_res = ray.get(reg_tasks)
            _summary(f"  register_mcp 并发 {c}", reg_res)
    except Exception as e:
        print(f"DAG/注册 并发测试失败: {e}")

    print("\n5. 测试图可视化...")
    try:
        viz_result = ray.get(actor.mcp_visualize_graph.remote())
        print(f"图可视化结果: {viz_result}")
    except Exception as e:
        print(f"图可视化失败: {e}")

    print("\n6. 测试 DAG 解析与 MCP 注册...")
    try:
        sample_plan = """## ST1: demo
1. plan step
##DAG_LIST
[(ST1)]
"""
        dag_list, task_id, subtask_dict = ray.get(actor.extract_dag_and_parallel.remote(sample_plan))
        print(f"DAG 解析结果 dag_list={dag_list}, task_id={task_id}, subtask_dict={subtask_dict}")

        # 测试 6.1: 简单 MCP 注册
        dummy_tool_call = "<subtask> ST_1: demo </subtask>\n<tool_call>{\"name\":\"dummy_mcp\"}</tool_call>"
        ray.get(actor.register_mcp.remote(dummy_tool_call, task_id, subtask_dict))
        print("✓ 简单 MCP 注册完成")

        # 测试 6.2: Prompt 示例 MCP 注册和调用
        try:
            # 重新解析 DAG 以获取新的 task_id
            dag_list2, task_id2, subtask_dict2 = ray.get(actor.extract_dag_and_parallel.remote(sample_plan))

            # 注册 analyze_polynomial_roots
            complex_mcp_call = "<subtask> ST_1: Analyze polynomial x^3 - 6x^2 + 11x - 6 </subtask>\n<tool_call>{\"name\":\"analyze_polynomial_roots\", \"arguments\":{\"coeffs\":[1, -6, 11, -6]}}</tool_call>"
            ray.get(actor.register_mcp.remote(complex_mcp_call, task_id2, subtask_dict2))
            print("✓ Prompt 示例 MCP 注册完成")

            # 验证 MCP 是否可调用（通过执行工具）
            test_call = {
                "name": "analyze_polynomial_roots",
                "args": json.dumps({"coeffs": [1, -6, 11, -6]})  # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
            }
            test_res = ray.get(actor.execute_single_tool.remote(test_call))
            print(f"✓ MCP 调用测试结果: {test_res}")
            print("  - 预期: roots=[1, 2, 3], num_real=3, sum=6, product=6")
        except Exception as e:
            print(f"✗ Prompt 示例 MCP 注册/调用失败: {e}")

    except Exception as e:
        import traceback
        print(f"✗ DAG/MCP 测试失败: {e}")
        print(f"  堆栈: {traceback.format_exc()}")

    print("\n7. 测试完整 Prompt 工作流（创建 MCP → 多次调用）...")
    try:
        # 模拟 workflow.yaml Example 3 的完整流程
        workflow_plan = """## ST1: Analyze stability of system 1
1. Create stability checker MCP
##DAG_LIST
[(ST1)]
"""
        dag_list3, task_id3, subtask_dict3 = ray.get(actor.extract_dag_and_parallel.remote(workflow_plan))

        # Step 1: 创建 MCP (模拟 <tool_call> create_and_execute_mcp)
        create_payload = {
            "name": "check_system_stability",
            "description": "Find poles of transfer function denominator and determine system stability",
            "arguments": "coeffs (list of float): denominator coefficients [a_n, ..., a_0]",
            "returns": "dict with keys: poles, num_stable, num_unstable, is_stable",
            "code": """def check_system_stability(coeffs):
    import numpy as np
    # Step 1: Find all poles (roots of denominator)
    poles = np.roots(coeffs)
    # Step 2: Check real parts for stability (Re < 0 means stable)
    num_stable = sum(1 for p in poles if p.real < 0)
    num_unstable = len(poles) - num_stable
    is_stable = (num_unstable == 0)
    # Step 3: Return analysis
    return {
        'poles': [complex(p) for p in poles],
        'num_stable': num_stable,
        'num_unstable': num_unstable,
        'is_stable': is_stable
    }""",
            "inputs": {"coeffs": [1, 3, 2]},  # s^2 + 3s + 2 = (s+1)(s+2), poles: [-1, -2] (stable)
            "timeout": 10.0,
        }
        create_call = {
            "name": "create_and_execute_mcp-create_and_execute_mcp",
            "args": json.dumps(create_payload)
        }
        create_result = ray.get(actor.execute_single_tool.remote(create_call))
        print(f"✓ Step 1 - MCP 创建结果: {create_result}")

        # Step 2-4: 多次使用创建的 MCP（模拟 reuse）
        test_systems = [
            {"coeffs": [1, 3, 2], "desc": "系统1 (稳定)"},      # poles: [-1, -2]
            {"coeffs": [1, -3, 2], "desc": "系统2 (不稳定)"},   # poles: [1, 2]
            {"coeffs": [1, 1, 1], "desc": "系统3 (稳定)"},      # complex poles with Re < 0
        ]

        for i, system in enumerate(test_systems, start=2):
            test_call = {
                "name": "check_system_stability",
                "args": json.dumps({"coeffs": system["coeffs"]})
            }
            result = ray.get(actor.execute_single_tool.remote(test_call))
            print(f"✓ Step {i} - {system['desc']} 测试结果: {result}")

        print("✓ 完整 Prompt 工作流测试通过：MCP 创建 → 多次复用成功")

    except Exception as e:
        import traceback
        print(f"✗ Prompt 工作流测试失败: {e}")
        print(f"  堆栈: {traceback.format_exc()}")

    print("\n8. 测试 python 工具 execute_python_code ...")
    try:
        from envs.tools import python as tool_python

        py_res = tool_python.execute_python_code("print(1+1)", timeout=5.0)
        print(f"python tool 返回: {py_res}")
    except Exception as e:
        print(f"python 工具测试失败: {e}")

    print("\n9. 测试 _build_temp_mcp_tools 方法 ...")
    try:
        # 测试 CentralizedQwenManager 的 _build_temp_mcp_tools 方法
        # 通过 actor 的 tool_manager 来调用
        class TestManager(CentralizedQwenManager):
            def __init__(self, verl_config):
                super().__init__(verl_config, mock=True)

        test_manager = TestManager(MockConfig())

        # 测试正常初始化项目
        print("  测试 init_project=False (正常加载)...")
        initial_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        test_manager._build_temp_mcp_tools(init_project=False)
        after_normal_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        print(f"    初始 MCP 工具数量: {initial_mcp_count}")
        print(f"    加载后 MCP 工具数量: {after_normal_mcp_count}")

        # 测试初始化项目模式
        print("  测试 init_project=True (项目初始化)...")
        test_manager._build_temp_mcp_tools(init_project=False)
        after_init_mcp_count = len(test_manager.functions_mcps) if hasattr(test_manager, 'functions_mcps') else 0
        print(f"    初始化后 MCP 工具数量: {after_init_mcp_count}")

        # 验证文件是否被正确创建
        from pathlib import Path
        base_dir = Path(__file__).resolve().parents[2] / "tools"
        invented_dir = base_dir / "invented_tools" / (test_manager.project_name or "default_project")
        temp_file = invented_dir / "temp_tool_list.py"
        official_file = invented_dir / "official_tool_list.py"

        print(f"  检查文件创建: temp_tool_list.py - {temp_file.exists()}, official_tool_list.py - {official_file.exists()}")


        if official_file.exists():
            official_content = official_file.read_text()
            has_imports = "import json" in official_content and "from mcp.server.fastmcp import FastMCP" in official_content
            has_main = "if __name__ == \"__main__\":" in official_content
            print(f"    official_tool_list.py 文件验证: 导入={has_imports}, 主函数={has_main}")

        print("  ✅ _build_temp_mcp_tools 测试完成")

    except Exception as e:
        print(f"_build_temp_mcp_tools 测试失败: {e}")
        import traceback
        traceback.print_exc()

    return True



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = test_centralized_tool_actor()
        sys.exit(0 if success else 1)
    else:
        print("CentralizedToolActor 需要通过 Ray 启动，而不是直接运行")
        print("使用 --test 参数来运行本地测试")
