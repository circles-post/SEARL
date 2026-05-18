import re
import sys
import json
import inspect
import asyncio
import importlib.util
from bs4 import BeautifulSoup
from typing import Any, Dict, Optional, List
from envs.tool_manager.base_manager import ToolManager


class ConfigParser:
    """配置解析器，用于读取和解析配置文件"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


class ToolParam:
    """工具参数类"""
    
    def __init__(self, param_config: Dict[str, Any]):
        """初始化工具参数
        
        Args:
            param_config: 参数配置字典
        """
        self.id = param_config.get("id", "")
        self.name = param_config.get("name", "")
        self.type = param_config.get("type", "string")
        self.description = param_config.get("description", "")
        self.required = param_config.get("required", False)
        self.default = param_config.get("default", None)
    
    def convert_value(self, value: Any) -> Any:
        """根据参数类型转换值
        
        Args:
            value: 输入值
            
        Returns:
            转换后的值
        """
        try:
            if self.type == "string":
                return str(value)
            elif self.type == "number":
                return float(value)
            elif self.type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "t", "y")
                return bool(value)
            elif self.type == "array":
                if isinstance(value, str):
                    return json.loads(value)
                return list(value)
            elif self.type == "object":
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value)
            else:
                return value
        except Exception as e:
            print(f"转换参数 {self.name} 失败: {e}")
            return value


class Tool:
    """工具类，表示内部工具或外部工具"""
    
    def __init__(self, tool_config: Dict[str, Any]):
        """初始化工具
        
        Args:
            tool_config: 工具配置字典
        """
        self.id = tool_config.get("id", "")
        self.name = tool_config.get("name", "")
        self.short_name = tool_config.get("shortName", "")
        self.type = tool_config.get("type", "")
        self.description = tool_config.get("description", "")
        self.python_code = tool_config.get("pythonCode", "")
        self._func = None  # 用于存储实际的函数对象
        
        # 只有外部工具才有参数
        if self.type == "external":
            # 创建参数对象
            self.params = [ToolParam(param) for param in tool_config.get("params", [])]
            
            # 创建参数名称映射
            self.param_map = {param.name: param for param in self.params}
        else:
            # 内部工具没有参数
            self.params = []
            self.param_map = {}
        
        # 如果是外部工具且有Python代码，编译并加载函数
        if self.type == "external" and self.python_code:
            self._compile_function()
    
    def _compile_function(self):
        """编译Python代码为函数"""
        try:
            # 创建一个新的模块
            module_name = f"tool_{self.id}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # 执行Python代码
            exec(self.python_code, module.__dict__)
            
            # 查找定义的函数
            for name, obj in module.__dict__.items():
                if inspect.isfunction(obj):
                    self._func = obj
                    break
            
            if not self._func:
                raise ValueError(f"工具 {self.name} 的Python代码中没有找到函数定义")
        
        except Exception as e:
            print(f"编译工具 {self.name} 失败: {e}")
    
    def __call__(self, *args, **kwargs):
        """调用工具函数"""
        if not self._func:
            raise ValueError(f"工具 {self.name} 没有可调用的函数")
        
        # 只处理外部工具的参数
        processed_kwargs = {}
        if self.type == "external":
            # 处理关键字参数
            for key, value in kwargs.items():
                if key in self.param_map:
                    param = self.param_map[key]
                    processed_kwargs[key] = param.convert_value(value)
        else:
            # 内部工具直接传递参数，不做处理
            processed_kwargs = kwargs
        
        # 调用函数
        return self._func(*args, **processed_kwargs)
    
    def get_param(self, name: str) -> Optional[ToolParam]:
        """通过名称获取参数
        
        Args:
            name: 参数名称
            
        Returns:
            找到的参数，如果没找到则返回None
        """
        if self.type != "external":
            return None  # 内部工具没有参数
        
        return self.param_map.get(name)


class ConfigManager(ToolManager):    
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
        return self.internal_tools + self.external_tools

    def _build_tools(self):
        config_path = self.verl_config.config_path
        self.config = ConfigParser.load_config(config_path)
        
        # 提取基本信息
        self.id = self.config.get("id", "")
        self.name = self.config.get("name", "")
        self.description = self.config.get("description", "")
        
        # 提取系统提示词模板
        self.system_prompt_template = self.config.get("systemPrompt", "")
        self.user_prompt_template = self.config.get("userPrompt", "")
        
        # 创建工具
        self.internal_tools = self._create_tools(self.config.get("internalTools", []))
        self.external_tools = self._create_tools(self.config.get("externalTools", []))
        
        # 获取所有参数
        all_params = self.config.get("params", [])
        user_params = self.config.get("userParams", [])
        
        # 初始化字段字典
        self.env_fields = {}      # 环境字段字典
        self.template_fields = {} # 模板字段字典
        self.user_fields = {}     # 用户字段字典
        
        # 处理所有参数，根据category分类
        for param in all_params:
            param_name = param.get("name", "")
            if not param_name:
                continue
                
            # 根据category进行分类
            category = param.get("category", "")
                
            # 根据category进行分类
            if category == "模板字段":
                self.template_fields[param_name] = param
            elif category == "环境字段":
                self.env_fields[param_name] = param
                # 将环境字段设置为实例变量
                param_default = param.get("default")
                setattr(self, param_name, param_default if param_default is not None else None)
        
        # 处理用户字段（userParams中的字段）
        for param in user_params:
            param_name = param.get("name", "")
            if not param_name:
                continue
            
            self.user_fields[param_name] = param
        
        # 创建工具名称和简写映射
        self.tool_map = {}
        for tool in self.internal_tools + self.external_tools:
            self.tool_map[tool.name] = tool
            if tool.short_name:
                self.tool_map[tool.short_name] = tool

    def _create_tools(self, tool_configs):
        """从配置创建工具列表
        
        Args:
            tool_configs: 工具配置列表
            
        Returns:
            工具对象列表
        """
        tools = []
        for tool_config in tool_configs:
            tool = Tool(tool_config)
            tools.append(tool)

        return tools

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
        
    async def _execute_tool_batch(self, action, tools):
        """异步并行执行一批工具
        
        Args:
            tools: 工具列表
            
        Returns:
            工具执行结果的列表
        """        
        async def execute_single_tool(tool):
            tool_instance = self.tool_manager.get_tool(tool["name"])
            args = tool["args"]
            if tool_instance is not None:
                if type(args) is dict:
                    try:
                        # 使用asyncio.to_thread将同步函数转换为异步执行
                        tool_result = await asyncio.to_thread(tool_instance, **args)
                        result = """# Execute the tool {} successed
  - The args are: {}
  - The result is:
{}""".format(tool["name"], args, tool_result)
                    except Exception as e:
                        result = """# Execute the tool {} failed
  - The original args are: {}
  - Error message:
{}""".format(tool["name"], args, str(e))
                elif type(args) is str:
                    # Json decode error: xxx
                    result = args
                else:
                    raise ValueError('Unexpected type of args: {}'.format(type(args)))
            else:
                if tool['name'] == '<empty>':
                    result = args
                else:
                    result = "# Failed to find the tool {} in the tool map".format(tool['name'])

            return result
        
        if action == 'answer':
            # 'tools' is a str (the answer)
            result_str = tools
        elif action == 'error':
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            # ('Cannot extract the actions tag' or 'There is no action after extraction')
            result_str = """# Extract the actions failed due to: {}""".format(tools)
        elif action == 'actions':
            # 'tools' is the list of the 'Tool' instances
            tasks = [execute_single_tool(temp_tool) for temp_tool in tools]
            results = await asyncio.gather(*tasks)
            result_str = '\n'.join(results)
        else:
            raise ValueError('Unexpected action: {}'.format(action))

        return result_str
    
    def execute_actions(self, responses: List[str]):
        # action: answer, xxx or error
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
        if_tools, tools = self.parse_tools(response_content)
        if if_tools:
            return 'actions', tools

        return None
    
    def parse_end_flag(self, response_content: str) -> bool:
        answer = None
        answer_section = re.findall(r'(<answer>.*?</answer>)', response_content, re.DOTALL)
        if len(answer_section) > 0:
            answer = answer_section[-1]
        
        return True, answer
    
    def parse_tools(self, response_content: str):
        actions_section = re.findall(r'(<actions>.*?</actions>)', response_content, re.DOTALL)
        if len(actions_section) > 0:
            soup = BeautifulSoup(actions_section[-1], 'html.parser')
            
            # 'error' only occurs when there is no 'actions' tag or there is no 'action' tag after extraction
            ext_actions = soup.find_all('actions')[-1]
            if not ext_actions:
                return 'error', 'Cannot extract the actions tag'
            
            ext_action_list = ext_actions.find_all('action')
            if not ext_action_list:
                return 'error', 'There is no action after extraction'
            
            tool_list = []
            for ext_action in ext_action_list:
                # 提取name标签内容
                name_tag = ext_action.find('name')
                if name_tag is not None:
                    action_name = name_tag.text.strip()
                    
                    # 提取args标签内容（可选）
                    args_tag = ext_action.find('args')
                    if args_tag:
                        args_text = args_tag.text.strip()
                        try:
                            args_dict = json.loads(args_text)
                        except json.JSONDecodeError as e:
                            args_dict = f"Json decode error: {e}"
                else:
                    action_name = '<empty>'
                    args_dict = """# Extract the tool name failed"""
                
                tool = {
                    "name": action_name,
                    "args": args_dict
                }
                tool_list.append(tool)

            return True, tool_list
        else:
            return False, None

    def get_prompt(self, input_data, tokenizer, mode='initial', add_generation_prompt=True):
        assert mode in ['initial', 'tool_call', 'assistant_response'], 'Invalid mode: {}'.format(mode)
        base_chat = [
            {'role': 'system', 'content': 'base'},
            {'role': 'user', 'content': 'base'},
        ]
        base_prompt = tokenizer.apply_chat_template(
            conversation=base_chat,
            tools=self.functions,
            tokenize=False
        )

        if mode == 'initial':
            chat = input_data
            prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=chat, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        elif mode in ['tool_call', 'assistant_response']:
            # NOTE: the assistant response might not be used
            role = 'tool' if mode == 'tool_call' else 'assistant'
            chat = {'role': role, 'content': input_data}
            temp_prompt_with_chat_template = tokenizer.apply_chat_template(
                conversation=base_chat + [chat], tokenize=False, add_generation_prompt=add_generation_prompt
            )
            prompt_with_chat_template = temp_prompt_with_chat_template.replace(base_prompt, '')
        else:
            raise ValueError('Invalid mode: {}'.format(mode))
        
        return prompt_with_chat_template