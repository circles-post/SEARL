from envs.tool_manager.qwen3_manager import QwenManager
from envs.tool_manager.centralized.centralized_qwen3_manager import CentralizedQwenManager
import os
import tempfile
import shutil
from pathlib import Path


def test_manager():
    env_config = {
        'name': 'base',
        'tool_manager': 'qwen3',
        'mcp_mode': 'sse',
        'config_path': 'envs/configs/sse_mcp_tools.pydata',
        'enable_thinking': True,
        'max_prompt_length': 2048,
    }
    manager = QwenManager(env_config)
    print('Tools:')
    for tool_name, tool in manager.all_tools.items():
        print('  - tool name: {}'.format(tool_name))
    
    for func in manager.tool_map.values():
        print(func.function)


def test_temp_tool_list():
    """测试CentralizedQwenManager的temp_tool_list方法"""

    # 创建临时目录结构
    temp_dir = tempfile.mkdtemp()
    project_name = "dafau"
    project_dir = Path(temp_dir) / project_name
    project_dir.mkdir(parents=True)

    # 创建模拟的temp_tool_list.py文件
    temp_tool_content = '''
import mcp

@mcp.tool
def add_numbers(a: int, b: int) -> int:
    """
    Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b

@mcp.tool(name="multiply")
def multiply_numbers(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        The product of x and y
    """
    return x * y

def not_a_tool():
    """This function is not decorated with @mcp.tool"""
    pass

@mcp.tool
def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The name to greet

    Returns:
        A greeting message
    """
    return f"Hello, {name}!"
'''

    temp_tool_file = project_dir / "temp_tool_list.py"
    temp_tool_file.write_text(temp_tool_content)

    try:
        # 设置环境变量指向临时目录
        original_rl_factory_root = os.environ.get("RL_FACTORY_ROOT")
        os.environ["RL_FACTORY_ROOT"] = temp_dir

        # 创建manager实例
        env_config = {
            'name': 'test',
            'tool_manager': 'centralized_qwen3',
            'mcp_mode': 'sse',
            'config_path': 'envs/configs/sse_mcp_tools.pydata',
            'enable_thinking': True,
            'max_prompt_length': 2048,
        }

        manager = CentralizedQwenManager(env_config)
        manager.project_name = project_name  # 设置项目名称

        # 调用temp_tool_list方法
        tools = manager.temp_tool_list()

        # 验证结果
        print("测试temp_tool_list方法:")
        print(f"找到 {len(tools)} 个工具")

        expected_tools = ['add_numbers', 'multiply', 'greet']

        for tool_name in expected_tools:
            assert tool_name in tools, f"工具 {tool_name} 应该存在"
            tool_info = tools[tool_name]
            print(f"\n工具: {tool_name}")
            print(f"  描述: {tool_info['description']}")
            print(f"  参数: {tool_info['arguments']}")
            print(f"  返回: {tool_info['returns']}")
            print(f"  代码长度: {len(tool_info['code'])} 字符")

            # 验证必要字段存在
            assert 'description' in tool_info
            assert 'arguments' in tool_info
            assert 'returns' in tool_info
            assert 'code' in tool_info

        # 验证not_a_tool没有被包含
        assert 'not_a_tool' not in tools, "未用@mcp.tool装饰的函数不应该被包含"

        # 测试缓存功能（再次调用应该返回相同结果）
        tools2 = manager.temp_tool_list()
        assert tools == tools2, "缓存功能应该返回相同的结果"

        print("\n✅ temp_tool_list方法测试通过!")

    finally:
        # 清理临时文件和环境变量
        if original_rl_factory_root is not None:
            os.environ["RL_FACTORY_ROOT"] = original_rl_factory_root
        else:
            os.environ.pop("RL_FACTORY_ROOT", None)

        shutil.rmtree(temp_dir)


def test_temp_tool_list_empty():
    """测试temp_tool_list方法在文件不存在时的行为"""

    # 创建临时目录但不创建文件
    temp_dir = tempfile.mkdtemp()
    project_name = "nonexistent_project"

    try:
        # 设置环境变量
        original_rl_factory_root = os.environ.get("RL_FACTORY_ROOT")
        os.environ["RL_FACTORY_ROOT"] = temp_dir

        # 创建manager实例
        env_config = {
            'name': 'test',
            'tool_manager': 'centralized_qwen3',
        }

        manager = CentralizedQwenManager(env_config)
        manager.project_name = project_name

        # 调用方法
        tools = manager.temp_tool_list()

        # 应该返回空字典
        assert tools == {}, f"文件不存在时应该返回空字典，实际返回: {tools}"

        print("✅ temp_tool_list空文件测试通过!")

    finally:
        # 清理
        if original_rl_factory_root is not None:
            os.environ["RL_FACTORY_ROOT"] = original_rl_factory_root
        else:
            os.environ.pop("RL_FACTORY_ROOT", None)

        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    test_manager()
    test_temp_tool_list()
    test_temp_tool_list_empty()
    print("\n🎉 所有测试通过!")
