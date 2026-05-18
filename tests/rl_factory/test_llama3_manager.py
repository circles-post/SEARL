from envs.tool_manager.qwen3_manager import QwenManager


def test_manager():
    env_config = {
        'name': 'base',
        'tool_manager': 'llama3',
        'mcp_mode': 'sse',
        'config_path': 'envs/configs/sse_mcp_tools.pydata',
        'max_prompt_length': 2048,
    }
    manager = QwenManager(env_config)
    print('Tools:')
    for tool_name, tool in manager.all_tools.items():
        print('  - tool name: {}'.format(tool_name))
    
    for func in manager.tool_map.values():
        print(func.function)


if __name__ == '__main__':
    test_manager()
