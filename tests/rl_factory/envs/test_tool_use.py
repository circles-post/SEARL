from transformers import AutoTokenizer
from envs.base import Env
from omegaconf import OmegaConf
import os


def test():
    config = OmegaConf.create({
        'config_path': 'envs/configs/calculator.json',
        'step_token': '\n',
        'tool_manager': 'qwen3'  # 指定使用 qwen3 工具管理器，避免自适应模式
    })
    env = Env(config)
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TEST_TOKENIZER_PATH", "Qwen/Qwen3-8B"))

    response_action = """
Hello!
<actions>
    <action>
        <args>
            {"expression": "1+1"}
        </args>
    </action>
    <action>
        <name>calculator</name>
        <args>
            {"expressions": "1+2"}
        </args>
    </action>
</actions>
"""
    response_answer = """
Hello!
<answer>
2
</answer>
"""
    env.step([response_action, response_answer], tokenizer)

if __name__ == '__main__':
    test()
