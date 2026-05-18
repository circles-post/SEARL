import unittest
from pathlib import Path

TARGET_FILE = (
    Path(__file__).resolve().parents[2]
    / "envs"
    / "tool_manager"
    / "centralized"
    / "centralized_qwen3_manager.py"
)
FORBIDDEN_SNIPPET = (
    "{'role': USER, 'content': "
    "\"Now begin executing the plan step by step, starting with ST_1.\"}"
)


class TestPlanCallPrompt(unittest.TestCase):
    def test_plan_call_prompt_does_not_inject_extra_st1_user_instruction(self):
        content = TARGET_FILE.read_text(encoding="utf-8")
        self.assertNotIn(FORBIDDEN_SNIPPET, content)


if __name__ == "__main__":
    unittest.main()
