#!/usr/bin/env python3
"""
Quick smoke test to verify update_mcp_box functionality can write
to the project's official_tool_list.py without needing the full runtime stack.

This test implements the core logic of CentralizedToolActor.update_mcp_box
in a standalone function to avoid Ray actor complications.

Run from repo root:
    python scripts/test_update_mcp_box.py
"""

import json
import sys
import traceback
import math
import numpy as np
import re
import os
import datetime
from collections import Counter, defaultdict, deque
from mcp.server.fastmcp import FastMCP
from pathlib import Path


def test_update_mcp_box(project_name: str, mcps: dict, similarity_threshold: float = 0.88):
    """
    Standalone implementation of update_mcp_box logic for testing.
    This avoids Ray actor complications while testing the core functionality.
    """
    print("Agent Log: Starting standalone update_mcp_box test...")

    # Mock clustering - assume no clusters found for simplicity in test
    print("Agent Log: Performing mock clustering (no actual computation)...")
    clusters = {i: [i] for i in range(len(mcps))}  # Each MCP in its own cluster
    print("Agent Log: Clustering complete.")

    # Build the official_tool_list.py content
    try:
        root_dir = os.environ.get("RL_FACTORY_ROOT", Path(__file__).resolve().parents[1])
        tool_dir = Path(root_dir) / "envs" / "tools" / "invented_tools" / project_name
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

        def build_tool_block(name: str, description: str, arguments: str, returns: str, code: str) -> str:
            """参考 mcp_creation.py 的落盘格式，生成带 @mcp.tool 的函数块。"""
            def doc_lines(base_indent: str = "    "):
                return "\n".join(
                    [
                        f'{base_indent}"""{description}',
                        f"{base_indent}",
                        f"{base_indent}Args:",
                        f"{base_indent}    {arguments}",
                        f"{base_indent}",
                        f"{base_indent}Returns:",
                        f"{base_indent}    {returns}",
                        f'{base_indent}"""',
                    ]
                )

            if code.lstrip().startswith("def "):
                code_lines = code.splitlines()
                doc_start = doc_end = -1
                for idx, ln in enumerate(code_lines[1:], start=1):
                    if '"""' in ln or "'''" in ln:
                        doc_start = idx
                        quote = '"""' if '"""' in ln else "'''"
                        for j in range(idx + 1, len(code_lines)):
                            if quote in code_lines[j]:
                                doc_end = j
                                break
                        break

                if doc_start != -1 and doc_end != -1 and doc_end > doc_start:
                    inner = "\n".join(code_lines[doc_start:doc_end + 1])
                    inner_content = inner.strip().strip('"\'').strip()
                    base_indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                    new_doc = [
                        f'{base_indent}"""{description}',
                        f"{base_indent}{inner_content}" if inner_content else f"{base_indent}A tool function.",
                        f"{base_indent}Args:",
                        f"{base_indent}    {arguments}",
                        f"{base_indent}",
                        f"{base_indent}Returns:",
                        f"{base_indent}    {returns}",
                        f'{base_indent}"""',
                    ]
                    code_lines = code_lines[:doc_start] + new_doc + code_lines[doc_end + 1 :]
                else:
                    indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                    insertion = doc_lines(indent)
                    code_lines = [code_lines[0], insertion, *code_lines[1:]]

                merged = "\n".join(code_lines)
                return f"\n\n@mcp.tool(name='{name}')\n{merged.lstrip()}\n"

            indented_body = "\n".join(
                f"        {line}" if line.strip() else "        "
                for line in (code.splitlines() if code else ["return None"])
            )
            wrapped_body = (
                "    try:\n"
                f"{indented_body}\n"
                "    except Exception as e:\n"
                "        return f\"tool execution failed: {str(e)}\"\n"
            )
            return (
                f"\n\n@mcp.tool(name='{name}')\n"
                f"def {name}(**kwargs):\n"
                f"{doc_lines()}\n"
                f"{wrapped_body}\n"
            )

        header = f"{common_imports}mcp = FastMCP('Official_Server')\n\n"
        blocks = []
        for mcp_name, mcp_data in mcps.items():
            desc = mcp_data.get('description') or ""
            args = mcp_data.get('arguments') or ""
            rets = mcp_data.get('returns') or ""
            code = mcp_data.get('code') or ""

            # 验证函数名是否为有效的Python标识符
            if not mcp_name.isidentifier():
                print(f"Warning: Invalid function name '{mcp_name}', skipping...")
                continue

            blocks.append(build_tool_block(mcp_name, desc, args, rets, code))

        content = header + "".join(blocks) + main_block

        # 验证生成的代码是否为有效的Python语法
        try:
            compile(content, str(official_file), 'exec')
            official_file.write_text(content, encoding="utf-8")
            print(f"Agent Log: Merged MCPs saved to {official_file}")
            return len(mcps)
        except SyntaxError as e:
            print(f"Agent Log: Generated code has syntax error: {e}")
            print(f"Agent Log: Content preview: {content[:500]}...")
            raise
    except Exception as e:
        print(f"Agent Log: Failed to write merged MCPs to official_tool_list.py: {e}")
        return 0


def main():
    # Ensure the method writes to the intended project directory
    rl_root = Path(os.environ.get("RL_FACTORY_ROOT", Path(__file__).resolve().parents[1]))
    os.environ["RL_FACTORY_ROOT"] = str(rl_root)

    project_name = "grpo_baseline_1gpu_reasoning_mcp_debug"
    official_path = (
        rl_root
        / "envs"
        / "tools"
        / "invented_tools"
        / project_name
        / "official_tool_list.py"
    )

    print(f"Will write to: {official_path}")

    # Prepare test MCP data
    test_mcps = {
        "add_one": {
            "description": "Adds 1 to the provided integer.",
            "arguments": "x (int): number to increment",
            "returns": "int: incremented result",
            "code": "def add_one(x: int) -> int:\n    return x + 1",
        },
        "echo_text": {
            "description": "Echoes back the provided text.",
            "arguments": "text (str): text to echo",
            "returns": "str: echoed text",
            "code": "def echo_text(text: str) -> str:\n    return text",
        },
    }

    try:
        result = test_update_mcp_box(project_name, test_mcps, similarity_threshold=0.5)
        print(f"test_update_mcp_box completed successfully, returned: {result}")
    except Exception as e:
        print(f"test_update_mcp_box raised an exception: {e}")
        traceback.print_exc()

    # Show the first 400 chars of the generated file
    try:
        if official_path.exists():
            content = official_path.read_text()
            print(f"\nGenerated file preview (first 400 chars):\n{content[:400]}...")
        else:
            print(f"File was not created: {official_path}")
    except Exception as e:
        print(f"Error reading generated file: {e}")


if __name__ == "__main__":
    main()
