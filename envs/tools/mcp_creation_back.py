import json
import re
import traceback
from typing import Tuple, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from config import SANDBOX_HOST
from pathlib import Path
import os
mcp = FastMCP("LocalServer")
set_endpoint(f"http://{SANDBOX_HOST}:8080")


def _get_project_name() -> str:
    """环境变量优先获取项目名，默认 fallback 到 default_project。"""
    return os.environ.get("PROJECT_NAME") or "grpo_baseline_4gpu_reasoning_mcp"


BASE_DIR = Path(__file__).parent


def _get_invented_paths():
    """每次调用时动态计算目录/文件，避免导入时锁死 default_project。"""
    project_name = _get_project_name()
    invented_dir = BASE_DIR / "invented_tools" / project_name
    invented_file = invented_dir / "temp_tool_list.py"
    official_file = invented_dir / "official_tool_list.py"
    return invented_dir, invented_file, official_file

def run_code_sync(code: str, language: str = "python", timeout: float = 15.0) -> Tuple[str, str, bool]:
    """
    同步执行代码，通过调用 sandbox_fusion 库而不是本地环境

    Args:
        code: 要执行的代码
        language: 编程语言（目前只支持python）
        timeout: 超时时间（秒）

    Returns:
        Tuple[str, str, bool]: (stdout, stderr, final_answer)
    """
    try:
        if language != "python":
            return "", f"Unsupported language: {language}", False

        # 使用 sandbox_fusion 库执行代码
        r = run_code(RunCodeRequest(code=code, language=language))
        try:
            data = json.loads(r.json())
            status = data.get('run_result', {}).get('status', 'unknown')
            if status == 'Finished':
                stdout = data.get('run_result', {}).get('stdout', '')
                stderr = data.get('run_result', {}).get('stderr', '')
            else:
                stdout = ''
                stderr = data.get('run_result', {}).get('stderr', f'Execution failed with status: {status}')
        except Exception:
            # 如果解析失败，尝试其他格式
            try:
                data = json.loads(r.json())
                stdout = ''
                stderr = data.get('run_result', {}).get('stderr', f'Execution failed: {data.get("status", "unknown")}')
            except Exception as e:
                return "", f"Failed to parse response: {str(e)}", False

        # 检查是否是最终答案（简单的启发式判断）
        final_answer = "FINAL ANSWER" in stdout.upper() or "FINAL ANSWER" in stderr.upper()

        return stdout, stderr, final_answer

    except Exception as e:
        return "", f"Code execution failed: {str(e)}", False

@mcp.tool()
def create_and_execute_mcp(
    name: str,
    description: str,
    arguments: str,
    returns: str,
    code: str,
    inputs: Dict[str, Any],
    timeout: float = 15.0
) -> str:
    """
    MCP Creation and Execution Tool

    Create and immediately execute an MCP tool function.

    Args:
        name: MCP tool name (Function name)
        description: Tool description
        arguments: Argument description string (e.g., "a, b (int)")
        returns: Return value description
        code: Complete Python function implementation code
        inputs: Input arguments dictionary required for this function call (e.g., {"a": 1, "b": 2})
        timeout: Execution timeout in seconds (default: 15.0)

    Returns:
        str: JSON formatted string containing creation status and execution result
             {
                 "creation_success": bool,
                 "execution_result": any,
                 "stdout": str,
                 "stderr": str,
                 "error": str (optional)
             }
    """
    try:
        print(f"Creating and executing MCP: {name}...")
        # 动态解析当前项目相关路径，避免模块导入时锁死 default_project
        INVENTED_DIR, INVENTED_FILE, OFFICIAL_FILE = _get_invented_paths()
        
        # 1. 准备依赖导入
        imports = []
        
        import_block = "\n".join(imports)

        # 2. 构建执行脚本
        # 我们将输入参数序列化嵌入到脚本中，以避免转义问题
        inputs_json = json.dumps(inputs)
        
        execution_script = f"""
import json
import sys
import traceback
import math
from math import *  
import sympy
import scipy as sp
import cmath
from fractions import Fraction
import numpy as np
import re
from itertools import *
import os
import json 
import datetime
import re
from collections import Counter, defaultdict, deque
# Imports
{import_block}

# User Defined Function
{code}

if __name__ == "__main__":
    try:
        # Load inputs
        inputs = json.loads({repr(inputs_json)})
        
        # Verify function exists
        if '{name}' not in globals():
            raise NameError(f"Function '{name}' not found in code definition.")
        
        target_func = globals()['{name}']
        
        # Execute function
        # 假设 inputs 是字典，对应关键字参数
        if isinstance(inputs, dict):
            result = target_func(**inputs)
        else:
            # 假如 inputs 不是字典，尝试作为单个参数传递 (虽然类型提示是Dict)
            result = target_func(inputs)
            
        # Print result in a marked format for extraction
        print(f"__MCP_RESULT_START__{{json.dumps(result)}}__MCP_RESULT_END__")
        print("__MCP_CREATION_SUCCESS__")
        
    except Exception as e:
        traceback.print_exc()
        print(f"__MCP_EXECUTION_ERROR__{{str(e)}}")
"""

        # 3. 执行代码
        stdout, stderr, _ = run_code_sync(execution_script, "python", timeout)
        
        # 4. 解析结果
        creation_success = "__MCP_CREATION_SUCCESS__" in stdout
        execution_result = None
        error_msg = None

        # 提取结果 JSON
        result_match = re.search(r"__MCP_RESULT_START__(.*?)__MCP_RESULT_END__", stdout, re.DOTALL)
        if result_match:
            try:
                execution_result = json.loads(result_match.group(1))
            except:
                execution_result = result_match.group(1) # Fallback to raw string
        
        # 提取错误信息
        if not creation_success:
            # 优先查看 stderr，其次查看 stdout 中的 error marker
            if stderr:
                error_msg = stderr
            else:
                error_match = re.search(r"__MCP_EXECUTION_ERROR__(.*)", stdout)
                if error_match:
                    error_msg = error_match.group(1)
                else:
                    error_msg = "Unknown execution error"
        
        # 5. 构建返回
        response = {
            "creation_success": creation_success,
            "execution_result": execution_result,
            "stdout": stdout.replace("__MCP_CREATION_SUCCESS__", "").replace(f"__MCP_RESULT_START__{json.dumps(execution_result) if execution_result is not None else ''}__MCP_RESULT_END__", "").strip(),
            "stderr": stderr
        }
        
        if error_msg:
            response["error"] = error_msg

        if creation_success and error_msg is None:
            try:
                INVENTED_DIR.mkdir(parents=True, exist_ok=True)
                
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
                
                # 如果文件不存在或为空，初始化
                if not INVENTED_FILE.exists() or INVENTED_FILE.stat().st_size == 0:
                    INVENTED_FILE.write_text(
                        f"{common_imports}"
                        "mcp = FastMCP('InventedServer')\n\n",
                        encoding="utf-8"
                    )
                if not OFFICIAL_FILE.exists() or OFFICIAL_FILE.stat().st_size == 0:
                    OFFICIAL_FILE.write_text(
                        f"{common_imports}"
                        "mcp = FastMCP('Official_Server')\n\n",
                        encoding="utf-8"
                    )
                # 构造持久化的函数定义：
                # 如果传入 code 已包含函数定义，则前面加装饰器；否则用 inputs 的 key 构造形参并缩进包裹。
                def build_tool_block():
                    sig_args = list(inputs.keys()) if isinstance(inputs, dict) else []
                    def doc_lines(base_indent: str = "    "):
                        # 兼容 temp.py 示例风格：描述 + 返回摘要置于 Args 前
                        return "\n".join(
                            [
                                f'{base_indent}"""{description}',
                                f"{base_indent}{returns}",
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

                        # 查找现有 docstring 边界
                        doc_start = doc_end = -1
                        for idx, ln in enumerate(code_lines[1:], start=1):
                            if '"""' in ln or "'''" in ln:
                                doc_start = idx
                                quote = '"""' if '"""' in ln else "'''"
                                # 往后找到结束
                                for j in range(idx, len(code_lines)):
                                    if quote in code_lines[j] and j != idx:
                                        doc_end = j
                                        break
                                break

                        if doc_start != -1 and doc_end != -1:
                            # 提取原 docstring 内容
                            inner = "\n".join(code_lines[doc_start:doc_end + 1])
                            inner_content = inner.strip().strip('"\'')
                            # 构造新的 docstring：描述 + 原文 + Args/Returns
                            base_indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                            new_doc = [
                                f'{base_indent}"""{description}',
                                f"{base_indent}{inner_content}",
                                f"{base_indent}Args:",
                                f"{base_indent}    {arguments}",
                                f"{base_indent}",
                                f"{base_indent}Returns:",
                                f"{base_indent}    {returns}",
                                f'{base_indent}"""',
                            ]
                            code_lines = code_lines[:doc_start] + new_doc + code_lines[doc_end + 1 :]
                        else:
                            # 无 docstring，插入标准 docstring
                            indent = re.match(r"\s*", code_lines[0]).group(0) + "    "
                            insertion = doc_lines(indent)
                            code_lines = [code_lines[0], insertion, *code_lines[1:]]

                        merged = "\n".join(code_lines)
                        # 避免装饰器与 def 之间出现空行
                        return f"\n\n@mcp.tool(name='{name}')\n{merged.lstrip()}\n"

                    # 否则包裹为函数体
                    if sig_args:
                        signature = ", ".join(sig_args)
                    else:
                        signature = "**kwargs"
                    indented_body = "\n".join(f"        {line}" if line.strip() else "        " for line in code.splitlines())
                    wrapped_body = (
                        "    try:\n"
                        f"{indented_body}\n"
                        "    except Exception as e:\n"
                        "        return f\"tool execution failed: {str(e)}\"\n"
                    )
                    return (
                        f"\n\n@mcp.tool(name='{name}')\n"
                        f"def {name}({signature}):\n"
                        f"{doc_lines()}\n"
                        f"{wrapped_body}\n"
                    )

                tool_block = build_tool_block()
                
                # 读取现有内容
                content = ""
                official_content = ""
                if INVENTED_FILE.exists():
                    content = INVENTED_FILE.read_text(encoding="utf-8")
                
                # 移除旧的 __main__ 块（如果存在）
                main_block_start = "\n\nif __name__ == \"__main__\":"
                if main_block_start in content:
                    content = content.split(main_block_start)[0]
                
                # 确保头部导入和初始化（如果是新文件或内容为空，或者之前版本没有这些）
                # 简单检查 FastMCP 是否存在，如果不存在（比如是旧文件），重写头部
                header = f"{common_imports}mcp = FastMCP('InventedServer')\n\n"
                
                if not content.strip():
                    content = header
                elif "FastMCP" not in content:
                    content = header + content

                # 追加新工具定义
                content += tool_block

                # 追加 __main__ 块（保持文件末尾只出现一次）
                main_block = (
                    "\n\nif __name__ == \"__main__\":\n"
                    "    print(\"\\nStart MCP service:\")\n"
                    "    mcp.run(transport='stdio')\n"
                )
                if main_block not in content:
                    content += main_block
                if main_block not in official_content:
                    official_content += main_block
                # 写回文件（去重简单防护：避免重复同名函数块）
                INVENTED_FILE.write_text(content, encoding="utf-8")
                OFFICIAL_FILE.write_text(official_content, encoding="utf-8")
            except Exception as e:
                # 不影响原有执行逻辑，只在 response 中附带落盘异常
                response.setdefault("persist_errors", [])
                response["persist_errors"].append(f"write tool to {INVENTED_FILE} failed: {e}")

        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "creation_success": False,
            "error": f"Tool internal error: {str(e)}"
        }, ensure_ascii=False)


def test_mcp_creation():
    """测试 MCP 创建与失败回滚，不污染 temp_tool_list.py"""
    import json
    import uuid
    from pathlib import Path

    INVENTED_DIR, INVENTED_FILE, _ = _get_invented_paths()
    # 备份原文件
    backup = INVENTED_FILE.read_text(encoding="utf-8") if INVENTED_FILE.exists() else ""

    ok_name = f"test_ok_{uuid.uuid4().hex[:8]}"
    bad_name = f"test_bad_{uuid.uuid4().hex[:8]}"

    try:
        # 成功用例：简单加法
        # 函数名必须与 name 一致，避免 “Function not found”
        ok_code = f"def {ok_name}(a:int,b:int):\n    return a + b\n"
        resp_ok = json.loads(
            create_and_execute_mcp(
                name=ok_name,
                description="add two numbers",
                arguments="a,b",
                returns="sum",
                code=ok_code,
                inputs={"a": 1, "b": 2},
                timeout=5.0,
            )
        )
        assert resp_ok["creation_success"] is True, resp_ok
        assert resp_ok["execution_result"] == 3, resp_ok
        # 确认成功函数已写入
        content = INVENTED_FILE.read_text(encoding="utf-8")
        assert ok_name in content, "成功创建的工具未写入文件"

        # 失败用例：缺参数，creation_success 应为 False，且不应写入文件
        bad_code = "def broken(x):\n    raise RuntimeError('boom')\n"
        resp_bad = json.loads(
            create_and_execute_mcp(
                name=bad_name,
                description="bad tool",
                arguments="x",
                returns="y",
                code=bad_code,
                inputs={},  # 缺少必需参数
                timeout=5.0,
            )
        )
        assert resp_bad["creation_success"] is False, "失败用例不应成功"
        content_after = INVENTED_FILE.read_text(encoding="utf-8")
        assert bad_name not in content_after, "失败工具不应写入文件"

        print("✅ test_mcp_creation passed")
        return True
    finally:
        # 还原文件，避免污染
        INVENTED_FILE.write_text(backup, encoding="utf-8")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mcp_creation()
    else:
        print("\nStart MCP service:")
        mcp.run(transport='stdio')
