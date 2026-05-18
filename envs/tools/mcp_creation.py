import json
import re
import traceback
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
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


def _extract_error_type(text: str) -> str:
    if not text:
        return "runtime_error"
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*Error)\s*:", text)
    if match:
        return match.group(1)
    return "runtime_error"


def _reason_from_status(status: str) -> str:
    lower_status = (status or "").lower()
    if "queue" in lower_status and "timeout" in lower_status:
        return "queue_timeout"
    if "timeout" in lower_status:
        return "worker_timeout"
    if "invalid" in lower_status:
        return "invalid_input"
    return "internal_error"


def _build_exec_result(
    *,
    run_success: bool,
    stdout: str = "",
    stderr: str = "",
    reason: str = "",
    error_type: str = "",
    error_message: str = "",
    status: str = "",
) -> dict[str, Any]:
    final_answer = "FINAL ANSWER" in (stdout or "").upper() or "FINAL ANSWER" in (stderr or "").upper()
    return {
        "stdout": stdout or "",
        "stderr": stderr or "",
        "final_answer": final_answer,
        "run_success": run_success,
        "reason": reason,
        "error_type": error_type,
        "error_message": error_message,
        "status": status,
    }


def run_code_sync(code: str, language: str = "python", timeout: float = 15.0) -> dict[str, Any]:
    """
    同步执行代码，通过调用 sandbox_fusion 库而不是本地环境

    Args:
        code: 要执行的代码
        language: 编程语言（目前只支持python）
        timeout: 超时时间（秒）

    Returns:
        dict: execution result with reason/error_type metadata
    """
    if language != "python":
        msg = f"Unsupported language: {language}"
        return _build_exec_result(
            run_success=False,
            reason="invalid_input",
            error_type="invalid_input",
            error_message=msg,
            stderr=msg,
            status="invalid_input",
        )

    try:
        r = run_code(RunCodeRequest(code=code, language=language))
    except TimeoutError:
        msg = f"Execution exceeded timeout ({timeout}s)"
        return _build_exec_result(
            run_success=False,
            reason="worker_timeout",
            error_type="worker_timeout",
            error_message=msg,
            stderr=msg,
            status="timeout",
        )
    except Exception as e:
        msg = f"Code execution failed: {str(e)}"
        return _build_exec_result(
            run_success=False,
            reason="internal_error",
            error_type="internal_error",
            error_message=msg,
            stderr=msg,
            status="internal_error",
        )

    try:
        data = json.loads(r.json())
    except Exception as e:
        msg = f"Failed to parse sandbox response: {str(e)}"
        return _build_exec_result(
            run_success=False,
            reason="internal_error",
            error_type="internal_error",
            error_message=msg,
            stderr=msg,
            status="parse_error",
        )

    run_result = data.get("run_result", {}) or {}
    status = str(run_result.get("status") or data.get("status") or "unknown")
    stdout = run_result.get("stdout", "") or ""
    stderr = run_result.get("stderr", "") or ""

    if status == "Finished":
        if stderr.strip():
            error_type = _extract_error_type(stderr)
            error_message = stderr.strip().splitlines()[-1]
            return _build_exec_result(
                run_success=False,
                stdout=stdout,
                stderr=stderr,
                reason="",
                error_type=error_type,
                error_message=error_message,
                status=status,
            )
        return _build_exec_result(run_success=True, stdout=stdout, stderr=stderr, status=status)

    reason = _reason_from_status(status)
    error_message = stderr.strip() or f"Execution failed with status: {status}"
    return _build_exec_result(
        run_success=False,
        stdout=stdout,
        stderr=stderr,
        reason=reason,
        error_type=reason,
        error_message=error_message,
        status=status,
    )

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
                 "run_success": bool,
                 "execution_result": any,
                 "stdout": str,
                 "stderr": str,
                 "reason": str,
                 "error_type": str,
                 "error_message": str,
                 "error": str (optional)
             }
    """
    try:
        print(f"Creating and executing MCP: {name}...")
        # 动态解析当前项目相关路径，避免模块导入时锁死 default_project
        INVENTED_DIR, INVENTED_FILE, OFFICIAL_FILE = _get_invented_paths()
        mcp_block = {
            "name": name,
            "description": description,
            "arguments": arguments,
            "returns": returns,
            "code": code,
            "inputs": inputs,
            "timeout": timeout
        }
        # 1. 准备依赖导入
        imports = []
        
        import_block = "\n".join(imports)

        # 2. 构建执行脚本
        # 我们将输入参数序列化嵌入到脚本中，以避免转义问题
        inputs_json = json.dumps(inputs)
        
        execution_script = f"""
import cmath
from fractions import Fraction
import numpy as np
import math
from math import *
import traceback
import json
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
        exec_result = run_code_sync(execution_script, "python", timeout)
        stdout = exec_result["stdout"]
        stderr = exec_result["stderr"]
        
        # 4. 解析结果
        creation_success = "__MCP_CREATION_SUCCESS__" in stdout
        execution_result = None
        error_msg = exec_result.get("error_message")
        error_type = exec_result.get("error_type", "")

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
            if stderr and not error_msg:
                error_msg = stderr
            else:
                error_match = re.search(r"__MCP_EXECUTION_ERROR__(.*)", stdout)
                if error_match:
                    error_msg = error_match.group(1).strip()
                    if not error_type:
                        error_type = _extract_error_type(error_msg)
                elif not error_msg:
                    error_msg = "Unknown execution error"
            if not error_type:
                error_type = _extract_error_type(error_msg or "")
        
        response = {
            "creation_success": creation_success,
            "run_success": bool(exec_result.get("run_success", False)) and creation_success,
            "execution_result": execution_result,
            "stdout": stdout.replace("__MCP_CREATION_SUCCESS__", "").replace(f"__MCP_RESULT_START__{json.dumps(execution_result) if execution_result is not None else ''}__MCP_RESULT_END__", "").strip(),
            "stderr": stderr,
            "reason": exec_result.get("reason", ""),
            "error_type": error_type or "",
            "error_message": error_msg or "",
        }
        
        if error_msg:
            response["error"] = error_msg

        return json.dumps(response, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "creation_success": False,
            "run_success": False,
            "reason": "internal_error",
            "error_type": "internal_error",
            "error_message": f"Tool internal error: {str(e)}",
            "error": f"Tool internal error: {str(e)}",
            "stdout": "",
            "stderr": "",
            "execution_result": None,
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
