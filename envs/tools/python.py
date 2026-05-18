# import json
# import ast
# from typing import Tuple
# from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
# from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
# from config import SANDBOX_HOST
# mcp = FastMCP("LocalServer")
# set_endpoint(f"http://{SANDBOX_HOST}:8080")


# def _preprocess_code(code: str) -> str:
#     """Make the last Python statement to be `print` statement"""
#     try:
#         tree = ast.parse(code)
#         if tree.body:
#             last_expr = tree.body[-1]
#             if isinstance(last_expr, ast.Expr):
#                 if not (
#                     isinstance(last_expr.value, ast.Call)
#                     and isinstance(last_expr.value.func, ast.Name)
#                     and last_expr.value.func.id == "print"
#                 ):
#                     print_call = ast.Expr(
#                         value=ast.Call(
#                             func=ast.Name(id="print", ctx=ast.Load()),
#                             args=[last_expr.value],
#                             keywords=[],
#                         )
#                     )
#                     tree.body[-1] = print_call
#                     code = ast.unparse(tree)
#     except:
#         pass
#     return code

# def run_code_sync(code: str, language: str = "python", timeout: float = 15.0) -> Tuple[str, str, bool]:
#     """
#     同步执行代码，通过调用 sandbox_fusion 库而不是本地环境

#     Args:
#         code: 要执行的代码
#         language: 编程语言（目前只支持python）
#         timeout: 超时时间（秒）

#     Returns:
#         Tuple[str, str]: (stdout, stderr)
#     """
#     try:
#         if language != "python":
#             return "", f"Unsupported language: {language}"

#         # 使用 sandbox_fusion 库执行代码
#         r = run_code(RunCodeRequest(code=_preprocess_code(code), language=language))
#         try:
#             data = json.loads(r.json())
#             status = data.get('run_result', {}).get('status', 'unknown')
#             if status == 'Finished':
#                 stdout = data.get('run_result', {}).get('stdout', '')
#                 stderr = data.get('run_result', {}).get('stderr', '')
#             else:
#                 stdout = ''
#                 stderr = data.get('run_result', {}).get('stderr', f'Execution failed with status: {status}')
#         except Exception:
#             # 如果解析失败，尝试其他格式
#             try:
#                 data = json.loads(r.json())
#                 stdout = ''
#                 stderr = data.get('run_result', {}).get('stderr', f'Execution failed: {data.get("status", "unknown")}')
#             except Exception as e:
#                 return "", f"Failed to parse response: {str(e)}"

#         return stdout, stderr

#     except Exception as e:
#         return "", f"Code execution failed: {str(e)}"

# @mcp.tool()
# def execute_python_code(code: str, timeout: float = 15.0):
#     """
#     Python Code Execution Tool (Synchronous Version)

#     Args:
#         code: Python code to execute
#         timeout: Execution timeout in seconds (default: 15.0)

#     Returns:
#         str: Execution result formatted as string
#     """
#     try:
#         print(f"Executing code: {code}")
#         stdout, stderr = run_code_sync(code, "python", timeout)
#         print(f"Code execution result: {stdout}")
#         print(f"Code execution result: {stderr}")
#         result_parts = []

#         if stdout:
#             result_parts.append(f"STDOUT:\n{stdout}")

#         if stderr:
#             result_parts.append(f"STDERR:\n{stderr}")

#         if not result_parts:
#             result_parts.append("Executed successfully (no output), remember to print the result")

#         return "\n\n".join(result_parts)

#     except Exception as e:
#         return f"Tool execution failed: {str(e)}"


# if __name__ == "__main__":
#     print("\nStart MCP service:")
#     mcp.run(transport='stdio')
import json
import ast
import asyncio
import re
from typing import Any
from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
from sandbox_fusion import run_code, RunCodeRequest, set_endpoint
from config import SANDBOX_HOST

mcp = FastMCP("LocalServer")
set_endpoint(f"http://{SANDBOX_HOST}:8080")


def _preprocess_code(code: str) -> str:
    """Make the last Python statement to be `print` statement"""
    try:
        tree = ast.parse(code)
        if tree.body:
            last_expr = tree.body[-1]
            if isinstance(last_expr, ast.Expr):
                if not (
                    isinstance(last_expr.value, ast.Call)
                    and isinstance(last_expr.value.func, ast.Name)
                    and last_expr.value.func.id == "print"
                ):
                    print_call = ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="print", ctx=ast.Load()),
                            args=[last_expr.value],
                            keywords=[],
                        )
                    )
                    tree.body[-1] = print_call
                    code = ast.unparse(tree)
    except:
        pass
    return code


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


def _build_result(
    *,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    reason: str = "",
    error_type: str = "",
    error_message: str = "",
    status: str = "",
) -> dict[str, Any]:
    return {
        "success": success,
        "run_success": success,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "reason": reason,
        "error_type": error_type,
        "error_message": error_message,
        "status": status,
    }


async def run_code_async(code: str, language: str = "python", timeout: float = 15.0) -> dict[str, Any]:
    """
    异步执行代码，通过调用 sandbox_fusion 库而不是本地环境

    Args:
        code: 要执行的代码
        language: 编程语言（目前只支持python）
        timeout: 超时时间（秒）

    Returns:
        dict: execution result with error type metadata
    """
    if language != "python":
        msg = f"Unsupported language: {language}"
        return _build_result(
            success=False,
            reason="invalid_input",
            error_type="invalid_input",
            error_message=msg,
            stderr=msg,
            status="invalid_input",
        )

    try:
        # 使用 asyncio.to_thread 将同步调用转换为异步，并显式做超时控制
        r = await asyncio.wait_for(
            asyncio.to_thread(run_code, RunCodeRequest(code=_preprocess_code(code), language=language)),
            timeout=timeout + 2.0,
        )
    except asyncio.TimeoutError:
        msg = f"Execution exceeded timeout ({timeout}s)"
        return _build_result(
            success=False,
            reason="worker_timeout",
            error_type="worker_timeout",
            error_message=msg,
            stderr=msg,
            status="timeout",
        )
    except Exception as e:
        msg = f"Code execution failed: {str(e)}"
        return _build_result(
            success=False,
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
        return _build_result(
            success=False,
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
            return _build_result(
                success=False,
                stdout=stdout,
                stderr=stderr,
                reason="",
                error_type=error_type,
                error_message=error_message,
                status=status,
            )
        return _build_result(success=True, stdout=stdout, stderr=stderr, status=status)

    reason = _reason_from_status(status)
    error_message = stderr.strip() or f"Execution failed with status: {status}"
    return _build_result(
        success=False,
        stdout=stdout,
        stderr=stderr,
        reason=reason,
        error_type=reason,
        error_message=error_message,
        status=status,
    )


@mcp.tool()
async def execute_python_code(code: str, timeout: float = 15.0):
    """
    Python Code Execution Tool (Asynchronous Version)

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 15.0)

    Returns:
        str: Execution result formatted as string.
             Success starts with "Tool call success";
             failure starts with "<error_type>: <error_message>".
    """
    try:
        print(f"Executing code: {code}")
        result = await run_code_async(code, "python", timeout)
        stdout = result["stdout"]
        stderr = result["stderr"]
        print(f"Code execution result: {stdout}")
        print(f"Code execution result: {stderr}")
        result_parts = []

        if result["success"]:
            result_parts.append("Tool call success")
        else:
            error_type = result.get("error_type") or "runtime_error"
            error_message = result.get("error_message") or "Unknown execution error"
            result_parts.append(f"{error_type}: {error_message}")

        if result.get("reason"):
            result_parts.append(f"REASON: {result['reason']}")

        if stdout:
            result_parts.append(f"STDOUT:\n{stdout}")

        if stderr:
            result_parts.append(f"STDERR:\n{stderr}")

        if not result_parts:
            result_parts.append("Executed successfully (no output), remember to print the result")

        return "\n\n".join(result_parts)

    except Exception as e:
        return f"internal_error: Tool execution failed: {str(e)}"


if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')
