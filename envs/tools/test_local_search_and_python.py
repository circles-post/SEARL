#!/usr/bin/env python3
"""
测试 local_search.py、python.py 和 mcp_creation.py 的脚本
用于验证这三个工具是否能够正常运行
"""

import asyncio
import sys
import json
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def test_local_search():
    """测试 local_search.py 的 search 函数"""
    print("=" * 60)
    print("测试 local_search.py")
    print("=" * 60)

    try:
        from local_search import search

        # 测试 1: 基本搜索
        print("\n[测试 1] 基本搜索 - 查询 'Python programming'")
        result = search("Python programming is what", topk=3)
        print(f"结果类型: {type(result)}")
        print(f"结果长度: {len(result) if isinstance(result, str) else 'N/A'}")
        if isinstance(result, str):
            preview = result[:300] if len(result) > 300 else result
            print(f"结果预览:\n{preview}")
        else:
            print(f"结果: {result}")

        # 测试 2: 不同的 topk 值
        print("\n[测试 2] 搜索 - topk=1")
        result = search("machine learning", topk=1)
        if isinstance(result, str):
            preview = result[:200] if len(result) > 200 else result
            print(f"结果预览:\n{preview}")
        else:
            print(f"结果: {result}")

        # 测试 3: 空查询
        print("\n[测试 3] 空查询测试")
        result = search("", topk=3)
        print(f"结果: {result}")

        print("\n✓ local_search.py 测试完成")
        return True

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装所有依赖 (requests, mcp.server.fastmcp, config)")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


async def test_python_execution():
    """测试 python.py 的 execute_python_code 函数"""
    print("\n" + "=" * 60)
    print("测试 python.py")
    print("=" * 60)

    try:
        from python import execute_python_code

        # 测试 1: 简单算术
        print("\n[测试 1] 简单算术: 2 + 2")
        code = "result = 2 + 2\nprint(result)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        # 测试 2: 字符串操作
        print("\n[测试 2] 字符串操作")
        code = "text = 'Hello, World!'\nprint(text.upper())"
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        # 测试 3: 列表操作
        print("\n[测试 3] 列表操作")
        code = "numbers = [1, 2, 3, 4, 5]\nprint(sum(numbers))"
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        # 测试 4: 错误处理
        print("\n[测试 4] 错误处理 - 除以零")
        code = "result = 1 / 0"
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        # 测试 5: 导入测试
        print("\n[测试 5] 导入测试 - 使用 math 模块")
        code = "import math\nprint(math.pi)"
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        # 测试 6: 多行代码
        print("\n[测试 6] 多行代码测试")
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(10)]
print(result)
"""
        result = await execute_python_code(code, timeout=15.0)
        print(f"结果:\n{result}")

        print("\n✓ python.py 测试完成")
        return True

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装所有依赖 (mcp.server.fastmcp, sandbox_fusion, config)")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """测试是否能导入所有必需的模块"""
    print("=" * 60)
    print("测试依赖导入")
    print("=" * 60)

    modules_to_test = [
        ("requests", "local_search.py 依赖"),
        ("mcp.server.fastmcp", "MCP 框架"),
        ("sandbox_fusion", "python.py 依赖"),
        ("config", "配置模块"),
        ("ast", "Python 标准库"),
        ("json", "Python 标准库"),
    ]

    results = {}
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name:30s} - {description}")
            results[module_name] = True
        except ImportError as e:
            print(f"✗ {module_name:30s} - {description} (错误: {e})")
            results[module_name] = False

    return all(results.values())


def test_mcp_creation():
    """测试 mcp_creation.py 的 create_and_execute_mcp 函数"""
    print("\n" + "=" * 60)
    print("测试 mcp_creation.py")
    print("=" * 60)

    try:
        from mcp_creation import create_and_execute_mcp

        # 测试 1: 简单的加法函数
        print("\n[测试 1] 创建并执行简单加法函数")
        result = create_and_execute_mcp(
            name="add_numbers",
            description="Add two numbers together",
            arguments="a, b (int)",
            returns="int: sum of a and b",
            code="def add_numbers(a: int, b: int):\n    return a + b",
            inputs={"a": 5, "b": 3},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"创建成功: {result_dict.get('creation_success')}")
        print(f"执行结果: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"错误信息: {result_dict.get('error')}")

        success_1 = result_dict.get('creation_success') and result_dict.get('execution_result') == 8

        # 测试 2: 使用数学库的函数
        print("\n[测试 2] 创建并执行使用 math 库的函数")
        result = create_and_execute_mcp(
            name="calculate_circle_area",
            description="Calculate the area of a circle",
            arguments="radius (float)",
            returns="float: area of the circle",
            code="def calculate_circle_area(radius: float):\n    import math\n    return math.pi * radius ** 2",
            inputs={"radius": 5.0},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"创建成功: {result_dict.get('creation_success')}")
        print(f"执行结果: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"错误信息: {result_dict.get('error')}")

        success_2 = result_dict.get('creation_success') and result_dict.get('execution_result') is not None

        # 测试 3: 列表处理函数
        print("\n[测试 3] 创建并执行列表处理函数")
        result = create_and_execute_mcp(
            name="sum_list",
            description="Sum all numbers in a list",
            arguments="numbers (list)",
            returns="int: sum of all numbers",
            code="def sum_list(numbers: list):\n    return sum(numbers)",
            inputs={"numbers": [1, 2, 3, 4, 5]},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"创建成功: {result_dict.get('creation_success')}")
        print(f"执行结果: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"错误信息: {result_dict.get('error')}")

        success_3 = result_dict.get('creation_success') and result_dict.get('execution_result') == 15

        # 测试 4: 错误处理 - 故意制造错误
        print("\n[测试 4] 错误处理测试 - 除以零")
        result = create_and_execute_mcp(
            name="divide_by_zero",
            description="Test error handling",
            arguments="a (int)",
            returns="int",
            code="def divide_by_zero(a: int):\n    return a / 0",
            inputs={"a": 10},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"创建成功: {result_dict.get('creation_success')}")
        print(f"执行结果: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"错误信息: {result_dict.get('error')[:100]}...")  # 只显示前100个字符

        # 这个测试应该失败，所以我们检查 creation_success 是否为 False
        success_4 = not result_dict.get('creation_success')

        # 测试 5: 复杂函数 - 斐波那契数列
        print("\n[测试 5] 创建并执行斐波那契函数")
        result = create_and_execute_mcp(
            name="fibonacci",
            description="Calculate nth Fibonacci number",
            arguments="n (int)",
            returns="int: nth Fibonacci number",
            code="""def fibonacci(n: int):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
            inputs={"n": 10},
            timeout=10.0
        )

        result_dict = json.loads(result)
        print(f"创建成功: {result_dict.get('creation_success')}")
        print(f"执行结果: {result_dict.get('execution_result')}")
        if result_dict.get('error'):
            print(f"错误信息: {result_dict.get('error')}")

        success_5 = result_dict.get('creation_success') and result_dict.get('execution_result') == 55

        print("\n✓ mcp_creation.py 测试完成")

        # 返回所有测试是否都通过
        all_success = success_1 and success_2 and success_3 and success_4 and success_5
        if all_success:
            print("✓ 所有 mcp_creation 测试通过")
        else:
            print(f"✗ 部分测试失败 (成功: {sum([success_1, success_2, success_3, success_4, success_5])}/5)")

        return all_success

    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        print("请确保已安装所有依赖 (mcp.server.fastmcp, sandbox_fusion, config)")
        return False
    except Exception as e:
        print(f"✗ 测试过程中出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def check_config():
    """检查配置文件"""
    print("\n" + "=" * 60)
    print("检查配置")
    print("=" * 60)

    try:
        from config import LOCAL_SEARCH_HOST, SANDBOX_HOST
        print(f"✓ LOCAL_SEARCH_HOST: {LOCAL_SEARCH_HOST}")
        print(f"✓ SANDBOX_HOST: {SANDBOX_HOST}")
        return True
    except ImportError as e:
        print(f"✗ 无法导入配置: {e}")
        return False
    except AttributeError as e:
        print(f"✗ 配置缺少必要的属性: {e}")
        return False


async def main():
    """主测试运行器"""
    print("\n" + "=" * 60)
    print("工具测试套件")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print(f"测试文件位置: {Path(__file__).absolute()}")
    print()

    # 首先测试导入
    imports_ok = test_imports()

    if not imports_ok:
        print("\n⚠ 警告: 某些导入失败。测试可能无法正常运行。")
        print("请在运行测试前安装缺少的依赖。")

    # 检查配置
    config_ok = check_config()

    # 测试 local_search
    search_ok = test_local_search()

    # 测试 python 执行
    python_ok = await test_python_execution()

    # 测试 mcp_creation
    mcp_ok = test_mcp_creation()

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"依赖导入:      {'✓ 通过' if imports_ok else '✗ 失败'}")
    print(f"配置检查:      {'✓ 通过' if config_ok else '✗ 失败'}")
    print(f"local_search:  {'✓ 通过' if search_ok else '✗ 失败'}")
    print(f"python:        {'✓ 通过' if python_ok else '✗ 失败'}")
    print(f"mcp_creation:  {'✓ 通过' if mcp_ok else '✗ 失败'}")
    print("=" * 60)

    if imports_ok and config_ok and search_ok and python_ok and mcp_ok:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n✗ 某些测试失败。请检查上面的输出。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
