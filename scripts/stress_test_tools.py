import asyncio
import time
import json
import os
import sys
import numpy as np
from tqdm.asyncio import tqdm

# 添加项目根目录到 path 以便导入模块
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from envs.tool_manager.qwen3_manager_baseline import QwenManagerBaseline

# ================= 配置区域 =================
CONFIG_PATH = os.environ.get("RL_FACTORY_MCP_CONFIG", "envs/configs/mcp_tools_local_graph.pydata")
TOOL_TIMEOUT = 60  # 设置较长的超时时间以便测试极限
TEST_ROUNDS = [10, 50, 100, 200] # 并发梯度

# 定义要测试的工具 Payload
# 注意：工具名称通常需要加上 Server 前缀，格式为 "ServerName-ToolName"
# 根据 envs/configs/mcp_tools_local_graph.pydata 配置：
# - search Server 下有 search Tool -> search-search
# - execute_python_code Server 下有 execute_python_code Tool -> execute_python_code-execute_python_code
TOOLS_TO_TEST = {
    "search": {
        "name": "search-search", 
        "args": json.dumps({"query": "current who is jij", "topk": 3})
    },
    "python": {
        "name": "execute_python_code-execute_python_code", 
        "args": json.dumps({"code": "print(123 + 456)", "timeout": 5})
    }
}
# ===========================================

class MockConfig:
    def __init__(self):
        self.config_path = CONFIG_PATH
        self.mcp_mode = 'stdio'
        self.enable_limiter = False # 关闭内置限制器以测试极限，或者开启以测试限制器效果
        self.tool_timeout = TOOL_TIMEOUT
        self.tool_name_selected = []
        self.use_storage_manager = False
        self.enable_thinking = False
        self.local_search = True
        self.max_concurrency = 1000

    def get(self, key, default=None):
        return getattr(self, key, default)

async def run_batch(manager, tool_name, payload, concurrency):
    print(f"\n>>> 开始测试工具 [{tool_name}] 并发数: {concurrency}")
    
    start_time = time.time()
    
    # 构造任务
    # 我们直接调用 manager._call_tool 的逻辑，模拟 execute_actions 的行为
    # 注意：_call_tool 是同步的，但在 manager 中被 asyncio.to_thread 包装
    
    async def single_request(idx):
        try:
            # 模拟不同参数避免缓存（如果是 search）
            current_payload = payload.copy()
            if tool_name == "search":
                args = json.loads(current_payload["args"])
                args["query"] = f"test query {idx} timestamp {time.time()}"
                current_payload["args"] = json.dumps(args)
            
            req_start = time.time()
            # 包装同步调用
            result = await asyncio.to_thread(
                manager._call_tool, 
                current_payload["name"], 
                current_payload["args"]
            )
            duration = time.time() - req_start
            # 简单检查结果是否包含错误标识
            success = True
            if "# 工具调用失败" in str(result) or "# Execute the tool" in str(result) and "failed" in str(result):
                success = False
            if "timeout" in str(result).lower():
                success = False
                
            return {"success": success, "duration": duration, "error": None if success else str(result)[:100]}
        except Exception as e:
            return {"success": False, "duration": time.time() - req_start, "error": str(e)}

    tasks = [single_request(i) for i in range(concurrency)]
    
    # 并发执行
    results = await tqdm.gather(*tasks, desc=f"Testing {tool_name}")
    
    total_time = time.time() - start_time
    
    # 统计
    success_count = sum(1 for r in results if r['success'])
    durations = [r['duration'] for r in results]
    errors = [r['error'] for r in results if not r['success']]
    
    print(f"--- 结果统计 (并发 {concurrency}) ---")
    print(f"总耗时: {total_time:.2f}s")
    print(f"QPS: {concurrency / total_time:.2f}")
    print(f"成功率: {success_count}/{concurrency} ({success_count/concurrency*100:.1f}%)")
    if durations:
        print(f"平均耗时: {np.mean(durations):.4f}s")
        print(f"P50 耗时: {np.percentile(durations, 50):.4f}s")
        print(f"P95 耗时: {np.percentile(durations, 95):.4f}s")
        print(f"P99 耗时: {np.percentile(durations, 99):.4f}s")
    
    if errors:
        print("常见错误 (Top 3):")
        from collections import Counter
        for err, count in Counter(errors).most_common(3):
            print(f"  - {count}x: {err}")
            
    return success_count == concurrency

async def main():
    print("初始化 Tool Manager...")
    config = MockConfig()
    manager = QwenManagerBaseline(config)
    
    for tool_key, payload in TOOLS_TO_TEST.items():
        print(f"\n{'='*40}")
        print(f"测试目标: {tool_key}")
        print(f"{'='*40}")
        
        for concurrency in TEST_ROUNDS:
            await run_batch(manager, tool_key, payload, concurrency)
            # 冷却时间，避免瞬间过载影响下一轮
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
