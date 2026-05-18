import asyncio
import argparse
from storage_manager import create_storage_manager, parse_args

async def test_storage_manager():
    # 解析命令行参数
    args = parse_args()
    
    # 创建存储管理器
    storage = create_storage_manager(args)
    
    # 模拟一个工具调用
    async def mock_tool_call(method_name: str, params: dict, ttl: Optional[int] = None):
        # 尝试从存储获取
        result = await storage.get(method_name, params)
        
        if result is not None:
            print(f"Cache hit for {method_name}")
            return result
        
        # 模拟实际调用
        print(f"Cache miss for {method_name}, executing...")
        result = {"result": f"Result for {method_name} with params {params}"}
        
        # 存入存储
        await storage.set(method_name, params, result, ttl)
        
        return result
    
    # 测试多次调用
    test_cases = [
        ("search", {"query": "test query", "page": 1}, None),  # 无过期时间
        ("search", {"query": "test query", "page": 1}, None),  # 应该命中缓存
        ("search", {"query": "different query", "page": 1}, 60),  # 60秒过期
    ]
    
    for method_name, params, ttl in test_cases:
        result = await mock_tool_call(method_name, params, ttl)
        print(f"Result: {result}\n")
    
    # 打印统计信息
    print("Storage Stats:")
    print(storage.get_stats())

if __name__ == "__main__":
    asyncio.run(test_storage_manager())