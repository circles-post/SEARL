import time
import json
from concurrent.futures import ThreadPoolExecutor
from envs.utils.async_mcp_manager import AsyncMCPManager

def main(num_instances=3):
    print("开始测试负载均衡功能...")
    
    # 配置MCP服务器
    config = {
        "mcpServers": {
            "test-server": {
                "url": "http://xxx:8080/sse",
            }
        }
    }
    
    # 初始化管理器
    print("\n1. 初始化AsyncMCPManager...")
    manager = AsyncMCPManager(num_instances=num_instances)
    tools = manager.initConfig(config)
    
    if not tools:
        print("错误：没有找到可用的工具")
        return
        
    # 获取第一个工具用于测试
    test_tool = tools[0]
    print(f"获取到工具: {test_tool.name}")
    print(f"实例数量: {len(test_tool.instances)}")
    
    # 记录每个实例的调用次数
    call_counts = [0] * len(test_tool.instances)
    
    def make_request(i):
        """发送测试请求"""
        try:
            params = {
                "queries": ["麦当劳"],
                "search_type": 0
            }
            
            # 发送请求
            response = test_tool.call(params)
            
            # 记录被调用的实例
            with test_tool._lock:
                for idx, (active, _) in enumerate(test_tool._instance_states):
                    if active > 0:
                        call_counts[idx] += 1
                        break
            
            return response
            
        except Exception as e:
            print(f"\n请求 {i} 失败: {e}")
            return None
    
    print("\n2. 开始并发测试...")
    print("发送30个并发请求...")
    
    start_time = time.time()
    # 使用线程池并发发送请求
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(make_request, i) for i in range(30)]
        
        # 等待所有请求完成
        for future in futures:
            future.result()
    
    end_time = time.time()
    print("Elapsed time: {:.2f}".format(end_time - start_time))


if __name__ == "__main__":
    main(num_instances=1) 