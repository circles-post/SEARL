import json
import uuid
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, LOCAL_SEARCH_HOST
from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
import socket

# 任务队列的名称
REQUEST_QUEUE_NAME = 'web_search_request_queue'

# 结果队列的前缀
RESPONSE_QUEUE_PREFIX = 'web_search_response_queue:'

# 等待结果的超时时间（秒）
RESULT_TIMEOUT = 180  # 等待3分钟

# 全局Redis连接池，避免每次重新创建连接
_redis_pool = None

def get_redis_client():
    """获取Redis客户端，使用连接池复用连接"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=10,  # 最大连接数
            socket_keepalive=True,
            socket_keepalive_options={
                socket.TCP_KEEPIDLE: 60,
                socket.TCP_KEEPINTVL: 10,
                socket.TCP_KEEPCNT: 3,
            }
        )
    return redis.Redis(connection_pool=_redis_pool)

mcp = FastMCP("LocalServer")

@mcp.tool()
def search(query: str, topk: int = 10):
    """Search Tool for Google search

    Args:
        query: query text

    Returns:
        str: The formatted query result
    """
    try:
        # 获取Redis客户端（使用连接池复用连接）
        redis_client = get_redis_client()

        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        response_queue = f"{RESPONSE_QUEUE_PREFIX}{task_id}"

        # 构建任务payload
        task_payload = {
            'task_id': task_id,
            'query': query,
            'topk': topk,
            'return_scores': True,
            'response_queue': response_queue
        }

        # 发送任务到请求队列
        redis_client.lpush(REQUEST_QUEUE_NAME, json.dumps(task_payload))
        print(f"任务 {task_id} 已发送: 查询 '{query}'")

        # 等待结果
        result_data = redis_client.brpop(response_queue, timeout=RESULT_TIMEOUT)

        if result_data is None:
            return f"等待搜索结果超时（超过 {RESULT_TIMEOUT} 秒）"

        # 解析结果
        response_payload = json.loads(result_data[1])
        print(f"成功处理了任务 {response_queue}")

        # 检查结果状态
        flag = response_payload.get('flag', 'error')
        content = response_payload.get('content', '无效的响应格式')

        if flag == 'error':
            return f"Search failed: {content}"
        else:
            return content

    except Exception as e:
        return f"Search query failed: {str(e)}"


if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')