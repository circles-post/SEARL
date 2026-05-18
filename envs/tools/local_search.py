import requests
from requests.adapters import HTTPAdapter
from mcp.server.fastmcp import FastMCP  # 假设您已有这个基础库
from config import LOCAL_SEARCH_HOST
mcp = FastMCP("LocalServer")

# 连接池 Session，减少建连开销
session = requests.Session()
adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
session.mount("http://", adapter)
session.mount("https://", adapter)

@mcp.tool()
def search(query: str, topk: int = 3):
    """Search Tool for retrieving relevant information
    
    Args:
        query: query text
    Returns:
        str: The formatted query result
    """
    try:
        # 构建请求数据
        request_data = {
            "queries": [query],
            "topk": topk,
            "return_scores": True
        }
        # 设置请求头和代理
        headers = {
            "Content-Type": "application/json"
        }
        # 使用本地连接，绕过代理
        proxies = {
            "http": None,
            "https": None
        }
        
        response = session.post(
            f"http://{LOCAL_SEARCH_HOST}:5003/retrieve",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=10
        )
        
        response.raise_for_status()
        
        # 解析响应
        result = response.json()

        if not result.get("result"):
            return "No relevant documents found"

        # 格式化搜索结果
        formatted_response = "Following are the search results given by the wiki search engine:"
        documents = result["result"][0] if result["result"] else []

        for doc_info in documents:
            document = doc_info["document"]
            contents = document["contents"]

            # 提取标题和内容（标题是第一行，内容是其余部分）
            lines = contents.split('\n', 1)
            title = lines[0].strip('\"')  # 移除引号
            content = lines[1] if len(lines) > 1 else ""

            formatted_response += f"\nsearch results: \"{title}\"\n{content}"

        return formatted_response
        
    except requests.exceptions.Timeout:
        return "RAG service request timeout, please check if the service is running properly"
    except requests.exceptions.ConnectionError:
        return "Unable to connect to RAG service, please ensure that the service is running"
    except requests.exceptions.RequestException as e:
        return f"RAG service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') else 'No detail'}"
    except Exception as e:
        return f"RAG query failed: {str(e)}\nError type: {type(e).__name__}"


if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')