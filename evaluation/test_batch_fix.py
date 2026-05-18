import asyncio
import aiohttp
import os
import time
import random
import argparse
from typing import List

# 从环境获取配置
# API_KEY = os.getenv('BRIGHTDATA_API_KEY')
API_KEY = "f824481d-3ee0-4d05-8ac3-8f208b156f31"
ZONE = os.getenv('BRIGHTDATA_ZONE', 'sxh_bing_search')

from urllib.parse import urlencode

async def single_search(session: aiohttp.ClientSession, query: str, task_id: int):
    """模拟单次搜索请求"""
    # 正确进行 URL 编码
    search_params = {
        "q": query,
        "brd_json": "1"
    }
    target_url = f"https://www.bing.com/search?{urlencode(search_params)}"

    payload = {
        "zone": ZONE,
        "url": target_url,
        "format": "raw"
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    start_time = time.time()
    try:
        # 使用 trust_env=False 确保不受系统环境变量代理的影响
        async with session.post(
            "https://api.brightdata.com/request",
            json=payload,
            headers=headers,
            timeout=120
        ) as response:
            status = response.status
            text = await response.text()
            duration = time.time() - start_time

            if status == 200:
                if not text or text.strip() == "":
                    # 打印 Headers 以便排查是哪个网关返回的空内容
                    headers_str = ", ".join([f"{k}: {v}" for k, v in response.headers.items() if k.lower() in ['server', 'x-brd-error', 'x-request-id', 'content-type']])
                    return False, duration, f"Empty Body (Headers: {headers_str})"
                try:
                    import json
                    data = json.loads(text)
                    if 'organic' in data:
                        return True, duration, "Success"
                    else:
                        return False, duration, f"JSON without organic (Keys: {list(data.keys())})"
                except:
                    return False, duration, f"Invalid JSON (First 100 chars: {text[:100]})"
            else:
                return False, duration, f"HTTP {status}: {text[:100]}"
    except Exception as e:
        return False, time.time() - start_time, f"Error: {str(e)}"

async def run_stress_test(concurrency: int, total_requests: int):
    """运行高并发压测"""
    if not API_KEY:
        print("错误: 请设置 BRIGHTDATA_API_KEY 环境变量")
        return

    print(f"=== 开始高并发搜索测试 ===")
    print(f"目标 API: Brightdata ({ZONE})")
    print(f"并发数: {concurrency}, 总请求数: {total_requests}")
    print("---------------------------------------")

    queries = [
        f"test query {i} - {random.randint(1000, 9999)}" for i in range(total_requests)
    ]

    # 限制并发
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def sem_task(session, query, i):
        nonlocal completed
        async with semaphore:
            success, duration, msg = await single_search(session, query, i)
            completed += 1
            status_str = "✅ 成功" if success else f"❌ 失败 ({msg})"
            print(f"[{completed}/{total_requests}] {status_str} | 耗时: {duration:.2f}s | Query: {query[:20]}...")
            return success, duration, msg

    # 使用较短的连接超时进行测试
    timeout = aiohttp.ClientTimeout(total=45, connect=10)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        tasks = [sem_task(session, queries[i], i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)

    # 统计结果
    successes = [r for r in results if r[0]]
    failures = [r for r in results if not r[0]]

    avg_lat = sum(r[1] for r in results) / len(results) if results else 0

    print("\n=== 测试结果报告 ===")
    print(f"总请求数: {total_requests}")
    print(f"成功次数: {len(successes)} ({(len(successes)/total_requests)*100:.1f}%)")
    print(f"失败次数: {len(failures)} ({(len(failures)/total_requests)*100:.1f}%)")
    print(f"平均耗时: {avg_lat:.2f}s")

    if failures:
        print("\n--- 错误类型分布 ---")
        error_counts = {}
        for f in failures:
            err_msg = f[2]
            error_counts[err_msg] = error_counts.get(err_msg, 0) + 1
        for msg, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {msg}: {count} 次")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=int, default=10, help="并发数")
    parser.add_argument("--n", type=int, default=20, help="总请求数")
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.c, args.n))