import asyncio, json, random, string, time
from functools import partial

from mcp_creation import create_and_execute_mcp

# 并发批次数与单批并发度
TOTAL_TASKS = 200
CONCURRENCY = 20
TIMEOUT = 8.0  # create_and_execute_mcp 的 timeout 参数

def rand_name(prefix="stress"):
    tail = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{tail}"

def one_call():
    name = rand_name()
    code = f"def {name}(x:int, y:int):\n    return x + y\n"
    start = time.time()
    resp = json.loads(
        create_and_execute_mcp(
            name=name,
            description="stress add",
            arguments="x:int, y:int",
            returns="int",
            code=code,
            inputs={"x": 1, "y": 2},
            timeout=TIMEOUT,
        )
    )
    cost = time.time() - start
    return resp, cost

async def worker(semaphore, stats):
    async with semaphore:
        try:
            resp, cost = await asyncio.to_thread(one_call)
            ok = resp.get("creation_success") and resp.get("execution_result") == 3
            stats["ok"] += int(ok)
            stats["fail"] += int(not ok)
            stats["latencies"].append(cost)
        except Exception as e:
            stats["fail"] += 1
            stats["errors"].append(str(e))

async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"ok": 0, "fail": 0, "errors": [], "latencies": []}
    tasks = [asyncio.create_task(worker(sem, stats)) for _ in range(TOTAL_TASKS)]
    await asyncio.gather(*tasks)
    if stats["latencies"]:
        p95 = sorted(stats["latencies"])[int(0.95 * len(stats["latencies"])) - 1]
    else:
        p95 = None
    print(json.dumps({
        "ok": stats["ok"],
        "fail": stats["fail"],
        "p95_latency_sec": p95,
        "errors_sample": stats["errors"][:5],
    }, indent=2))

if __name__ == "__main__":
    asyncio.run(main())