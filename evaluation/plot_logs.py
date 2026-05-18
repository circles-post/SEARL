import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_log_line(line):
    """
    从一行 log 中提取 key:value 对
    返回 dict
    """
    # 匹配类似 key:value 的字段，忽略 ANSI 颜色码
    line = re.sub(r'\x1b\[[0-9;]*m', '', line)  # 去掉颜色码
    parts = line.split(" - ")
    metrics = {}
    for part in parts:
        if ":" in part:
            key, val = part.split(":", 1)
            try:
                val = float(val.strip())
            except ValueError:
                continue
            metrics[key.strip()] = val
    return metrics

def collect_metrics(log_dir):
    """
    遍历 log 文件夹，收集所有指标
    """
    all_metrics = defaultdict(list)
    file_list = sorted(
        [f for f in os.listdir(log_dir) if f.startswith("chunk_") and f.endswith(".log")],
        key=lambda x: int(re.search(r"chunk_(\d+)\.log", x).group(1))
    )

    for fname in file_list:
        path = os.path.join(log_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            metrics = parse_log_line(first_line)
            for k, v in metrics.items():
                all_metrics[k].append(v)
    return all_metrics

def plot_metrics(all_metrics, output_dir):
    """
    绘制每个指标的曲线图
    """
    os.makedirs(output_dir, exist_ok=True)
    for metric, values in all_metrics.items():
        plt.figure(figsize=(10, 5))
        plt.plot(values, marker="o")
        plt.title(metric)
        plt.xlabel("Log Index (chunk_i)")
        plt.ylabel("Value")
        plt.grid(True)
        safe_name = metric.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()

if __name__ == "__main__":
    log_dir = os.environ.get("RL_FACTORY_LOG_CHUNKS_DIR", "logs/chunks")
    output_dir = os.environ.get("RL_FACTORY_LOG_CURVES_DIR", "logs/curves")

    all_metrics = collect_metrics(log_dir)
    plot_metrics(all_metrics, output_dir)

    print(f"绘图完成，共生成 {len(all_metrics)} 张图，保存在 {output_dir}")
