"""
将 plan.yaml 中第 2-30 行追加到 prompt 字段的 system prompt，并生成新 parquet。
"""

import json
import math
import os
from pathlib import Path
import pandas as pd


SRC = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_SRC", "data/train.parquet"))
DST = Path(os.environ.get("RL_FACTORY_APPEND_PLAN_DST", "data/train_plan.parquet"))
PLAN_PATH = Path(os.environ.get("RL_FACTORY_PLAN_PROMPT", "prompt/plan.yaml"))


def load_plan_text() -> str:
    """读取 plan.yaml 的第 2-30 行作为 system prompt 片段。"""
    lines = PLAN_PATH.read_text().splitlines()
    # 行号从 1 开始：取索引 1-29（含）对应第 2-30 行
    sliced = lines[1:30]
    return "\n".join(sliced)


def normalize_prompt(prompt):
    """
    将 prompt 归一化为 list[dict]，避免 pyarrow 结构混杂报错。
    支持 None/NaN、dict、list、JSON 字符串、普通字符串等。
    """
    if prompt is None or (isinstance(prompt, float) and math.isnan(prompt)):
        return []

    if isinstance(prompt, list):
        return prompt

    if isinstance(prompt, dict):
        return [prompt]

    if isinstance(prompt, str):
        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            # 非 JSON 字符串则作为 system 文本包裹
            return [{"role": "system", "content": prompt}]

    # 其他类型兜底为 system 文本
    return [{"role": "system", "content": str(prompt)}]


def add_plan_to_prompt(prompt, plan_text: str):
    """
    将 plan_text 追加到 prompt 的 system 段。
    prompt 预期为 list[dict]，但做了健壮性处理。
    """
    prompt = normalize_prompt(prompt)

    if prompt and isinstance(prompt[0], dict) and prompt[0].get("role", "").lower() == "system":
        prompt[0]["content"] = f"{prompt[0].get('content', '')}\n\n{plan_text}"
        return prompt

    # 若无 system，则创建新的 system 放在开头
    return [{"role": "system", "content": plan_text}] + prompt


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"源文件不存在: {SRC}")
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"plan 文件不存在: {PLAN_PATH}")

    plan_text = load_plan_text()
    df = pd.read_parquet(SRC)
    df["prompt"] = df["prompt"].apply(lambda p: add_plan_to_prompt(p, plan_text))
    df.to_parquet(DST, index=False)
    print(f"完成: {len(df)} 行写入 {DST}")


if __name__ == "__main__":
    main()
