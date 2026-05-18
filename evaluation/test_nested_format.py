#!/usr/bin/env python3
"""测试嵌套输出格式"""

import json
from pathlib import Path


def test_nested_structure():
    """测试三层嵌套结构的构建逻辑"""

    print("测试三层嵌套结构构建逻辑")
    print("=" * 80)

    # 模拟场景
    test_scenarios = [
        {
            "input_folder": "/path/to/grpo_baseline_1gpu/validation_samples",
            "json_files": [
                "validation_samples_20251223_232253.json",
                "validation_samples_20251224_100000.json"
            ],
            "expected_model_name": "grpo_baseline_1gpu"
        },
        {
            "input_folder": "/path/to/grpo_baseline_6gpu/validation_samples",
            "json_files": [
                "validation_samples_20251223_150000.json"
            ],
            "expected_model_name": "grpo_baseline_6gpu"
        }
    ]

    # 模拟构建结果结构
    all_results = {}

    for scenario in test_scenarios:
        model_name = Path(scenario["input_folder"]).parent.name

        if model_name not in all_results:
            all_results[model_name] = {}

        for json_file in scenario["json_files"]:
            # 模拟数据集评估结果
            file_results = {
                "hotpotqa": {
                    "overall": {
                        "acc": 0.85,
                        "correct": 85,
                        "total": 100
                    }
                },
                "gsm8k": {
                    "overall": {
                        "acc": 0.90,
                        "correct": 90,
                        "total": 100
                    }
                }
            }

            all_results[model_name][json_file] = file_results

    # 打印结果结构
    print("\n生成的结构:")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))

    # 验证结构
    print("\n\n结构验证:")
    print("=" * 80)

    for model_name, files_results in all_results.items():
        print(f"✓ 第一层 - 模型名称: {model_name}")
        for file_name, datasets_results in files_results.items():
            print(f"  ✓ 第二层 - JSON文件: {file_name}")
            for dataset_name, eval_results in datasets_results.items():
                print(f"    ✓ 第三层 - 数据集: {dataset_name}")
                if "overall" in eval_results:
                    acc = eval_results["overall"]["acc"]
                    print(f"      → acc={acc:.2f}")

    print("\n" + "=" * 80)
    print("✓ 三层嵌套结构正确！")
    print("=" * 80)

    # 展示期望的输出格式
    print("\n期望的JSON输出格式:")
    print("-" * 80)
    expected_format = """
{
  "grpo_baseline_1gpu": {
    "validation_samples_20251223_232253.json": {
      "hotpotqa": {...},
      "gsm8k": {...}
    },
    "validation_samples_20251224_100000.json": {
      "hotpotqa": {...},
      "gsm8k": {...}
    }
  },
  "grpo_baseline_6gpu": {
    "validation_samples_20251223_150000.json": {
      "hotpotqa": {...},
      "gsm8k": {...}
    }
  }
}
    """
    print(expected_format)
    print("-" * 80)


def test_model_name_extraction():
    """测试模型名称提取逻辑"""

    print("\n\n测试模型名称提取逻辑")
    print("=" * 80)

    test_cases = [
        ("/path/to/grpo_baseline_1gpu_reasoning_nonthinking/validation_samples",
         "grpo_baseline_1gpu_reasoning_nonthinking"),
        ("/path/to/grpo_baseline_6gpu_reasoning_nonthinking_1211/validation_samples",
         "grpo_baseline_6gpu_reasoning_nonthinking_1211"),
        ("/path/to/reinforce_pp_baseline_6gpu_reasoning/validation_samples",
         "reinforce_pp_baseline_6gpu_reasoning"),
    ]

    for input_path, expected in test_cases:
        p = Path(input_path)
        if p.name == "validation_samples":
            extracted = p.parent.name
        else:
            extracted = p.name

        status = "✓" if extracted == expected else "✗"
        print(f"{status} {input_path}")
        print(f"   期望: {expected}")
        print(f"   提取: {extracted}")
        print()


if __name__ == "__main__":
    test_nested_structure()
    test_model_name_extraction()
