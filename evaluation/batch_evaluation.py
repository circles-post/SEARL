#!/usr/bin/env python3
"""
批量评测脚本：
- 扫描指定文件夹下的所有JSON文件
- 对每个JSON文件运行多个数据集的评测
- 输出统一的JSON文件，格式为：{json_filename: {dataset_name: evaluation_results}}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

from evaluation.evaluation import (
    prepare_answer_file,
    evaluate_dataset,
    _build_dataset_map,
)
from evaluation.gpt_judge import MathLLM


class BatchEvaluator:
    """批量评测器：处理文件夹下所有JSON文件的评测任务"""

    def __init__(
        self,
        input_folder: str,
        output_file: str,
        datasets: List[str],
        model_path: str = None,
        dataset_paths: Dict[str, str] = None,
        file_pattern: str = "answer_*.json",
    ):
        """
        Args:
            input_folder: 输入文件夹路径
            output_file: 输出JSON文件路径
            datasets: 要评测的数据集列表
            model_path: vLLM模型路径
            dataset_paths: 数据集路径覆盖字典 {dataset_name: path}
            file_pattern: JSON文件匹配模式
        """
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.datasets = datasets
        self.model_path = model_path
        self.dataset_paths = dataset_paths or {}
        self.file_pattern = file_pattern

        # 从输入文件夹路径中提取模型名称（validation_samples的父目录名）
        self.model_name = self.extract_model_name(self.input_folder)

        # 创建输出目录
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建临时工作目录
        self.temp_dir = self.output_file.parent / "temp_evaluation"
        self.temp_dir.mkdir(exist_ok=True)

        # 初始化判分器（复用单个实例，避免重复加载模型）
        print(f"\n{'='*80}")
        print("正在初始化Judge模型（全局复用，避免重复加载）...")
        print(f"模型路径: {model_path or 'Default (Qwen/Qwen2.5-7B-Instruct)'}")
        print(f"{'='*80}\n")
        self.judge = MathLLM(model_path=model_path) if model_path else MathLLM()
        print("✓ Judge模型初始化完成，将在所有评测中复用此实例\n")

    def extract_model_name(self, input_folder: Path) -> str:
        """
        从输入文件夹路径中提取模型名称
        例如:
            /path/to/baseline_gpu6_grpo_prm_new_set/to_do -> baseline_gpu6_grpo_prm_new_set
            /path/to/grpo_baseline_1gpu/validation_samples -> grpo_baseline_1gpu

        Args:
            input_folder: 输入文件夹路径

        Returns:
            str: 模型名称
        """
        folder_name = input_folder.name

        # 如果输入文件夹是 validation_samples 或 to_do，则使用其父目录名称
        if folder_name in ["validation_samples", "to_do"]:
            return input_folder.parent.name

        # 否则使用输入文件夹本身的名称
        return folder_name

    def find_json_files(self) -> List[Path]:
        """查找文件夹下所有匹配的JSON文件"""
        if not self.input_folder.exists():
            raise ValueError(f"输入文件夹不存在: {self.input_folder}")

        json_files = []

        # 使用glob递归查找匹配的文件
        for json_file in self.input_folder.rglob(self.file_pattern):
            if json_file.is_file():
                json_files.append(json_file)

        print(f"在 {self.input_folder} 中找到 {len(json_files)} 个JSON文件")
        return sorted(json_files)

    def evaluate_single_file(self, json_file: Path) -> Dict[str, Dict]:
        """
        对单个JSON文件进行所有数据集的评测

        Returns:
            Dict[dataset_name, evaluation_results]
        """
        results = {}
        file_name = json_file.name

        print(f"\n{'='*80}")
        print(f"正在评测文件: {file_name}")
        print(f"{'='*80}")

        for dataset_name in self.datasets:
            print(f"\n--- 数据集: {dataset_name} ---")

            try:
                # 为每个文件+数据集创建独立的临时目录
                temp_output_dir = self.temp_dir / file_name.replace('.json', '') / dataset_name
                temp_output_dir.mkdir(parents=True, exist_ok=True)

                # 步骤1: 准备答案文件（对齐ground truth）
                dataset_path = self.dataset_paths.get(dataset_name)
                prepared_path = prepare_answer_file(
                    dataset_name=dataset_name,
                    answer_json_path=str(json_file),
                    output_dir=str(temp_output_dir),
                    dataset_root_override=dataset_path,
                )

                # 步骤2: 评估（传入复用的judge实例）
                eval_output_path = evaluate_dataset(
                    dataset_name=dataset_name,
                    prepared_answer_path=prepared_path,
                    model_path=self.model_path,
                    output_dir=str(temp_output_dir),
                    judge=self.judge,  # 传入复用的judge实例，避免重复加载模型
                )

                # 步骤3: 读取评估结果
                with open(eval_output_path, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)

                results[dataset_name] = eval_results
                print(f"✓ {dataset_name} 评测完成")

            except Exception as e:
                print(f"✗ {dataset_name} 评测失败: {e}")
                results[dataset_name] = {"error": str(e)}

        return results

    def run(self, skip_existing: bool = True):
        """
        执行批量评测

        Args:
            skip_existing: 是否跳过已存在的结果
        """
        # 查找所有JSON文件
        json_files = self.find_json_files()

        if not json_files:
            print("未找到任何匹配的JSON文件")
            return

        # 读取已有结果（如果存在）
        all_results = {}
        if skip_existing and self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
                print(f"\n读取到已有结果: {len(all_results)} 个模型")
            except json.JSONDecodeError:
                print("\n警告: 已有结果文件损坏，将重新开始")
                all_results = {}

        # 初始化当前模型的结果字典（如果不存在）
        if self.model_name not in all_results:
            all_results[self.model_name] = {}

        # 逐个处理JSON文件
        for json_file in tqdm(json_files, desc=f"评测 {self.model_name}"):
            file_name = json_file.name

            # 跳过已评测的文件
            if skip_existing and file_name in all_results[self.model_name]:
                print(f"\n跳过已评测文件: {file_name}")
                continue

            print(f"\n处理文件: {file_name}")

            # 评测单个文件
            file_results = self.evaluate_single_file(json_file)

            # 保存到模型字典下，以文件名为key
            all_results[self.model_name][file_name] = file_results

            # 实时保存结果（防止中途失败丢失数据）
            self.save_results(all_results)

        print(f"\n{'='*80}")
        print(f"模型 {self.model_name} 评测完成！结果已保存到: {self.output_file}")
        print(f"{'='*80}")

        # 打印摘要
        self.print_summary(all_results)

    def save_results(self, results: Dict):
        """保存结果到JSON文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def print_summary(self, results: Dict):
        """打印评测摘要"""
        print("\n" + "="*80)
        print("评测摘要")
        print("="*80)

        for model_name, files_results in results.items():
            print(f"\n模型: {model_name}")
            for file_name, datasets_results in files_results.items():
                print(f"  文件: {file_name}")
                for dataset_name, eval_results in datasets_results.items():
                    if "error" in eval_results:
                        print(f"    - {dataset_name}: 失败 ({eval_results['error']})")
                    else:
                        # 提取关键指标
                        if isinstance(eval_results, dict):
                            if "overall" in eval_results:
                                acc = eval_results["overall"].get("acc", "N/A")
                                print(f"    - {dataset_name}: acc={acc:.4f}" if isinstance(acc, float) else f"    - {dataset_name}: acc={acc}")
                            else:
                                # 对于有子类别的数据集（如HLE、GAIA等）
                                accs = []
                                for category, metrics in eval_results.items():
                                    if isinstance(metrics, dict) and "acc" in metrics:
                                        accs.append(metrics["acc"])
                                if accs:
                                    avg_acc = sum(accs) / len(accs)
                                    print(f"    - {dataset_name}: avg_acc={avg_acc:.4f}")
                                else:
                                    print(f"    - {dataset_name}: 完成")


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量评测工具：处理文件夹下所有JSON文件的评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 评测单个文件夹的所有answer_*.json文件：
   python batch_evaluation.py \\
       --input-folder /path/to/validation_samples \\
       --output-file results/batch_eval.json \\
       --datasets 2wiki bamboogle hotpotqa musique

2. 指定自定义数据集路径：
   python batch_evaluation.py \\
       --input-folder /path/to/validation_samples \\
       --output-file results/batch_eval.json \\
       --datasets gsm8k math500 \\
       --dataset-path gsm8k=/path/to/gsm8k.jsonl \\
       --dataset-path math500=/path/to/math500.jsonl

3. 使用自定义模型路径：
   python batch_evaluation.py \\
       --input-folder /path/to/validation_samples \\
       --output-file results/batch_eval.json \\
       --datasets 2wiki bamboogle \\
       --model-path /path/to/model
        """
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        help="输入文件夹路径（包含要评测的JSON文件）"
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="输出JSON文件路径"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="要评测的数据集列表（支持: HLE, GAIA, WebWalker, XBench, 2wiki, bamboogle, hotpotqa, musique, gsm8k, math500, aime24, aime25）"
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="vLLM模型权重路径（HF名称或本地路径）"
    )

    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        help="数据集路径覆盖（格式: dataset_name=path），可多次使用"
    )

    parser.add_argument(
        "--file-pattern",
        default="answer_*.json",
        help="JSON文件匹配模式（默认: answer_*.json）"
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="不跳过已评测的文件，重新评测所有文件"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析数据集路径覆盖
    dataset_paths = {}
    for item in args.dataset_path:
        if "=" not in item:
            print(f"警告: 忽略格式错误的dataset-path参数: {item}")
            continue
        dataset_name, path = item.split("=", 1)
        dataset_paths[dataset_name] = path

    # 创建批量评测器
    evaluator = BatchEvaluator(
        input_folder=args.input_folder,
        output_file=args.output_file,
        datasets=args.datasets,
        model_path=args.model_path,
        dataset_paths=dataset_paths,
        file_pattern=args.file_pattern,
    )

    # 执行评测
    evaluator.run(skip_existing=not args.no_skip_existing)


if __name__ == "__main__":
    main()
