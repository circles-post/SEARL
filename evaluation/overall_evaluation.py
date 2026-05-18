from evaluation.evaluation import (
    prepare_answer_file,
    evaluate_dataset,
    _build_dataset_map,
    _norm_text,
    main_HLE,
    main_gaia,
    main_webwalker,
    main_xbench,
    main_qa,
    process_item,
    process_item_gaia,
    process_item_webwalker,
    process_item_xbench,
    process_item_qa,
    evaluation_metrics_HLE_key,
    evaluation_metrics_gaia_level,
    evaluation_metrics_webwalker_level,
    evaluation_metrics_HLE,
)
from plot_logs import *
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset



class DatasetEvaluator:
    """
    一个用于评估不同数学和逻辑推理数据集的类。
    它封装了数据加载、答案匹配和调用特定评估函数的过程。
    """
    def __init__(self, input_json_path: str, output_dir: str):
        """
        初始化评估器。

        Args:
            output_dir (str): 用于保存评估结果的文件夹路径。
        """
        self.input_json_path = input_json_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"评估结果将保存到: {self.output_dir}")
        # 初始化 MathLLM，避免在评估函数中重复创建
        # 注意: 请确保 MathLLM 的定义可以被这个类访问
        # self.math_llm = MathLLM() 

    def _load_json(self, file_path: str) -> list:
        """从文件路径加载 JSON 数据。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[!] 错误: 找不到文件 {file_path}")
            return []

    def prepare_answer_json_path(self, input_data, dataset_name):
        answer_json = self._load_json(self.input_json_path)
        if dataset_name == "HLE":
            output_path = main_HLE(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "GAIA":
            output_path = main_gaia(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "WebWalker":
            output_path = main_webwalker(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        elif dataset_name == "XBench":
            output_path = main_xbench(data=input_data, answer_json=answer_json, output_dir=self.output_dir)
        else:
            output_path = main_qa(data=input_data, answer_json=answer_json, output_dir=self.output_dir, dataset_name=dataset_name)
        return output_path

    def _filter_and_process(self, dataset_name: str, raw_item: dict, processor, idx: int):
        """与 match_stats 保持一致的过滤逻辑。"""
        if dataset_name == "HLE":
            if raw_item.get("image") or raw_item.get("category") not in ["Physics", "Math", "Computer Science/AI"]:
                return None
        if dataset_name == "GAIA":
            if raw_item.get("file_name") != "":
                return None
        return processor(raw_item, idx)

    def _load_split_safe(self, dataset_name: str, cfg: dict):
        """安全加载数据集：若指定 split 不存在则回退到 train。"""
        split = cfg["split"]
        if dataset_name in ["HLE", "GAIA", "XBench"]:
            ds = load_dataset(cfg["loader"])
        elif dataset_name == "WebWalker":
            ds = load_dataset("json", data_files=cfg["loader"])
        else:
            ds = load_dataset("json", data_files=cfg["loader"])
        if split in ds:
            return ds[split]
        # 回退：若没有指定 split，则使用 train 或第一个可用 split
        if "train" in ds:
            return ds["train"]
        first_split = list(ds.keys())[0]
        return ds[first_split]

    def prepare_data(self, dataset_name: str) -> list:
        """
        准备用于评估的数据。
        加载原始数据集和模型答案，并将它们匹配起来。

        Args:
            dataset_name (str): 数据集名称 (例如 'HLE', 'GAIA', 'WebWalker', 'XBench').
            answer_json_path (str): 模型生成的答案文件路径。

        Returns:
            list: 包含了标准答案和元数据的模型答案列表。
        """
        dataset_map = _build_dataset_map()
        if dataset_name not in dataset_map:
            raise ValueError(f"不支持的数据集: {dataset_name}。支持的选项为: {list(dataset_map.keys())}")
        cfg = dataset_map[dataset_name]

        input_data = self._load_split_safe(dataset_name, cfg)

        answer_json_path = self.prepare_answer_json_path(input_data, dataset_name)
        print(f"--- 正在为数据集 '{dataset_name}' 准备数据 ---, {answer_json_path}")
        answer_json = self._load_json(answer_json_path)
        if not answer_json:
            print(f"--- 数据集 '{dataset_name}' 准备数据失败 ---")
            return []

        config = cfg
        source_data = self._load_split_safe(dataset_name, config)
        processor = config["processor"]

        processed = []
        for idx, item in enumerate(tqdm(source_data, desc=f"从 {dataset_name} 提取标准答案")):
            processed_item = self._filter_and_process(dataset_name, item, processor, idx)
            if processed_item:
                processed.append(processed_item)

        # 执行匹配逻辑
        matched_data = []
        for item in tqdm(answer_json, desc="匹配问题与答案"):
            inputs_raw = item.get('input', item.get('question', ''))  # 兼容不同格式
            inputs_norm = _norm_text(inputs_raw)
            for tup in processed:
                tgt_norm = _norm_text(tup[0])
                if not tgt_norm:
                    continue
                # 双向包含匹配，参照 match_stats
                if tgt_norm in inputs_norm or inputs_norm in tgt_norm:
                    item['ground_truth'] = tup[1]
                    item['question'] = tup[0]
                    # 根据数据集添加额外元数据
                    if dataset_name == 'HLE':
                        item['category'] = tup[2]
                    elif dataset_name == 'GAIA':
                        item['Level'] = tup[2]
                    elif dataset_name == 'WebWalker':
                        item['difficulty_level'] = tup[2]
                    
                    matched_data.append(item)
                    break
        
        print(f"成功匹配 {len(matched_data)} / {len(answer_json)} 条数据。")
        return matched_data

    def split_log_into_chunks(self, log_file_path, output_dir=None, start_pattern='actor/kl_loss:'):
        """
        根据起始模式将一个大日志文件分割成多个小文件。

        :param log_file_path: 输入的日志文件路径。
        :param start_pattern: 标志着一个新 chunk 开始的字符串。
        :param output_dir: 存放切分后文件的目录。
        """
        # 确保输出目录存在
        if output_dir:
            output_dir = os.path.join(output_dir, "chunks")
        else:
            output_dir = os.path.join(self.output_dir, "chunks")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        chunk_count = 0
        current_chunk_file = None

        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if start_pattern in line.strip():
                        # 如果已经有一个文件在写入，先关闭它
                        if current_chunk_file:
                            current_chunk_file.close()
                        
                        # 准备开始写入下一个新文件
                        chunk_count += 1
                        chunk_filename = os.path.join(output_dir, f'chunk_{chunk_count}.log')
                        current_chunk_file = open(chunk_filename, 'w', encoding='utf-8')
                        print(f"检测到新 chunk，开始写入文件: {chunk_filename}")
                    
                    # 如果已经开始了一个 chunk，就把当前行写入
                    if current_chunk_file:
                        current_chunk_file.write(line)
            # 循环结束后，确保最后一个文件被关闭
            if current_chunk_file:
                current_chunk_file.close()

            print(f"\n处理完成！总共分割出 {chunk_count} 个文件。")

        except FileNotFoundError:
            print(f"错误: 文件 '{log_file_path}' 未找到。")
        except Exception as e:
            print(f"发生错误: {e}")

    def run_evaluation(self, dataset_name: str, judge: MathLLM | None = None):
        """
        执行完整的评估流程，并将结果以可续写的方式保存到单个JSON文件中。

        Args:
            dataset_name (str): 要评估的数据集名称。
            judge: 复用的judge实例，如果为None则在内部创建（避免重复加载模型）
        """
        # 1. 准备数据
        eval_data = self.prepare_data(dataset_name)
        print(len(eval_data), "######")
        if not eval_data:
            print("数据准备失败，评估终止。")
            return

        print(f"--- 开始评估数据集 '{dataset_name}' ---")

        # 如果没有传入judge，则创建新的（但建议外部传入以复用）
        if judge is None:
            from evaluation.gpt_judge import MathLLM
            judge = MathLLM()
            print("警告: 未传入judge实例，创建了新的judge（建议外部传入以复用模型）")

        # 用于存储当前数据集评估结果的字典
        results_for_current_dataset = defaultdict(dict)
        metric_keys = ["no_answer", "boxed_not_found", "correct", "false_answer", "acc"]

        # 2. 根据数据集调用评估函数，并将结果存入字典
        if dataset_name == 'HLE':
            for category in ["Math", "Computer Science/AI", "Physics"]:
                print(f"\n评估 HLE - {category} 分类:")
                metrics = evaluation_metrics_HLE_key(eval_data, category=category, math_llm=judge)
                results_for_current_dataset[category] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'GAIA':
            for level in [1, 2, 3]:
                print(f"\n评估 GAIA - Level {level}:")
                metrics = evaluation_metrics_gaia_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name == 'WebWalker':
            levels = sorted(list(set(item['difficulty_level'] for item in eval_data)))
            for level in levels:
                print(f"\n评估 WebWalker - Difficulty Level {level}:")
                metrics = evaluation_metrics_webwalker_level(eval_data, level=level, math_llm=judge)
                results_for_current_dataset[f'Difficulty Level {level}'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["XBench", "2wiki", "bamboogle", "hotpotqa", "musique"]:
            print(f"\n评估 {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        elif dataset_name in ["gsm8k", "math500", "aime24", "aime25"]:
            print(f"\n评估 math - {dataset_name}:")
            metrics = evaluation_metrics_HLE(eval_data, math_llm=judge)
            results_for_current_dataset['overall'] = dict(zip(metric_keys, metrics))

        else:
            print(f"'{dataset_name}' 的评估逻辑未定义，评估终止。")
            return


        answer_path = Path(self.input_json_path)
        sample_key = answer_path.stem
        # 适配 .../{run_name}/validation_samples/{file}.json 结构
        if answer_path.parent.name == "validation_samples":
            run_name = answer_path.parent.parent.name
        else:
            run_name = answer_path.parent.name

        results_root = Path(os.environ.get("RL_FACTORY_EVAL_RESULTS_DIR", "evaluation/results"))
        run_dir = results_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        results_json_path = run_dir / "evaluation.json"

        # 读取已有结果并按 sample_key 合并
        all_results = {}
        if results_json_path.exists() and results_json_path.stat().st_size > 0:
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    all_results = json.load(f)
            except json.JSONDecodeError as e:
                print(f"读取结果文件时解析失败，将覆盖写入: {e}")
                all_results = {}

        sample_bucket = all_results.setdefault(sample_key, {})
        sample_bucket[dataset_name] = dict(results_for_current_dataset)

        try:
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print(f"已将数据集 '{dataset_name}' 的结果写入 {results_json_path}，键为 '{sample_key}'")
        except Exception as e:
            print(f"保存结果时发生错误: {e}")

        print(f"--- 数据集 '{dataset_name}' 评估完成 ---")


def _load_split_safe_global(dataset_name: str, cfg: dict):
    """独立函数版安全加载，供 CLI 直接调用 prepare_answer_file 时使用。"""
    split = cfg["split"]
    if dataset_name in ["HLE", "GAIA", "XBench"]:
        ds = load_dataset(cfg["loader"])
    elif dataset_name == "WebWalker":
        ds = load_dataset("json", data_files=cfg["loader"])
    else:
        ds = load_dataset("json", data_files=cfg["loader"])
    if split in ds:
        return ds[split]
    if "train" in ds:
        return ds["train"]
    first_split = list(ds.keys())[0]
    return ds[first_split]


def _prepare_answer_file_safe(dataset_name: str, answer_json_path: str, output_dir: str, dataset_root_override: str | None = None):
    """带 split 回退的 prepare_answer_file，避免 KeyError('test')。"""
    # 若未提供输出目录，则默认放在答案文件同级目录下的 prepared_answers 内
    if output_dir is None:
        answer_path = Path(answer_json_path)
        output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[提示] 未指定 output_dir，使用默认路径: {output_dir}")
    dataset_map = _build_dataset_map()
    if dataset_name not in dataset_map:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    cfg = dataset_map[dataset_name].copy()
    cfg["loader"] = dataset_root_override or cfg["loader"]
    input_data = _load_split_safe_global(dataset_name, cfg)

    with open(answer_json_path, "r", encoding="utf-8") as f:
        answer_json = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    prepare_fn = cfg["prepare_fn"]
    if prepare_fn == main_qa:
        return prepare_fn(input_data, answer_json, output_dir, dataset_name)
    return prepare_fn(input_data, answer_json, output_dir)


def cli():
    """简单 CLI：对齐答案并评估，输出到指定目录。"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集名称，例如 HLE/GAIA/WebWalker/XBench/...")
    parser.add_argument("--answers", required=True, help="模型生成的答案 JSON（未对齐 ground_truth）")
    parser.add_argument("--output-dir", required=False, help="输出目录，包含对齐答案与评估报告")
    parser.add_argument("--model-path", default=None, help="vLLM 模型权重（HF 名称或本地路径）")
    parser.add_argument("--dataset-path", default=None, help="可选：覆盖默认数据集路径")
    args = parser.parse_args()

    # CLI 层也做一次默认输出目录兜底，避免 None 传入导致 os.makedirs 报错
    if args.output_dir is None:
        answer_path = Path(args.answers)
        args.output_dir = str(answer_path.parent / "prepared_answers")
        print(f"[提示] 未指定 output_dir，使用默认路径: {args.output_dir}")

    # 先对齐答案，再评估（使用安全版，避免 split 缺失导致 KeyError）
    
    prepared = _prepare_answer_file_safe(
        dataset_name=args.dataset,
        answer_json_path=args.answers,
        output_dir=args.output_dir,
        dataset_root_override=args.dataset_path,
    )
    evaluate_dataset(
        dataset_name=args.dataset,
        prepared_answer_path=prepared,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )
    


if __name__ == '__main__':
    cli()
