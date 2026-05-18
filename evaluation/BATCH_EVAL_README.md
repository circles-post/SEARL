# 批量评测工具 (batch_evaluation.py)

## 功能说明

`batch_evaluation.py` 是一个批量评测工具，可以处理给定文件夹下的所有JSON文件，并生成统一的评测结果。

## ⚡ 性能优化

本工具包含两项关键性能优化，确保大规模评测的高效执行：

### 1. Judge模型全局复用（避免重复加载）

**问题**: vLLM模型加载非常耗时（通常需要数十秒到数分钟）

**解决方案**: Judge模型在整个评测过程中**只初始化一次**，所有文件、所有数据集共享同一个judge实例。

```python
# 在 BatchEvaluator.__init__ 中初始化一次
self.judge = MathLLM(model_path=model_path)

# 在所有评测中复用
evaluate_dataset(..., judge=self.judge)
```

**效果**:
- 单次模型加载：~30秒
- 如果评测10个数据集，节省：~4.5分钟（9次加载）
- 如果评测10个文件 × 10个数据集，节省：~45分钟（99次加载）

### 2. 批处理推理（Batch Inference）

**问题**: 逐个样本推理效率低，GPU利用率不足

**解决方案**: 所有评测函数使用批处理方式，一次性处理多个样本（默认batch_size=32）

```python
# 旧方式（逐个处理）
for item in items:
    score = judge.evaluation(pred, ref, q)

# 新方式（批处理）
batch_items = [(pred, ref, q) for ...]
scores = judge.evaluation_batch(batch_items)  # 一次处理32个
```

**效果**:
- 单样本推理：~0.5秒/样本
- 批处理（32样本）：~0.1秒/样本
- **加速比**: ~5x（视GPU和模型而定）

### 性能对比

以评测1000个样本为例：

| 方式 | Judge加载 | 推理时间 | 总时间 |
|------|----------|---------|--------|
| 旧方式（每次加载） | 30s × 10次 = 300s | 0.5s × 1000 = 500s | **800s (13.3分钟)** |
| 新方式（复用+批处理） | 30s × 1次 = 30s | 0.1s × 1000 = 100s | **130s (2.2分钟)** |
| **加速比** | | | **6.2x** |

### 与原脚本的主要区别

**原脚本 (baseline_reason.sh) 的输出:**
```
evaluation/results/baseline_reasoning/
├── run_name1/
│   ├── answer_file1/
│   │   ├── evaluation_2wiki.json
│   │   ├── evaluation_bamboogle.json
│   │   ├── evaluation_hotpotqa.json
│   │   └── ...
│   └── answer_file2/
│       └── ...
└── run_name2/
    └── ...
```
- 每个数据集生成单独的JSON文件
- 结果分散在多个文件中
- 难以快速查看整体情况

**新脚本 (batch_evaluation.py) 的输出:**
```json
{
  "answer_file1.json": {
    "2wiki": {
      "overall": {
        "no_answer": 0,
        "boxed_not_found": 2,
        "correct": 45,
        "false_answer": 3,
        "acc": 0.9
      }
    },
    "bamboogle": { ... },
    "hotpotqa": { ... }
  },
  "answer_file2.json": {
    "2wiki": { ... },
    ...
  }
}
```
- 所有结果汇总在一个JSON文件中
- 以文件名为key，便于快速查找
- 支持实时保存和断点续评

## 使用方法

### 基础用法

```bash
python -m evaluation.batch_evaluation \
    --input-folder /path/to/validation_samples \
    --output-file results/batch_eval.json \
    --datasets 2wiki bamboogle hotpotqa musique
```

### 完整参数说明

```bash
python -m evaluation.batch_evaluation \
    --input-folder /path/to/validation_samples \     # 输入文件夹（包含answer_*.json文件）
    --output-file results/batch_eval.json \          # 输出JSON文件路径
    --datasets 2wiki bamboogle hotpotqa musique \    # 要评测的数据集列表
    --model-path /path/to/model \                    # vLLM模型路径（可选）
    --dataset-path gsm8k=/path/to/gsm8k.jsonl \      # 自定义数据集路径（可选）
    --file-pattern "answer_*.json" \                 # JSON文件匹配模式（可选）
    --no-skip-existing                               # 不跳过已评测文件（可选）
```

### 支持的数据集

- **QA数据集**: 2wiki, bamboogle, hotpotqa, musique
- **数学数据集**: gsm8k, math500, aime24, aime25
- **Agent数据集**: HLE, GAIA, WebWalker, XBench

### 实际使用示例

#### 示例1: 评测reasoning相关数据集

```bash
python -m evaluation.batch_evaluation \
    --input-folder outputs/checkpoints/grpo_reasoning/validation_samples \
    --output-file evaluation/results/grpo_reasoning_eval.json \
    --datasets 2wiki bamboogle hotpotqa musique \
    --model-path /path/to/Qwen3-32B
```

#### 示例2: 评测数学数据集

```bash
python -m evaluation.batch_evaluation \
    --input-folder outputs/checkpoints/dapo_qwen3_4b/validation_samples \
    --output-file evaluation/results/dapo_math_eval.json \
    --datasets gsm8k math500 aime24 aime25 \
    --model-path /path/to/Qwen3-32B
```

#### 示例3: 使用自定义数据集路径

```bash
python -m evaluation.batch_evaluation \
    --input-folder /path/to/validation_samples \
    --output-file results/custom_eval.json \
    --datasets gsm8k math500 \
    --dataset-path gsm8k=/custom/path/to/gsm8k.jsonl \
    --dataset-path math500=/custom/path/to/math500.jsonl \
    --model-path /path/to/model
```

## 特性说明

### 1. 断点续评

如果评测过程中断，重新运行脚本时会：
- 自动读取已有的评测结果
- 跳过已完成的文件
- 只评测未完成的文件

如果想重新评测所有文件，使用 `--no-skip-existing` 参数。

### 2. 实时保存

每完成一个JSON文件的评测，结果会立即保存到输出文件中，防止意外中断导致数据丢失。

### 3. 自动摘要

评测完成后会自动打印摘要信息：

```
评测摘要
================================================================================

文件: answer_step_100.json
  - 2wiki: acc=0.9000
  - bamboogle: acc=0.8500
  - hotpotqa: acc=0.8750
  - musique: acc=0.8200

文件: answer_step_200.json
  - 2wiki: acc=0.9200
  - bamboogle: acc=0.8700
  ...
```

### 4. 灵活的文件匹配

默认匹配 `answer_*.json`，可以通过 `--file-pattern` 自定义：

```bash
# 匹配所有JSON文件
--file-pattern "*.json"

# 匹配特定前缀
--file-pattern "validation_*.json"

# 匹配特定命名模式
--file-pattern "answer_step_*.json"
```

## 批量评测多个文件夹

如果需要评测多个文件夹，推荐使用 shell 脚本循环调用：

```bash
# 参考 eva_scripts/batch_eval_example.sh

FOLDERS=(
    "/path/to/folder1/validation_samples"
    "/path/to/folder2/validation_samples"
    "/path/to/folder3/validation_samples"
)

for folder in "${FOLDERS[@]}"; do
    folder_name=$(basename "$(dirname "$folder")")
    python -m evaluation.batch_evaluation \
        --input-folder "$folder" \
        --output-file "results/batch_eval_${folder_name}.json" \
        --datasets 2wiki bamboogle hotpotqa musique \
        --model-path /path/to/model
done
```

## 输出格式详解

输出的JSON文件结构：

```json
{
  "answer_step_100.json": {
    "2wiki": {
      "overall": {
        "no_answer": 0,        // 未找到答案的数量
        "boxed_not_found": 2,  // 未找到boxed标记的数量
        "correct": 45,         // 正确答案的数量
        "false_answer": 3,     // 错误答案的数量
        "acc": 0.9             // 准确率
      }
    },
    "bamboogle": { ... }
  },
  "answer_step_200.json": { ... }
}
```

对于有子类别的数据集（如HLE、GAIA），输出会包含每个子类别的结果：

```json
{
  "answer_hle.json": {
    "HLE": {
      "Math": {
        "no_answer": 1,
        "boxed_not_found": 3,
        "correct": 85,
        "false_answer": 11,
        "acc": 0.85
      },
      "Physics": { ... },
      "Computer Science/AI": { ... }
    }
  }
}
```

## 常见问题

### Q: 如何只评测特定的JSON文件？

A: 使用 `--file-pattern` 参数指定更精确的匹配模式：

```bash
--file-pattern "answer_step_100.json"  # 只评测这一个文件
```

### Q: 如何处理评测失败的数据集？

A: 评测失败的数据集会在结果中标记为 `{"error": "错误信息"}`，不会影响其他数据集的评测。

### Q: 如何加速评测？

A:
1. 使用更少的数据集（只评测需要的数据集）
2. 使用 `--no-skip-existing` 重新评测时会跳过已完成的部分
3. 考虑并行运行多个评测任务（不同的输出文件）

### Q: 临时文件保存在哪里？

A: 临时文件保存在输出文件同级目录的 `temp_evaluation/` 文件夹中，评测完成后可以手动删除。

## 技术细节

### 与原有评测流程的兼容性

`batch_evaluation.py` 完全基于现有的评测框架构建：
- 使用 `evaluation.py` 中的 `prepare_answer_file()` 和 `evaluate_dataset()`
- 使用 `gpt_judge.py` 中的 `MathLLM` 进行判分
- 与原有的数据集配置 `_build_dataset_map()` 完全兼容

### 性能优化

- 判分器 (MathLLM) 在整个评测过程中只初始化一次
- 支持vLLM的批次推理，提高评测速度
- 实时保存结果，避免重复计算

## 迁移指南

如果你之前使用 `baseline_reason.sh`，迁移到新工具非常简单：

**之前:**
```bash
# baseline_reason.sh 会为每个文件夹生成多个evaluation_*.json文件
bash evaluation/eva_scripts/baseline_reason.sh
```

**现在:**
```bash
# 生成单个统一的JSON文件
python -m evaluation.batch_evaluation \
    --input-folder /path/to/validation_samples \
    --output-file results/batch_eval.json \
    --datasets 2wiki bamboogle hotpotqa musique gsm8k math500 aime24 aime25 \
    --model-path /path/to/model
```

## 后续分析

评测完成后，可以使用Python轻松分析结果：

```python
import json

# 读取结果
with open('results/batch_eval.json', 'r') as f:
    results = json.load(f)

# 分析特定文件的结果
file_results = results['answer_step_100.json']
for dataset, metrics in file_results.items():
    if 'overall' in metrics:
        print(f"{dataset}: {metrics['overall']['acc']:.4f}")

# 比较不同checkpoint的性能
for filename, datasets in results.items():
    print(f"\n{filename}:")
    for dataset, metrics in datasets.items():
        if 'overall' in metrics:
            print(f"  {dataset}: {metrics['overall']['acc']:.4f}")
```

## 测试

我们提供了完整的测试套件来验证关键功能：

### 运行测试

```bash
cd /path/to/SEARL

# 运行测试套件
python -m evaluation.test_batch_eval
```

### 测试内容

测试套件包含4个测试：

1. **测试1: Judge模型全局复用**
   - 验证多个Judge实例共享同一个LLM
   - 确保模型只加载一次

2. **测试2: 批处理功能**
   - 验证批处理vs单个处理的性能差异
   - 测量加速比
   - 确认结果一致性

3. **测试3: BatchEvaluator中judge的复用**
   - 验证BatchEvaluator正确传递judge实例
   - 确保evaluate_dataset接收并使用外部judge

4. **测试4: evaluation.py中的批处理调用**
   - 验证evaluation_metrics_HLE等函数使用批处理
   - 确认调用evaluation_batch而非单个evaluation

### 预期输出

```
================================================================================
# 批量评测功能测试套件
================================================================================

测试1: 验证Judge模型只初始化一次
================================================================================
✓ 测试通过: 两个Judge共享同一个LLM实例（模型只加载一次）

测试2: 验证批处理功能
================================================================================
批处理耗时: 2.35秒
批处理加速比: 5.12x
✓ 结果一致性: ✓ 一致

测试3: 验证BatchEvaluator中Judge的复用
================================================================================
✓ 测试通过: BatchEvaluator初始化时会创建全局Judge实例

测试4: 验证evaluation.py中的批处理调用
================================================================================
✓ 测试通过: evaluation_metrics_HLE 使用批处理进行评测

================================================================================
测试总结
================================================================================
✓ 通过: judge_single_instance
✓ 通过: batch_processing
✓ 通过: batch_evaluator_reuse
✓ 通过: evaluation_batch

总计: 4/4 测试通过

🎉 所有测试通过！批量评测功能正常工作。

关键特性:
  1. ✓ Judge模型只加载一次，所有评测复用同一实例
  2. ✓ 批处理功能正常，提供显著的性能提升
  3. ✓ BatchEvaluator正确传递judge实例
  4. ✓ evaluation.py中的评测函数使用批处理
```

## 性能监控

运行批量评测时，会看到以下信息：

```
================================================================================
正在初始化Judge模型（全局复用，避免重复加载）...
模型路径: /path/to/model
================================================================================

Loading model weights...
✓ Judge模型初始化完成，将在所有评测中复用此实例

在 /path/to/validation_samples 中找到 5 个JSON文件

================================================================================
正在评测文件: answer_step_100.json
================================================================================

--- 数据集: 2wiki ---
[prepare] matched 50/50 -> /tmp/temp_evaluation/answer_step_100/2wiki/answer_2wiki.json
[evaluation] 结果写入 /tmp/temp_evaluation/answer_step_100/2wiki/evaluation_2wiki.json
✓ 2wiki 评测完成

--- 数据集: bamboogle ---
✓ bamboogle 评测完成

...

================================================================================
所有评测完成！结果已保存到: evaluation/results/batch_eval.json
================================================================================
```

注意：
- Judge模型只在开始时加载一次
- 每个数据集评测时都会显示"复用judge实例"的提示
- 批处理会自动应用于所有评测函数
