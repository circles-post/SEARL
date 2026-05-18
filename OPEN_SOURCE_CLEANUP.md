# Open Source Cleanup Notes

本文档记录本次开源整理的范围、删除内容和运行入口调整，便于后续提交前复核。

## 保留的核心入口

主要训练入口保留为：

```bash
bash scripts/rebuttal/epoch1.sh
```

该脚本已改为使用仓库相对路径和环境变量，不再包含个人机器路径或明文 WandB key。常用覆盖参数如下：

```bash
MODEL_PATH=/path/to/model \
REWARD_MODEL_PATH=/path/to/reward_model \
TRAIN_FILES=data/train.parquet \
VAL_FILES='[data/val1.parquet,data/val2.parquet]' \
bash scripts/rebuttal/epoch1.sh
```

如需登录 WandB，运行前自行设置 `WANDB_API_KEY`；脚本默认 `WANDB_MODE=offline`。

## 已删除内容

- 独立代码评测目录、对应 MCP 工具脚本、veRL multi-turn 示例配置。
- 旧实验脚本目录：`scripts/MCPS/`、`scripts/reasoning/`、`scripts/deep_reasearch/`、`scripts/debug/`、`scripts/evaluation/`、`scripts/tool_num/`。
- 训练/评测生成物：`outputs/`、`tensorboard_log/`、`evaluation/results/`、`evaluation/eva_scripts/results/`、`recipe/dapo/logs/`。
- 生成的 MCP 工具和图缓存：`envs/tools/invented_tools/`、`envs/tools/graphs/`。
- 根目录一次性修复说明、case study 草稿、debug/test 脚本和本地日志。
- 本地代理/编辑器状态目录：`.claude/`；`.codex/` 和 `.agents/` 已加入 `.gitignore`。

## 已做的代码整理

- `scripts/rebuttal/epoch1.sh` 和 `epoch2.sh` 改为开源可配置入口。
- MCP `.pydata` 配置改为相对路径，并移除未使用的代码评测 server。
- 搜索缓存、GraphRAG embedding model、评测结果目录等改为环境变量或仓库相对默认值。
- 删除/替换仓库中的个人绝对路径和明文 WandB key。
- `.gitignore` 增补日志、缓存、生成 MCP 文件、图缓存、checkpoint、本地代理状态等开源不应提交的路径。

## 提交前建议

1. 运行 `git status --short` 确认只包含预期文件。
2. 若需要首个开源提交，执行 `git add .` 后再人工复核 `git status --short`。
3. 大文件、私有数据、模型 checkpoint、WandB/API key 不应加入 git。
