# SEARL

**Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents**

SEARL is the official codebase for the paper **"SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents."** It focuses on reinforcement learning for self-evolving agents that can call tools, build reusable tool graph memory, and improve policy behavior through iterative interaction.

> Code release and documentation cleanup are in progress. The core implementation is being prepared for public use.

## Highlights

- **Tool-augmented RL training**: supports multi-turn tool use during rollout.
- **Tool graph memory**: maintains generated tools and their relationships for reuse.
- **Multiple tool backends**: supports search, Python execution, and MCP-style tool creation.
- **veRL-based training stack**: reuses the distributed RL infrastructure from [veRL](https://github.com/volcengine/verl).

## Repository Structure

```text
SEARL/
├── envs/                 # RL environments, tool managers, and tool definitions
│   └── tools/            # Built-in tools: search, Python execution, MCP creation
├── rag_server/           # Local retrieval server for Wikipedia-style search
├── verl/                 # veRL-based training components
├── evaluation/           # Evaluation utilities
├── docs/                 # Extended documentation
└── tests/                # Unit and integration tests
```

## Tools

SEARL currently includes three core tool categories under [`envs/tools`](envs/tools):

1. **Search Tool**

   The search tool supports both local and web search modes.

   - **Local Wikipedia search**: implemented with a local retrieval service under [`rag_server`](rag_server). This setup follows the retrieval-style training idea used in Search-R1.
   - **Web search**: can be connected to a SERP provider such as Bright Data through environment-specific configuration.

2. **Python Tool**

   The Python execution tool is designed to run code in an isolated sandbox. It follows the interface style of [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/category/reference).

3. **MCP Creation Tool**

   The MCP creation tool extends the Python execution workflow by creating reusable MCP-style tools that can be added to the agent's tool memory.

## Installation

```bash
git clone https://github.com/circles-post/SEARL.git
cd SEARL

pip install -r requirements.txt
pip install -e . --no-deps
```

## Training Entry Point

The main cleaned training entrypoint is:

```bash
bash main_searl.sh
```

Common runtime variables can be overridden without editing the script:

```bash
MODEL_PATH=models/qwen3 \
REWARD_MODEL_PATH=models/reward_model \
TRAIN_FILES=data/train.parquet \
VAL_FILES='[data/validation.parquet]' \
bash scripts/rebuttal/epoch1.sh
```

For custom experiments, update the dataset paths, model paths, and tool configuration files according to your environment.

## Local Retrieval Server

The local search backend lives in [`rag_server`](rag_server). A typical workflow is:

```bash
cd rag_server
bash launch.sh
```

Make sure the corpus, index, and retriever model paths are configured before starting the server.

## Citation

If you find SEARL useful, please cite:

```bibtex
@article{feng2026searl,
  title={SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents},
  author={Feng, Xinshun and Song, Xinhao and Li, Lijun and Liu, Gongshen and Shao, Jing},
  journal={arXiv preprint arXiv:2604.07791},
  year={2026}
}
```
For any inquiries, please contact fengxinshun@pjlab.org.cn.
## Acknowledgement

This project builds on ideas and infrastructure from [veRL](https://github.com/volcengine/verl), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [RL-Factory](https://github.com/Simple-Efficient/RL-Factory), and [SandboxFusion](https://bytedance.github.io/SandboxFusion/docs/category/reference).

Reference for Search-R1:

> Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., et al. (2025). *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*. arXiv:2503.09516.
