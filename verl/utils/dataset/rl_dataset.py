# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional
import yaml
import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from jinja2 import Template, StrictUndefined
logger = logging.getLogger(__name__)
def populate_template(template: str, variables) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

_plan_prompt_path = Path(
    os.environ.get(
        "RL_FACTORY_PLAN_PROMPT",
        Path(__file__).resolve().parents[3] / "prompt" / "plan.yaml",
    )
)
with open(_plan_prompt_path, "r") as file:
    prompt_templates = yaml.safe_load(file)

def initalize_plan_prompt() -> str:
    initial_plan_template = populate_template(
        prompt_templates["initial_plan_with_subtask"],
        variables={
            "task": "task",
            "max_turns": 5
        }
        )
    print(initial_plan_template,"initial_plan_template")
    return initial_plan_template

def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        env_object,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.env_object = env_object
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        #控制是否使用工具
        self.use_tool = config.get("use_tool", True)

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        # new_messages = []
        # for message in messages:
        #     if message["role"] == "system":
        #         content = initalize_plan_prompt()
        #         new_messages.append({"role": "system", "content": content})
        #     else:
        #         new_messages.append(message)
        # messages = new_messages
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            if self.use_tool and hasattr(self.env_object, "tool_manager") and hasattr(self.env_object.tool_manager, "get_prompt"):
                #  使用 tool_manager 提供的 prompt
                raw_prompt = self.env_object.tool_manager.get_prompt(messages, self.tokenizer, mode='initial', add_generation_prompt=True)
            else:
                # 使用 tokenizer 生成 prompt（禁用工具）
                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict and row_dict.get(self.video_key, None) is not None:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)



        else:
            if self.use_tool and hasattr(self.env_object, "tool_manager") and hasattr(self.env_object.tool_manager, "get_prompt"):
                #  使用 tool_manager 提供的 prompt
                raw_prompt = self.env_object.tool_manager.get_prompt(messages, self.tokenizer, mode='initial', add_generation_prompt=True)
            else:
                # 使用 tokenizer 生成 prompt（禁用工具）
                raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            #raw_prompt = self.env_object.tool_manager.get_prompt(messages, self.tokenizer, mode='initial', add_generation_prompt=True)
            # raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
    
def test_rlhf_dataset():
    """测试 RLHFDataset 的基本功能"""
    import sys
    import traceback
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from envs.mcp_base import MCPBaseEnv
    from envs.tool_manager.qwen3_manager import QwenManager

    print("🚀 开始测试 RLHFDataset...")
    from qwen_agent.llm.schema import ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem
    print(ASSISTANT, SYSTEM, USER, FUNCTION, ContentItem, "ASSISTANT SYSTEM USER FUNCTION ContentItem")
    try:
        # 1. 设置测试配置
        print("1. 设置测试配置...")
        config = OmegaConf.create({
            "cache_dir": "~/.cache/verl/test",
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "max_prompt_length": 2048,
            "return_raw_chat": False,
            "return_full_prompt": False,
            "truncation": "error",
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
            "use_shm": False,
            "use_tool": True,  # 启用工具以测试 tool_manager
            "serialize_dataset": False
        })

        # 2. 初始化tokenizer
        print("2. 初始化 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen3-8B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. 创建环境对象
        print("3. 创建环境对象...")
        # 为环境创建配置，包含工具管理器配置
        env_config = OmegaConf.create({
            'use_verify_tool': False,
            'tool_manager': 'qwen3',  # 使用 qwen3 工具管理器
            # 工具管理器的配置
            'config_path': 'envs/configs/mcp_tools_local.pydata',
            'mcp_mode': 'stdio',
            'enable_limiter': False,
            'tool_name_selected': [],
            'use_storage_manager': False,
            'enable_thinking': False,
            'local_search': True
        })

        # 创建工具管理器
        tool_manager = QwenManager(env_config)

        # 创建环境并手动设置工具管理器
        env_object = MCPBaseEnv(env_config)
        env_object.tool_manager = tool_manager


        # 4. 初始化数据集
        print("4. 初始化 RLHFDataset...")
        dataset = RLHFDataset(
            data_files=data_path,
            tokenizer=tokenizer,
            config=config,
            env_object=env_object,
            processor=None
        )

        print("✅ RLHFDataset 初始化成功")

        # 5. 测试基本属性
        print("5. 测试数据集基本属性...")
        print(f"   数据集大小: {len(dataset)}")
        print(f"   数据文件: {dataset.data_files}")
        print(f"   提示键: {dataset.prompt_key}")
        print(f"   最大提示长度: {dataset.max_prompt_length}")

        # 6. 测试获取第一个项目
        print("6. 测试获取第一个数据项目...")
        if len(dataset) > 0:
            try:
                first_item = dataset[0]
                print("✅ 成功获取第一个数据项目")

                # 显示项目的基本信息
                keys = list(first_item.keys())
                print(f"   项目包含的键: {keys}")
                print(first_item)
                print(tokenizer.decode(first_item["input_ids"]))

            except Exception as e:
                print(f"❌ 获取第一个数据项目失败: {str(e)}")
                traceback.print_exc()
        else:
            print("⚠️ 数据集为空")

        # 7. 测试多个项目（如果数据集足够大）
        print("7. 测试批量获取数据...")
        test_indices = [0, min(1, len(dataset)-1), min(2, len(dataset)-1)]
        for idx in test_indices:
            try:
                item = dataset[idx]
                print(f"   索引 {idx}: 获取成功")
            except Exception as e:
                print(f"   索引 {idx}: 获取失败 - {str(e)}")

        # 8. 测试数据集统计
        print("8. 数据集统计信息...")
        print(f"   总样本数: {len(dataset)}")
        print(f"   使用的缓存目录: {dataset.cache_dir}")
        print(f"   使用工具: {dataset.use_tool}")

        print("✅ RLHFDataset 测试完成")

    except Exception as e:
        print(f"❌ RLHFDataset 测试失败: {str(e)}")
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    import sys
    data_path = os.environ.get("RL_FACTORY_TEST_DATA", "data/train.parquet")

    # 运行测试
    success = test_rlhf_dataset()

    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
