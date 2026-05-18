# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from envs import TOOL_ENV_REGISTRY
from omegaconf import OmegaConf
from transformers import AutoConfig

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn

from .dapo_ray_trainer import RayDAPOTrainer


def get_custom_reward_fn(config):
    import importlib.util

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))



@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer

        print(f"TaskRunner hostname: {socket.gethostname()}")
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # instantiate tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # 检查是否使用集中式工具管理
        centralized_tool_actor = None
        tool_manager_name = config.actor_rollout_ref.env.get("tool_manager", "qwen3")
        if not tool_manager_name:
            tool_manager_name = "adaptive"
        if tool_manager_name.startswith("centralized_"):
            from envs.tool_manager.centralized.centralized_qwen3_manager import CentralizedToolActor

            # 在主进程中创建集中式工具Actor
            try:
                centralized_tool_actor = ray.get_actor("centralized_tool_actor")
                print("- main dapo: 发现已存在的集中式工具Actor")
            except ValueError:
                print("- main dapo: 在trainer中创建集中式工具Actor")
                centralized_tool_actor = CentralizedToolActor.options(
                    name="centralized_tool_actor",
                    max_concurrency=config.actor_rollout_ref.env.get("max_concurrency", 10),
                ).remote(config.actor_rollout_ref.env)

        model_config = AutoConfig.from_pretrained(local_path)
        config.actor_rollout_ref.env.model_type = getattr(model_config, "model_type", "unknown")

        env_object = TOOL_ENV_REGISTRY[config.actor_rollout_ref.env.name](
            config=config.actor_rollout_ref.env, centralized_actor=centralized_tool_actor
        )

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        elif reward_manager_name == "parallel":
            from verl.workers.reward_manager import AsyncRewardManager

            reward_manager_cls = AsyncRewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        # AsyncRewardManager 不接受 max_resp_len/overlong_buffer 参数，其他 RM 需要
        if reward_manager_name == "parallel":
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
            )
        else:
            reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=0,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
            )
        # 让奖励管理器绑定环境（如 parallel 需要 env.compute_score），与 main_ppo 一致处理
        try:
            reward_fn.set_env_object(env_object)
        except Exception as e:
            print(f"Error setting env object for reward_fn: {e}")
        try:
            # stop_token 供工具/验证截断使用
            reward_fn.stop_token = config.actor_rollout_ref.rollout.stop[0]
        except Exception:
            pass

        # Note that we always use function-based RM for validation
        if reward_manager_name == "parallel":
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
            )
        else:
            val_reward_fn = reward_manager_cls(
                tokenizer=tokenizer,
                num_examine=1,
                compute_score=compute_score,
                reward_fn_key=config.data.reward_fn_key,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.overlong_buffer,
            )
        try:
            val_reward_fn.set_env_object(env_object)
            val_reward_fn.if_val = True
        except Exception as e:
            print(f"Error setting env object for val_reward_fn: {e}")
        try:
            val_reward_fn.stop_token = config.actor_rollout_ref.rollout.stop[0]
        except Exception:
            pass
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 数据集与采样器
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, is_train=True, env_object=env_object
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, is_train=False, env_object=env_object
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            env_object=env_object,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
