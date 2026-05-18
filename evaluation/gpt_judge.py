import os
from typing import Optional

from vllm import LLM, SamplingParams

"""
基于 vLLM 的开源模型判分器，默认使用开源权重（可通过环境变量/入参替换）。
环境变量：
- VLLM_MODEL_PATH：模型权重（HF 名称或本地路径），默认 Qwen/Qwen2.5-7B-Instruct
- VLLM_TP_SIZE：tensor parallel 大小，默认 1
- VLLM_GPU_MEM_UTIL：GPU 显存利用率，默认 0.9
"""

DEFAULT_MODEL = os.environ.get("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")


class VLLMJudge:
    """使用 vLLM 部署的开源模型做数值答案判分。"""

    _shared_llm: Optional[LLM] = None
    _shared_model_path: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 8,
        batch_size: int = 256,
    ):
        self.model_path = model_path or DEFAULT_MODEL
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._ensure_llm()

    def _ensure_llm(self):
        if (VLLMJudge._shared_llm is None) or (
            VLLMJudge._shared_model_path != self.model_path
        ):
            tp = int(os.environ.get("VLLM_TP_SIZE", "1"))
            gpu_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9"))
            VLLMJudge._shared_model_path = self.model_path
            VLLMJudge._shared_llm = LLM(
                model=self.model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_util,
            )

    @property
    def llm(self) -> LLM:
        return VLLMJudge._shared_llm

    def _build_prompt(self, prediction: str, reference: str, question: str) -> str:
        content = (
            f"question: {question}\n"
            f"prediction: {prediction}\n"
            f"reference: {reference}\n"
        )
        return [
            {"role": "system", "content": "You are a strict answer grader. Compare the prediction with the reference answer to determine if they convey the same information or answer. Consider:\n1. For numerical answers: check if they are mathematically equivalent (e.g., 0.5 = 1/2 = 50%)\n2. For text answers: check if the core meaning matches, ignoring minor formatting differences\n3. Ignore case, whitespace, and punctuation differences\n\nReturn ONLY:\n- '1' if prediction matches reference\n- '0' if prediction does NOT match reference\n\nOutput only a single digit: 1 or 0, nothing else."},
            {"role": "user", "content": content}
        ]

    def _call_llm(self, messages: list) -> str:
        """单样本推理（兼容旧接口）"""
        results = self._call_llm_batch([messages])
        return results[0] if results else ""

    def _call_llm_batch(self, messages_list: list) -> list:
        """批次推理"""
        if not messages_list:
            return []

        # 1. 获取 tokenizer
        tokenizer = self.llm.get_tokenizer()

        # 2. 应用聊天模板到所有消息
        prompts = []
        for messages in messages_list:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)

        # 3. 生成参数
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["\n"]
        )

        # 4. 批次生成
        outputs = self.llm.generate(prompts, params)

        # 5. 提取结果
        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")

        return results

    def judge(self, prediction: str, reference: str, question: str) -> float:
        """
        返回 1.0 / 0.0 表示是否匹配。
        """
        results = self.judge_batch([(prediction, reference, question)])
        return results[0] if results else 0.0

    def judge_batch(self, items: list) -> list:
        """
        批次判断多个样本
        items: list of (prediction, reference, question) tuples
        返回: list of float scores
        """
        if not items:
            return []

        # 构建所有prompt
        messages_list = []
        for prediction, reference, question in items:
            prompt = self._build_prompt(prediction, reference, question)
            messages_list.append(prompt)

        # 按batch_size分批处理
        all_results = []
        for i in range(0, len(messages_list), self.batch_size):
            batch_messages = messages_list[i:i + self.batch_size]
            try:
                batch_responses = self._call_llm_batch(batch_messages)
                batch_scores = []
                for resp in batch_responses:
                    # 严格匹配：只有精确输出"1"或"0"才认为是有效响应
                    resp_clean = resp.strip()
                    if resp_clean == "1":
                        batch_scores.append(1.0)
                    elif resp_clean == "0":
                        batch_scores.append(0.0)
                    else:
                        # 如果输出不是精确的"0"或"1"，尝试提取数字
                        # 这处理了模型可能输出 "Answer: 1" 等情况
                        if "1" in resp_clean and "0" not in resp_clean:
                            print(f"[VLLMJudge] Warning: non-standard response '{resp}', extracted as 1")
                            batch_scores.append(1.0)
                        elif "0" in resp_clean and "1" not in resp_clean:
                            print(f"[VLLMJudge] Warning: non-standard response '{resp}', extracted as 0")
                            batch_scores.append(0.0)
                        else:
                            # 完全无法解析，默认为0
                            print(f"[VLLMJudge] Error: unparseable response '{resp}', defaulting to 0")
                            batch_scores.append(0.0)
                all_results.extend(batch_scores)
            except Exception as e:
                print(f"[VLLMJudge] batch inference failed: {e}")
                # 失败时返回0分
                all_results.extend([0.0] * len(batch_messages))

        return all_results

    def evaluation(self, res: str, golden: str, question: str) -> float:
        """
        兼容旧接口，直接调用 judge。
        """
        return self.judge(res, golden, question)

    def evaluation_batch(self, items: list) -> list:
        """
        批次评估多个样本
        items: list of (res, golden, question) tuples
        返回: list of float scores
        """
        return self.judge_batch(items)


# 兼容旧调用名
MathLLM = VLLMJudge


def main_debug(prompt: str, reference: str, question: str):
    judge = VLLMJudge()
    score = judge.judge(prompt, reference, question)
    print("judge score:", score)


if __name__ == "__main__":
    demo_pred = "The answer is 42."
    demo_ref = "42"
    demo_q = "What is 6 * 7?"
    main_debug(demo_pred, demo_ref, demo_q)

