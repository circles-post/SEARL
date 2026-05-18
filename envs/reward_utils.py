import re
import sys
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter
from transformers import AutoTokenizer
import sympy
import math

def verify_math_answer(answer_float: float, 
                       ground_truth_expr: str, 
                       rel_tol: float = 1e-4) -> bool:
    """
    Verifies if a float answer is mathematically close to a LaTeX expression (or equivalent forms).
    支持:
      - 纯数/表达式/LaTeX(\frac, \sqrt, ^, \cdot, \times, \left...\right)
      - 百分数: "12%" -> 0.12
      - 区间: (a,b), [a,b], (a,b], [a,b]
      - 多候选: {a,b,c}
      - \pm: 1 \pm 2  ->  {1+2, 1-2}
      - 传入 tuple/list/dict 时的鲁棒处理（从 dict 取 value/answer/target；二元容器视为区间，否则视为多候选）
    语义:
      - ground truth 是单值: 与 answer_float 做 isclose
      - ground truth 是区间: 判断 answer_float 是否落在区间（含开闭端点）
      - ground truth 是多候选: 任一候选匹配即为 True
    """
    # 基本健壮性
    if not isinstance(answer_float, (int, float)) or not math.isfinite(answer_float):
        return False
    x = float(answer_float)

    # 允许传入非 str（虽然注解是 str）
    gt = ground_truth_expr

    # ---- dict: 抽值 ----
    if isinstance(gt, dict):
        for k in ("value", "answer", "target"):
            if k in gt:
                gt = gt[k]
                break

    # ---- list/tuple: 2 元视为区间，其他视为多候选 ----
    if isinstance(gt, (list, tuple, set)):
        items = list(gt)
        if len(items) == 2:
            try:
                lo = _to_float(items[0])
                hi = _to_float(items[1])
                lo, hi = (lo, hi) if lo <= hi else (hi, lo)
                # Python 容器区间，默认开区间 (lo, hi)
                return _in_interval(x, lo, hi, left_closed=False, right_closed=False, rel_tol=rel_tol)
            except Exception:
                # 非纯数区间 -> 当作多候选
                pass
        # 多候选：任意一个匹配即可
        return any(verify_math_answer(x, it, rel_tol=rel_tol) for it in items)

    # 其余统一转成字符串
    s = str(gt).strip()

    # 直接数值（包含百分数）快速路径
    num = _parse_number_maybe_percent(s)
    if num is not None:
        return math.isclose(x, num, rel_tol=rel_tol)

    # \pm  -> 两个候选
    if r"\pm" in s:
        left, right = s.split(r"\pm", 1)
        cand1 = f"{left}+{right}"
        cand2 = f"{left}-{right}"
        return verify_math_answer(x, cand1, rel_tol) or verify_math_answer(x, cand2, rel_tol)

    # 多候选 {a,b,c}
    m_set = re.fullmatch(r"\s*\{(.+)\}\s*", s)
    if m_set:
        parts = [p.strip() for p in m_set.group(1).split(",")]
        return any(verify_math_answer(x, p, rel_tol) for p in parts)

    # 区间 [a,b] / (a,b) / (a,b] / [a,b)
    m_iv = re.fullmatch(r"\s*([\(\[])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\)\]])\s*", s)
    if m_iv:
        left_br, a_str, b_str, right_br = m_iv.groups()
        try:
            lo = _to_float(a_str)
            hi = _to_float(b_str)
            if lo > hi:
                lo, hi = hi, lo
            return _in_interval(
                x, lo, hi,
                left_closed=(left_br == "["),
                right_closed=(right_br == "]"),
                rel_tol=rel_tol
            )
        except Exception:
            # 如果端点不是纯数，继续走表达式解析
            pass

    # 单值表达式（含 LaTeX）
    try:
        val = _to_float(s)  # 尝试解析为数字/表达式后取数值
        return math.isclose(x, val, rel_tol=rel_tol)
    except Exception:
        return False


# ----------------- 辅助函数 -----------------

def _parse_number_maybe_percent(s: str):
    """尝试解析纯数字或百分数；成功返回 float，否则返回 None。"""
    s1 = s.strip().strip("$")
    # 12% 或 12 \%
    m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?)\s*(\\?%)\s*", s1)
    if m:
        return float(m.group(1)) / 100.0
    # 纯浮点
    try:
        return float(s1.replace(",", ""))
    except Exception:
        return None

def _in_interval(x: float, lo: float, hi: float, left_closed: bool, right_closed: bool, rel_tol: float) -> bool:
    # 给边界一点容差
    pad_lo = rel_tol * max(1.0, abs(lo))
    pad_hi = rel_tol * max(1.0, abs(hi))
    left_ok = (x + pad_lo) >= lo if left_closed else (x - pad_lo) > lo
    right_ok = (x - pad_hi) <= hi if right_closed else (x + pad_hi) < hi
    return left_ok and right_ok

def _to_float(expr: str) -> float:
    """
    把表达式/LaTeX 转成 float。
    尝试顺序：
      1) 直接 float
      2) sympy.parse_latex
      3) 轻量 LaTeX 预处理后 sympy.sympify
    """
    s = str(expr).strip().strip("$")
    # 直接 float
    try:
        return float(s.replace(",", ""))
    except Exception:
        pass

    # 百分数
    num = _parse_number_maybe_percent(s)
    if num is not None:
        return num

    # 优先 parse_latex（若可用）
    try:
        from sympy.parsing.latex import parse_latex as _parse_latex
        expr_obj = _parse_latex(s)
        if getattr(expr_obj, "is_number", False):
            return float(expr_obj.evalf())
        # 非纯数表达式，也尝试 evalf
        return float(expr_obj.evalf())
    except Exception:
        pass

    # 基础 LaTeX 预处理 -> sympify
    s2 = _preprocess_latex_basic(s)
    try:
        import sympy as sp
        expr_obj = sp.sympify(s2, rational=True)
        return float(expr_obj.evalf())
    except Exception:
        # 再尝试一次直接 float（预处理后）
        return float(s2)

def _preprocess_latex_basic(s: str) -> str:
    """
    轻量 LaTeX 归一：\frac, \sqrt, ^, \cdot, \times, \div, \left, \right, 花括号等
    """
    s = s.strip()
    # 去掉 \left \right 和薄空格
    for k in (r"\left", r"\right", r"\,", r"\!"):
        s = s.replace(k, "")
    # 常见运算符
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    # \sqrt{a} -> sqrt(a)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # 花括号 -> 圆括号
    s = s.replace("{", "(").replace("}", ")")
    # 幂 ^ -> **  (在合适位置)
    s = re.sub(r"(?<=[0-9a-zA-Z\)])\^(?=[0-9a-zA-Z\(])", "**", s)
    # 去掉多余反斜杠（保留必要的）
    s = s.replace("\\", "")
    return s


def validate_format(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.

    Args:
        text: The text to validate

    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <thinking></thinking>, <answer></answer> is paired
    if text.count('<tool_call>') != text.count('</tool_call>'):
        return False, "<tool_call> </tool_call> not paired"

    if text.count('<subtask>') != text.count('</subtask>'):
        return False, "<subtask> </subtask> not paired"
    else:
        if text.count('<subtask>') == 0 or text.count('</subtask>') == 0:
            return False, "<subtask> or </subtask> not found"

    # Check the order of python/result  # todo_sxh ？
    if text.count('<thinking>') == 0 or text.count('</thinking>') == 0:
       return False, "<thinking> or </thinking> not found"
    else:
        if text.count('<thinking>') == 0 or text.count('</thinking>') == 0:
            return False, "<thinking> or </thinking> not found"

    # Check if \boxed{} is in the answer
    answer_start = text.rfind('<answer>')
    answer_end = text.rfind('</answer>')

    if answer_start == -1 or answer_end == -1:
        return False, "<answer> or </answer> not found"

    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"

    answer_content = text[answer_start:answer_end]

    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def validate_format_baseline(text: str) -> Tuple[bool, str]:
    """
    Validate if the text follows the required format with paired tags.

    Args:
        text: The text to validate

    Returns:
        tuple: (is_valid, reason)
    """
    # Check if <thinking></thinking>, <answer></answer> is paired
    if text.count('<tool_call>') != text.count('</tool_call>'):
        return False, "<tool_call> </tool_call> not paired"

    # if text.count('<answer>') <= 1 or text.count('</answer>') <= 1:
    #     return False, "<answer> or </answer> not found"

    # Check the order of python/result  # todo_sxh ？
    # if text.count('<python>') == 0 or text.count('</python>') == 0:
    #    return False, "<python> or </python> not found"

    # Check if \boxed{} is in the answer
    answer_start = text.rfind('<answer>')
    answer_end = text.rfind('</answer>')

    if answer_start == -1 or answer_end == -1:
        return False, "<answer> or </answer> not found"

    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"

    answer_content = text[answer_start:answer_end]

    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"

    return True, "format is correct"


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the last <answer>...</answer> block from the text.

    Args:
        text: The text to extract answer from

    Returns:
        Optional[str]: The extracted answer or None if no match
    """
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)  
    if not matches:
        return None

    return matches[-1].strip()  

def remove_boxed(s: str) -> str:
    """
    Remove the LaTeX \boxed{} wrapper from the string.

    Args:
        s: String potentially containing \boxed{}

    Returns:
        str: String with \boxed{} removed
    """
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \boxed{} content from the string.

    Args:
        string: String to extract \boxed{} from

    Returns:
        Optional[str]: The extracted \boxed{} content or None if not found
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s: str) -> str:
    """
    Normalize the answer string by removing articles, white spaces, punctuation and converting to lowercase.

    Args:
        s: String to normalize

    Returns:
        str: Normalized string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    """
    Calculate F1 score between prediction and ground truths.

    Args:
        prediction: The predicted answer
        ground_truths: The ground truth answer(s)

    Returns:
        float: F1 score
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    for ground_truth in ground_truths:

        # --- START: NEW MATH VERIFICATION BLOCK ---
        try:
            # Step 1: Attempt to convert the prediction to a float.
            # If this fails, it's not a numeric answer, and a ValueError will be raised.
            pred_float = float(prediction)

            # Step 2: Call the math verification function.
            # If the ground truth is a valid math expression and close to the prediction,
            # we have a perfect match (F1 = 1.0).
            if verify_math_answer(pred_float, ground_truth):
                # Since we found a perfect match against one of the ground truths,
                # the max possible F1 score is 1.0, so we can return immediately.
                return 1.0
        except (ValueError, TypeError):
            # This triggers if float(prediction) fails or if verify_math_answer has an issue.
            # It means this is not a valid math comparison, so we'll fall through
            # to the standard text-based F1 score calculation below.
            pass
        # --- END: NEW MATH VERIFICATION BLOCK ---

        # Fallback to string normalization check
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            return 1.0


    # 统一使用浮点，避免 0 被打印成 False
    final_metric = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return float(final_metric['f1'])


def compute_score_baseline(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.

    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information

    Returns:
        Dict[str, Any]: A dictionary containing the score and detailed metrics for wandb tracking
    """

    # 初始化统一的返回结构 - 添加更多监控指标
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        # 新增：细分的reward指标用于wandb监控
        "format_reward": 0.0,
        "tool_penalty": 0.0,
        "answer_correctness": 0.0,  # -1 或 f1_score
    }

    response = solution_str
    valid_template, reason = validate_format_baseline(response)  ### todo_sxh 我们有 <tag> </tag> 吗？

    if not valid_template:
        # print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        # result["score"] = -1
        # result["reason"] = f"bad format: {reason}"
        # return result["score"]
        format_reward = -0.2
    else:
        format_reward = 0.2

    result["format_reward"] = format_reward

    # Remove EOS token if present # todo_sxh 我们没有 tokenizer
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    answer_part = extract_answer(response)  # todo_sxh <answer> </answer>
    if answer_part is None or answer_part == r'\boxed{[2, 1]}' or answer_part == r'\boxed{...}':
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        result["answer_correctness"] = -1.0
        return result  # 返回完整的字典而不仅仅是分数

    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        result["answer_correctness"] = -1.0
        return result  # 返回完整的字典而不仅仅是分数

    ### 能走到这里，说明有 answer
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    # Bonus for using multiple tools correctly
    # if f1_score > 0 and "</search>" in response and "</python>" in response:
    if f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score + format_reward
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
        result["answer_correctness"] = f1_score
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward
        result["reason"] = f"wrong answer but good format: {answer}"
        result["answer_correctness"] = -1.0

    return result  # 返回完整的字典，包含所有监控指标


def compute_score(solution_str: str, ground_truth: Any, extra_info: Optional[Dict[str, Any]] = None, mcp_worthiness: Optional[float] = None, tool_call_count: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute reward score for a solution based on the ground truth.

    Args:
        data_source: The data source identifier
        solution_str: The solution string to evaluate
        ground_truth: The ground truth answer(s)
        extra_info: Optional additional information
        mcp_worthiness: Optional MCP worthiness
        tool_call_count: Optional tool call count
    Returns:
        Dict[str, Any]: A dictionary containing the score and detailed metrics for wandb tracking
    """

    expected_tool_calls = None
    if tool_call_count is not None:
        try:
            expected_tool_calls = int(tool_call_count)
        except (TypeError, ValueError):
            expected_tool_calls = None

    # 初始化统一的返回结构 - 添加更多监控指标
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        # 新增：细分的reward指标用于wandb监控
        "format_reward": 0.0,
        "tool_penalty": 0.0,
        "answer_correctness": 0.0,  # -1 或 f1_score
        "num_tool_calls": 0,  # 实际工具调用次数
        "expected_tool_calls": expected_tool_calls if expected_tool_calls is not None else 0,  # 期望工具调用次数
    }
    # 统计工具调用次数（基于 <tool_call> ... </tool_call> 块），过多则惩罚
    response = solution_str
    penalty_per_extra = extra_info.get("tool_call_penalty", 0.05) if extra_info else 0.05
    tool_calls = re.findall(r"<tool_call>([\s\S]*?)</tool_call>", response)
    num_tool_calls = len(tool_calls) - 3 # remove the prompt <tool_call>
    result["num_tool_calls"] = num_tool_calls
    # assert mcp_worthiness is not None

    tool_penalty = 0.0
    if expected_tool_calls is None:
        # Keep reward neutral when expectation is missing/invalid.
        tool_penalty = 0.0
    elif abs(num_tool_calls - expected_tool_calls) > 1:
        tool_penalty = -penalty_per_extra * max(abs(num_tool_calls - expected_tool_calls) - 1, 0)
    else:
        tool_penalty = 0.1
    result["tool_penalty"] = tool_penalty
    print(f"tool_penalty: {tool_penalty}, num_tool_calls: {num_tool_calls}, tool_call_count: {expected_tool_calls}")
    valid_template, reason = validate_format(response)  ### todo_sxh 我们有 <tag> </tag> 吗？

    if not valid_template:
        format_reward = -0.1
    else:
        format_reward = 0.1
    result["format_reward"] = format_reward

    # Remove EOS token if present # todo_sxh 我们没有 tokenizer
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    answer_part = extract_answer(response)  # todo_sxh <answer> </answer>
    if answer_part is None or answer_part == r'\boxed{[2, 1]}' or answer_part == r'\boxed{...}':
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = "cannot extract answer"
        result["answer_correctness"] = -1.0
        return result  # 返回完整的字典而不仅仅是分数

    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = f"find box error: {e}"
        result["answer_correctness"] = -1.0
        return result  # 返回完整的字典而不仅仅是分数

    ### 能走到这里，说明有 answer
    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    # Bonus for using multiple tools correctly
    # if f1_score > 0 and "</search>" in response and "</python>" in response:
    if f1_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = f1_score * 1.5 + format_reward + tool_penalty
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
        result["answer_correctness"] = f1_score
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1 + format_reward + tool_penalty
        result["reason"] = f"wrong answer but good format: {answer}"
        result["answer_correctness"] = -1.0

    return result  # 返回完整的字典，包含所有监控指标  


if __name__ == "__main__":
    solution_str = "\frac{14}"
    ground_truth = "7"
    extra_info = None
    print(compute_score_baseline(solution_str, ground_truth))
