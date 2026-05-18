# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from lm-evaluation-harness/lm_eval/tasks/hendrycks_math/utils.py at main · EleutherAI/lm-evaluation-harne
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
    # Check if <thinking><thinking>, <answer></answer> is paired
    if text.count('<thinking>') != text.count('<thinking>'):
        return False, "<thinking> <thinking> not paired"

    if text.count('<subtask>') != text.count('</subtask>'):
        return False, "<subtask> </subtask> not paired"

    if text.count('<thinking>') == 0 or text.count('<thinking>') == 0:
        return False, "<thinking> or <thinking> not found"

    # if text.count('<answer>') <= 1 or text.count('</answer>') <= 1:
    #     return False, "<answer> or </answer> not found"

    # Check the order of python/result  # todo_sxh ？
    if text.count('<python>') == 0 or text.count('</python>') == 0:
        return False, "<python> or </python> not found"

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
    matches = re.findall(pattern, text, re.DOTALL)  # 找所有匹配
    if not matches:
        return None

    return matches[-1].strip()  # 返回最后一个匹配

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


def compute_score(solution_str, ground_truth, extra_info) -> float:
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0
    }
    response = solution_str
    valid_template, reason = validate_format(response) 
    
    if not valid_template:
        # print(f"--------------------------------bad format: {reason}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        # result["score"] = -1
        # result["reason"] = f"bad format: {reason}"
        # return result["score"]
        format_reward = -0.5
    else:
        format_reward = 0.5
        
    # Remove EOS token if present # todo_sxh 我们没有 tokenizer
    if extra_info is not None and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]
    
    answer_part = extract_answer(response)
    if answer_part is None:
        print(f"--------------------------------cannot extract answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result["score"]
    
    try:
        answer = remove_boxed(last_boxed_only_string(answer_part))
        result["answer"] = answer
    except Exception as e:
        print(f"--------------------------------find box error: {e}--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"find box error: {e}"
        return result["score"]
    
    math_score = is_equiv(answer, ground_truth)
    result["f1_score"] = math_score
    print(f"f1_score: {math_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    if math_score > 0 and "</python>" in response:
        print(f"--------------------------------correct answer with multi tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = math_score + 0.1 + format_reward
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {math_score + 0.1}"
    elif math_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = math_score + format_reward
        result["reason"] = f"correct answer, get f1 score: {math_score}"
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0 + format_reward
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result["score"]
    # try:
    #     string_in_last_boxed = last_boxed_only_string(solution_str) ### todo 返回 \boxer{answer} or None
    #     if string_in_last_boxed is not None:
    #         answer = remove_boxed(string_in_last_boxed) ### todo 返回 answer
    #         if is_equiv(answer, ground_truth):
    #             retval = 1.0
    # except Exception as e:
    #     print(e)
    #     return 0.0

    # return retval


# string normalization from lm-evaluation-harness/lm_eval/tasks/hendrycks_math.py at master · EleutherAI/lm-evaluation-harness
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
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

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string