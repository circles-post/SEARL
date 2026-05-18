from sympy import *
from sympy.core.function import AppliedUndef
from sympy.core.numbers import Pi, Exp1,I,Infinity,NegativeInfinity
import numpy as np
import timeout_decorator
from extended_zss import ext_distance
from latex_pre_process import *
from sympy.simplify import *

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
    # Check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"

    if text.count('<subtask>') != text.count('</subtask>'):
        return False, "<subtask> </subtask> not paired"

    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"

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


"""
Guide:
You only need to use EED and install the following packages:
- sympy
- numpy
- latex2sympy2_extended
- timeout_decorator
"""

"""
There are four main categories:

Constants: such as integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems (e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition, multiplication, and exponentiation.
"""
# The costs can be modified if you think their values are different
insert_cost={"number":1,"symbol":1,"operator":1,"function":1}
delete_cost={"number":1,"symbol":1,"operator":1,"function":1}
update_cost={"number":1,"symbol":1,"operator":1,"function":1}

change_type_cost=1 #the cost of an update between different types,can be set to higher

bar_size=5 # the minimum size of triggering cluster discount
discount_slope=0.6 #discount

simplify_time_limit=30 #set the time limit of simplify
equals_time_limit=10 #set the time limit of equals

def update_func(x,y):
    
    if x.label==y.label:
        return 0
    
    elif x.label.split("_")[0]==y.label.split("_")[0]:
        return update_cost[x.label.split("_")[0]]
    return change_type_cost
def remove_func(x):
    return delete_cost[x.label.split("_")[0]]

def remove_tree_func(x):
    if not x.children:
        return remove_func(x)
    s=calc_tree_size(x)
    return min(s,discount_slope*(s-bar_size)+bar_size)


def insert_func(x):
    return insert_cost[x.label.split("_")[0]]
def insert_tree_func(x):
    return remove_tree_func(x)



def calc_tree_size(node):
    """
    Calculate the size of a subtree based on its total insertion cost.
    The function computes the size of a subtree by summing up the insertion 
    costs of the current node and all its descendant nodes. If the subtree 
    size has already been calculated and stored in `node.subtree_size`, it 
    returns the cached value to avoid redundant computation.
    Args:
        node (Node): The root node of the subtree for which the size is to 
                     be calculated
    Returns:
        int: The total size of the subtree, calculated as the sum of the 
             insertion costs of the current node and all its descendants.
    Notes:
        - The `insert_cost` dictionary is assumed to be globally defined 
          and maps node labels to their respective insertion costs.
        - The function modifies the `subtree_size` attribute of the input 
          node to store the calculated subtree size for future use.
    """
    """The size of a subtree equals to its total insertion cost"""
    
    total = insert_cost[node.label.split("_")[0]]
    
    if node.children and node.subtree_size !=0:

        return node.subtree_size
    
    for child in node.children:
        total += calc_tree_size(child)
    
    node.subtree_size=total

    return total
"""
Scoring function from relative distance
"""
def score_calc(tree_dist,tree_size):

    if tree_dist==0.:
        return 100
    return max(0,100*discount_slope-100*tree_dist/tree_size)




@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def simplify_with_timeout(expr):
    return simplify(expr)
def time_simplify(expr):
    try:
        result=simplify_with_timeout(expr)
        return result
    except TimeoutError:
        return expr

@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def equal_with_timeout(expr1,expr2):
    return expr1.equals(expr2)
def time_equal(expr1,expr2):
    try:
        result=equal_with_timeout(expr1,expr2)
        return result
    except TimeoutError:
        return False


def sympy_to_tree(expr):
    """
    Convert a SymPy expression into a tree structure.
    This function takes a SymPy expression and recursively converts it into a tree
    representation using `TreeNode` objects. Each node in the tree is labeled based
    on the type of the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity
        - Symbols: Symbol
        - Binary Operators: Add, Mul, Pow
        - Functions: Any subclass of `sympy.Function`
    Example:
        >>> from sympy import symbols, sin, pi
        >>> x, y = symbols('x y')
        >>> expr = x + y * sin(pi)
        >>> tree = sympy_to_tree(expr)
        >>> print(tree)
    """
    #print(expr)

    """Convert the sympy expression to a tree"""
    # Symbols and constants
    if isinstance(expr, (Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity)):
        return TreeNode(label="number_"+str(expr), children=[])
    elif isinstance(expr, (Symbol,)):

        return TreeNode(label="symbol_"+str(expr),children=[])

    
    # Binary operators
    elif isinstance(expr, (Add, Mul, Pow)):

        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="operator_"+op_name, children=children)
    

    elif isinstance(expr, (Function)):
        # Functions

        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="function_"+func_name, children=children)

    else:
        #print(expr)
        print(f"Unsupported Sympy type: {type(expr).__name__}, Expression: {expr}")
        raise ValueError(f"Unsupported SymPy type: {type(expr)}")

class TreeNode:
    def __init__(self, label, children=None,node_type='other'):
        self.label = label
        self.children = children if children is not None else []
        self.node_type=node_type
        self.subtree_size=0
    def get_children(self):
        return self.children
    
    def __str__(self):
        return self.label




def print_tree(node, indent=0):
    """Print a tree structure"""
    print('  ' * indent + f'└─ {node.label}')
    for child in node.children:
        print_tree(child, indent + 1)



import timeout_decorator

class LaTeXError(Exception):
    def __init__(self, message="LaTeXError"):
        super().__init__(message)
class SymPyError(Exception):
    def __init__(self, message="SymPyError"):
        super().__init__(message)


class TreeError(Exception):
    def __init__(self, message="TreeError"):
        super().__init__(message)


class DistError(Exception):
    def __init__(self, message="DistanceError"):
        super().__init__(message)

def EED(answer_latex,test_latex,debug_mode=False):  ### 相当于 compute_score
    """
        Computes the similarity score and distance metrics between two LaTeX expressions.
        This function evaluates the equivalence of two mathematical expressions represented 
        in LaTeX format. It uses symbolic computation and tree-based distance metrics to 
        calculate a similarity score and other related metrics.
    
            tuple: A tuple containing the following elements:
                - score (float): The similarity score between the two expressions (0 to 100).
                - relative_distance (float): The normalized distance between the two expressions.
                - answer_tree_size (int): The size of the expression tree for the answer.
                - distance (float): The raw distance between the two expression trees.
        Notes:
            - If either input contains unsupported LaTeX constructs (e.g., integrals or sums), 
              the function returns default values indicating failure.
            - If the test expression is significantly longer than the answer expression, 
              the function assumes they are not equivalent.
            - The function uses symbolic simplification and tree-based distance metrics to 
              evaluate equivalence.
            - In case of errors during processing, the function returns default values unless 
              `debug_mode` is enabled, in which case it raises specific exceptions.
        Exceptions:
            - LaTeXError: Raised when LaTeX conversion to symbolic expressions fails (if `debug_mode` is True).
            - SymPyError: Raised when symbolic simplification or tree construction fails (if `debug_mode` is True).
            - DistError: Raised when distance calculation fails (if `debug_mode` is True).
        Args:
            answer_latex: the latex expression of answer expression
            test_latex: the latex expression of test expression
            debug_mode: whether it raise errors or just skip it
        Returns:
             tuple: A tuple containing the following elements:
                - score (float): The similarity score between the two expressions (0 to 100).
                - relative_distance (float): The normalized distance between the two expressions.
                - answer_tree_size (int): The size of the expression tree for the answer.
                - distance (float): The raw distance between the two expression trees.
    """

    if not test_latex:
        return 0,-1,-1,-1
    if '\\int' in test_latex or '\\int' in answer_latex:
        return 0,-1,-1,-1
    if '\\sum' in test_latex or '\\sum' in answer_latex:
        return 0,-1,-1,1
    if answer_latex==test_latex:
        return 100,0.0,-1,0
    if len(test_latex)>3*len(answer_latex):
        return 0,-1,-1,-1

    try:

        answer_exp=master_convert(answer_latex)
        test_exp=master_convert(test_latex)
    except:
        print(f"Failed to convert input latex to sympy expression,please check it")
        if debug_mode:
            raise LaTeXError(f"Fail to convert latex.\n GT:{answer_latex}\n GEN:{test_latex}")
        return 0,-1,-1,-1

    try:

        answer_exp,rep1=posify(answer_exp)
        
        answer_exp=time_simplify(answer_exp)
        
        
        test_exp,rep2=posify(test_exp)
        test_exp=time_simplify(test_exp)

        
        
        answer_exp=answer_exp.subs(rep1)
        test_exp=test_exp.subs(rep2)

        zero_exp=time_simplify(expand(answer_exp-test_exp))
        

        if answer_exp==test_exp or zero_exp==0:
            return 100,0.,0,0

        if time_equal(answer_exp,test_exp):
            return 100,0.,0,0

    except:
        print("Something happened during simplification,returning zero")
        if debug_mode:
            raise SymPyError(f"Failed to simplify the sympy expression. Expressions: answer_exp={answer_exp}, test_exp={test_exp}")
        return 0,-1,-1,-1

    try:
        tree_answer=sympy_to_tree(answer_exp)
        tree_test=sympy_to_tree(test_exp)

    except:
        
        print("Failed to build expression tree,returning zero")
        if debug_mode:
            raise SymPyError(f"Failed to build the sympy expression tree.\n GT:{answer_exp}\n GEN:{test_exp}")
        return 0,-1,-1,-1

    distance=ext_distance(
                tree_test,
                tree_answer,
                get_children=lambda x:x.get_children(),
                single_insert_cost=insert_func,
                insert_cost=insert_tree_func,
                single_remove_cost=remove_func, 
                remove_cost=remove_tree_func, 
                update_cost=update_func)    
    try:
        

        distance=ext_distance(
                tree_test,
                tree_answer,
                get_children=lambda x:x.get_children(),
                single_insert_cost=insert_func,
                insert_cost=insert_tree_func,
                single_remove_cost=remove_func, 
                remove_cost=remove_tree_func, 
                update_cost=update_func
            )
    except:
        print("Failed to calculate distance")
        if debug_mode:
            raise DistError(f"Failed to calculate the distance between trees.\n GT:{answer_latex}\n GEN:{test_latex}")
        return 0,-1,calc_tree_size(tree_answer),-1
    tree_size=calc_tree_size(tree_answer)
    distance_number=distance

    rel_distance=distance/tree_size

    score=score_calc(distance_number,tree_size)

    return score,rel_distance,tree_size,distance_number


def compute_score_EED(solution_str, ground_truth, extra_info):
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
    
    EED_score = EED(answer, ground_truth)[0] / 100
    result["f1_score"] = EED_score
    print(f"f1_score: {EED_score}, answer: {answer}, ground_truth: {ground_truth}")
    
    if EED_score > 0 and "</python>" in response:
        print(f"--------------------------------correct answer with multi tool call--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = EED_score + 0.1 + format_reward
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {EED_score + 0.1}"
    elif EED_score > 0:
        print(f"--------------------------------correct answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = EED_score + format_reward
        result["reason"] = f"correct answer, get f1 score: {EED_score}"
    else:
        print(f"--------------------------------wrong answer--------------------------------\nsolution_str: {solution_str}, ground_truth: {ground_truth}")
        result["score"] = 0 + format_reward
        result["reason"] = f"wrong answer but good format: {answer}"
    
    return result["score"]