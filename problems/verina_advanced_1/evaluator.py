"""
Evaluator for Lean 4 theorem proving using Lean verification server.

使用 lean-code-prover 项目的 Lean 验证服务器进行评估。

评分标准:
- 证明完全成功（无 sorry）: 1.0
- 作弊: 0.0
- 一般情况 (combined_score = error_score + node_score):
  - error_score (占 0.2): 基于第一遍编译的错误数量
    - 0 个 error = 0.2
    - 20 个或以上 = 0
    - 线性递减: error_score = 0.2 * max(0, (20 - error_count) / 20)
  - node_score (占 0.8): 基于 revise 后的代码计算 node counts
    - revise 后仍然编译失败: 0
    - 否则用 compute_reward 计算

报告规则:
- 第一遍有语法错误: 报告语法错误
- 第一遍无语法错误: 报告 sorries
- revise 结果只用于打分，不用于报告
"""

import re
import sys
import math
import requests
from pathlib import Path
from openevolve.evaluation_result import EvaluationResult

# 添加 lean-code-prover 到 path
_LEAN_CODE_PROVER_ROOT = "/home/argustest/lizn/lean-code-prover"
if _LEAN_CODE_PROVER_ROOT not in sys.path:
    sys.path.insert(0, _LEAN_CODE_PROVER_ROOT)

from LeanCodeParser import LeanCodeTree
from revise import revise_proof as smart_revise_proof 

# Lean 验证服务器配置
LEAN_SERVER_URL = "http://localhost:8000"
TIMEOUT = 60
TIMEOUT_BUFFER = 600
MAX_RETRIES = 3

def verify_lean_code(code: str, timeout: int = TIMEOUT) -> dict:
    """
    向 Lean 验证服务器发送验证请求
    
    Args:
        code: Lean 代码
        timeout: 超时时间（秒）
        
    Returns:
        验证结果字典
    """
    url = f"{LEAN_SERVER_URL.rstrip('/')}/verify"
    payload = {
        "code": code,
        "timeout": timeout,
        "max_retries": 1,
        "overflow_threshold": 2,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=timeout + TIMEOUT_BUFFER)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        return {"error": "timeout", "is_valid": False}
    except requests.RequestException as e:
        return {"error": str(e), "is_valid": False}


# 标准公理列表（这些是 Lean/Mathlib 的合法公理）
STANDARD_AXIOMS = {
    'propext',           # 命题外延性
    'Quot.sound',        # 商类型公理
    'Classical.choice',  # 选择公理
    'funext',            # 函数外延性（在某些版本中）
    'Eq.ndrec',          # 等式消去
    'rfl',               # 自反性
    'sorryAx',           # sorry 内部使用的公理（不算作弊，但表示证明未完成）
}


def extract_node_scores(result: dict) -> list:
    """从验证结果中提取节点数分数"""
    node_scores = []
    if 'response' in result and 'response' in result['response']:
        messages = result['response']['response'].get('messages', [])
        for msg in messages:
            if msg.get('severity') == 'info' and 'data' in msg:
                data = msg['data']
                match = re.search(r'Total nodes \(goal \+ hypotheses\): (\d+) //', data)
                if match:
                    node_count = int(match.group(1))
                    node_scores.append(node_count)
    return node_scores


def extract_axioms_from_result(result: dict) -> list:
    """从验证结果中提取 #print axioms 输出的公理列表"""
    axioms = []
    if 'response' in result and 'response' in result['response']:
        messages = result['response']['response'].get('messages', [])
        for msg in messages:
            if msg.get('severity') == 'info' and 'data' in msg:
                data = msg['data']
                # #print axioms 输出格式: '[axiom1, axiom2, ...]' 或 'axiom1\naxiom2\n...'
                # 或者可能是 'axiom_name depends on axioms: [...]'
                # 提取所有可能的公理名
                # 格式1: 列表形式 [a, b, c]
                list_match = re.search(r'\[([^\]]+)\]', data)
                if list_match:
                    items = list_match.group(1).split(',')
                    axioms.extend([item.strip() for item in items if item.strip()])
                # 格式2: 每行一个公理名（以 '.' 分隔的标识符）
                for line in data.split('\n'):
                    line = line.strip()
                    if line and re.match(r'^[A-Za-z_][A-Za-z0-9_\.]*$', line):
                        axioms.append(line)
    return list(set(axioms))  # 去重


# 缓存原始定理声明
_ORIGINAL_THEOREM_CACHE = None


def get_original_theorem_info() -> tuple:
    """
    从 initial_program.lean 提取原始定理信息
    
    Returns:
        (theorem_name, theorem_params, theorem_type, theorem_call_args)
        例如: ("FindSingleNumber_spec_satisfied", 
               "(nums: List Int) (h_precond : FindSingleNumber_precond (nums))",
               "FindSingleNumber_postcond (nums) (FindSingleNumber (nums) h_precond) h_precond",
               "nums h_precond")
    """
    global _ORIGINAL_THEOREM_CACHE
    
    if _ORIGINAL_THEOREM_CACHE is not None:
        return _ORIGINAL_THEOREM_CACHE
    
    initial_program_path = Path(__file__).parent / "initial_program.lean"
    
    if not initial_program_path.exists():
        print(f"[Warning] initial_program.lean not found at {initial_program_path}")
        return None, None, None, None
    
    with open(initial_program_path, "r") as f:
        initial_code = f.read()
    
    # 提取 EVOLVE-BLOCK 中的 theorem 声明
    block_match = re.search(
        r'-- EVOLVE-BLOCK-START\s*([\s\S]*?)\s*-- EVOLVE-BLOCK-END',
        initial_code
    )
    
    if not block_match:
        print("[Warning] EVOLVE-BLOCK not found in initial_program.lean")
        return None, None, None, None
    
    block_content = block_match.group(1)
    
    # 匹配 theorem 声明: theorem Name ... := by
    # 首先找到 theorem 名称和完整声明
    theorem_match = re.search(
        r'theorem\s+(\w+)\s*([\s\S]*?)\s*:=\s*by',
        block_content
    )
    
    if not theorem_match:
        print("[Warning] theorem declaration not found in EVOLVE-BLOCK")
        return None, None, None, None
    
    theorem_name = theorem_match.group(1)
    full_signature = theorem_match.group(2).strip()
    
    # 找到最后一个 : 之后的类型部分（定理结论）
    # 使用括号平衡来找到真正的类型分隔符
    paren_depth = 0
    last_colon_pos = -1
    for i, ch in enumerate(full_signature):
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
        elif ch == ':' and paren_depth == 0:
            last_colon_pos = i
    
    if last_colon_pos == -1:
        print("[Warning] Cannot find type separator in theorem declaration")
        return None, None, None, None
    
    theorem_params = full_signature[:last_colon_pos].strip()
    theorem_type = full_signature[last_colon_pos + 1:].strip()
    
    # 从参数中提取调用参数 (去掉类型注解)
    # "(nums: List Int) (h_precond : FindSingleNumber_precond (nums))" -> "nums h_precond"
    # 匹配每个 ( 后面紧跟的标识符和冒号
    param_names = re.findall(r'\((\w+)\s*:', theorem_params)
    theorem_call_args = ' '.join(param_names)
    
    _ORIGINAL_THEOREM_CACHE = (theorem_name, theorem_params, theorem_type, theorem_call_args)
    print(f"[OriginalTheorem] name={theorem_name}, call_args={theorem_call_args}")
    
    return _ORIGINAL_THEOREM_CACHE


def check_axiom_cheating(code: str, theorem_name: str) -> tuple:
    """
    检查定理是否使用了非标准公理或改变了定理类型（作弊检测）
    
    在 EVOLVE-BLOCK-END 之后插入：
    1. 原始定理声明（用 block 中的定理来证明）
    2. #print axioms 检查
    
    如果 block 中的定理改变了类型，就无法证明原始定理。
    
    Args:
        code: Lean 代码
        theorem_name: 要检查的定理名
        
    Returns:
        (is_cheating, error_message, all_axioms)
    """
    # 获取原始定理信息
    orig_name, orig_params, orig_type, orig_call_args = get_original_theorem_info()
    
    if not orig_name:
        # 无法获取原始定理信息，回退到简单的 axiom 检测
        print("[Warning] Cannot get original theorem info, falling back to simple axiom check")
        return _simple_axiom_check(code, theorem_name)
    
    # 构造验证代码：在 EVOLVE-BLOCK-END 之后插入原始定理声明
    # 原始定理用 block 中的定理来证明
    verification_code = f"""
-- [Cheating Check] Verify that the theorem in EVOLVE-BLOCK proves the original specification
theorem {orig_name}_original {orig_params} :
    {orig_type} :=
  {theorem_name} {orig_call_args}

#print axioms {orig_name}_original
"""
    
    # 在 EVOLVE-BLOCK-END 之后插入验证代码
    block_end_pattern = re.compile(r'(-- EVOLVE-BLOCK-END)')
    match = block_end_pattern.search(code)
    
    if match:
        insert_pos = match.end()
        check_code = code[:insert_pos] + verification_code + code[insert_pos:]
    else:
        check_code = code + verification_code
    
    # 发送到 Lean server 验证
    result = verify_lean_code(check_code)
    
    # 检查编译是否成功
    is_valid_no_sorry, is_valid_with_sorry, error_messages = check_verification_result(result)
    
    if not is_valid_with_sorry:
        # 编译失败 = 定理类型被改变了，无法证明原始定理
        error_str = "\n".join(error_messages[:3]) if error_messages else "Unknown error"
        return True, f"Theorem signature changed, cannot prove original spec: {error_str}", []
    
    # 编译成功，检查 axioms
    axioms = extract_axioms_from_result(result)
    
    # 检查是否有非标准公理
    cheating_axioms = []
    for axiom in axioms:
        axiom_base = axiom.split('.')[-1] if '.' in axiom else axiom
        if axiom_base not in STANDARD_AXIOMS and axiom not in STANDARD_AXIOMS:
            cheating_axioms.append(axiom)
    
    if cheating_axioms:
        return True, f"Non-standard axioms used: {cheating_axioms}", axioms
    
    return False, None, axioms


def _simple_axiom_check(code: str, theorem_name: str) -> tuple:
    """简单的 axiom 检测（回退方案）"""
    block_end_pattern = re.compile(r'(-- EVOLVE-BLOCK-END)')
    match = block_end_pattern.search(code)
    
    if match:
        insert_pos = match.end()
        check_code = code[:insert_pos] + f"\n#print axioms {theorem_name}\n" + code[insert_pos:]
    else:
        check_code = code + f"\n#print axioms {theorem_name}\n"
    
    result = verify_lean_code(check_code)
    axioms = extract_axioms_from_result(result)
    
    cheating_axioms = []
    for axiom in axioms:
        axiom_base = axiom.split('.')[-1] if '.' in axiom else axiom
        if axiom_base not in STANDARD_AXIOMS and axiom not in STANDARD_AXIOMS:
            cheating_axioms.append(axiom)
    
    if cheating_axioms:
        return True, f"Non-standard axioms used: {cheating_axioms}", axioms
    return False, None, axioms


class CounterexampleFound(Exception):
    """反例异常，包含 quickcheck 发现的反例信息"""
    def __init__(self, messages: list):
        self.messages = messages
        super().__init__(f"Counterexample found: {messages}")


def get_node_counts(code: str) -> list:
    """
    获取代码中所有 sorry 目标的节点数列表
    
    将 sorry 替换为 countNodesAll; quickcheck 策略，发送到 Lean 服务器验证，
    提取返回的所有节点数。
    
    Returns:
        节点数列表（每个 sorry 对应一个）
    
    Raises:
        CounterexampleFound: 如果 quickcheck 发现反例
    """
    score_code = "import Tacs\n" + code
    score_code = score_code.replace("sorry", "countNodesAll; quickcheck") \
                           .replace("admit", "countNodesAll; quickcheck")
    
    result = verify_lean_code(score_code)
    error_info = result.get("error_message", (False, []))
    if error_info[0]:
        raise CounterexampleFound(error_info[1])
    return extract_node_scores(result)


# 缓存 initial_program 的节点数
_INITIAL_PROGRAM_NODE_COUNT = None


def get_theorem_node_count() -> int:
    """
    获取 initial_program 中 theorem 的节点数（缓存）
    """
    global _INITIAL_PROGRAM_NODE_COUNT
    
    if _INITIAL_PROGRAM_NODE_COUNT is not None:
        return _INITIAL_PROGRAM_NODE_COUNT
    
    initial_program_path = Path(__file__).parent / "initial_program.lean"
    
    if not initial_program_path.exists():
        print(f"[Warning] initial_program.lean not found at {initial_program_path}")
        return 0
    
    with open(initial_program_path, "r") as f:
        initial_code = f.read()
    
    node_counts = get_node_counts(initial_code)
    
    if node_counts:
        _INITIAL_PROGRAM_NODE_COUNT = max(node_counts)
        print(f"[NodeCount] Theorem node count from initial_program: {_INITIAL_PROGRAM_NODE_COUNT}")
    else:
        _INITIAL_PROGRAM_NODE_COUNT = 0
        print("[Warning] Could not get theorem node count from initial_program")
    
    return _INITIAL_PROGRAM_NODE_COUNT


def compute_error_score(error_count: int, max_errors: int = 20, weight: float = 0.2) -> float:
    """
    计算 error 分数（基于第一遍编译的错误数量）
    
    使用 sigmoid 映射：
    - 0 个 error = weight (满分)
    - max_errors 个或以上 = 接近 0
    - 中间区域有更好的梯度
    
    Args:
        error_count: 错误数量
        max_errors: 超过这个数量分数接近 0
        weight: 满分权重
        
    Returns:
        error 分数 [0, weight]
    """
    
    if error_count <= 0:
        return weight
    
    # 将 error_count 映射到 [0, 1]：0 个错误 = 1，max_errors 个 = 0
    ratio = max(1 - error_count / max_errors, 0)
    
    # Sigmoid 映射: 将 [0, 1] 映射到 [0, 1]，中间区域梯度更大
    k = 6
    sigmoid_score = 1 / (1 + math.exp(-k * (ratio - 0.5)))
    # 归一化：确保 ratio=0 时为 0，ratio=1 时为 1
    sigmoid_min = 1 / (1 + math.exp(-k * (0 - 0.5)))
    sigmoid_max = 1 / (1 + math.exp(-k * (1 - 0.5)))
    normalized_score = (sigmoid_score - sigmoid_min) / (sigmoid_max - sigmoid_min)
    
    return weight * normalized_score


def compute_node_score(
    theorem_node_count: int,
    lemma_node_counts: list,
    weight: float = 0.8,
) -> float:
    """
    计算 node counts 分数（基于 revise 后的代码）
    
    使用 LogSumExp 代替 max，让多个高复杂度目标产生更高的"有效复杂度"，
    从而区分"2个复杂度98的sorry"和"1个复杂度98的sorry"。
    
    使用 sigmoid 映射，让中间进展更有区分度：
    - 当 effective_max = theorem_node_count 时，分数接近 0
    - 当 effective_max = 0 时，分数接近 weight
    - 中间区域有更好的梯度
    
    Args:
        theorem_node_count: 原始定理的节点数
        lemma_node_counts: 各个 sorry 目标的节点数列表
        weight: 满分权重
        
    Returns:
        node 分数 [0, weight]
    """
    
    if theorem_node_count <= 0 or not lemma_node_counts:
        return 0.0
    
    raw_max = max(lemma_node_counts)
    
    if raw_max <= 0:
        return 0.0
    
    # 使用 LogSumExp 代替 max，让多个接近最大值的目标产生更高的有效复杂度
    # temperature 控制平滑程度：越小越接近 max，越大越接近 sum
    # temperature=5 时：2个98 → effective_max ≈ 101.5，1个98 → effective_max = 98
    temperature = 5.0
    
    # LogSumExp: log(sum(exp(x/T))) * T ≈ max(x) 但对多个接近 max 的值敏感
    scaled = [nc / temperature for nc in lemma_node_counts]
    max_scaled = max(scaled)
    logsumexp = max_scaled + math.log(sum(math.exp(s - max_scaled) for s in scaled))
    effective_max = logsumexp * temperature
    
    # 节点减少比例: 0 (无进展) 到 1 (完全解决)
    reduction_ratio = max(1 - effective_max / theorem_node_count, 0)
    
    # Sigmoid 映射: 将 [0, 1] 映射到 [0, 1]，中间区域梯度更大
    # 使用 sigmoid(k * (x - 0.5)) 并归一化到 [0, 1]
    # k 控制曲线陡峭程度，k=6 时效果较好
    k = 6
    sigmoid_score = 1 / (1 + math.exp(-k * (reduction_ratio - 0.5)))
    # 归一化：确保 reduction_ratio=0 时为 0，reduction_ratio=1 时为 1
    sigmoid_min = 1 / (1 + math.exp(-k * (0 - 0.5)))
    sigmoid_max = 1 / (1 + math.exp(-k * (1 - 0.5)))
    normalized_score = (sigmoid_score - sigmoid_min) / (sigmoid_max - sigmoid_min)
    
    return weight * normalized_score


def check_verification_result(result: dict) -> tuple:
    """
    检查验证结果
    
    Returns:
        (is_valid_no_sorry, is_valid_with_sorry, error_messages_list)
    """
    if "error" in result:
        return False, False, [result.get("error", "Unknown error")]
    
    is_valid_no_sorry = result.get("is_valid_no_sorry", False)
    is_valid_with_sorry = result.get("is_valid_with_sorry", False)
    
    # 提取错误信息列表
    ok, errors = result.get("error_message", (False, []))
    error_list = errors if isinstance(errors, list) else [str(errors)] if errors else []
    
    return is_valid_no_sorry, is_valid_with_sorry, error_list


def extract_unsolved_goals(result: dict) -> list:
    """
    从验证结果中提取 unsolved goals（sorry 位置的目标）
    
    Args:
        result: Lean 服务器返回的验证结果
        
    Returns:
        unsolved goals 列表，每个元素是 dict: {"line": X, "goal": "⊢ goal_type", "full_goal": "context\n⊢ type"}
    """
    unsolved_goals = []
    
    try:
        # sorry 信息在 response.response.sorries 中
        sorries = result.get('response', {}).get('response', {}).get('sorries', [])
        
        for sorry_info in sorries:
            # 每个 sorry 的结构：{"pos": {"line": X}, "goal": "context\n⊢ type"}
            pos = sorry_info.get('pos', {})
            line_num = pos.get('line', 0)
            goal = sorry_info.get('goal', '')
            
            if goal:
                # goal 格式通常是 "x : Nat\n⊢ x + 1 > x"
                # 提取 ⊢ 后面的部分作为目标类型
                if '⊢' in goal:
                    goal_type = goal.split('⊢', 1)[-1].strip()
                else:
                    goal_type = goal.strip()
                
                unsolved_goals.append({
                    "line": line_num,
                    "goal": f"⊢ {goal_type}",
                    "full_goal": goal
                })
            else:
                unsolved_goals.append({
                    "line": line_num,
                    "goal": "sorry (no goal info)",
                    "full_goal": ""
                })
    except Exception as e:
        print(f"[Warning] Failed to extract unsolved goals: {e}")
    
    return unsolved_goals


def get_hardest_unsolved_goal(unsolved_goals: list, node_counts: list) -> str:
    """
    获取复杂度最高的那个 unsolved goal
    
    Args:
        unsolved_goals: unsolved goals 列表 (from extract_unsolved_goals)
        node_counts: 节点数列表 (from get_node_counts)，用于衡量复杂度
        
    Returns:
        格式化的字符串，只包含复杂度最高的那个 goal（包含完整假设）
    """
    if not unsolved_goals:
        return "No unsolved goals"
    
    if not node_counts:
        # 没有复杂度信息，返回第一个（使用 full_goal 包含假设）
        g = unsolved_goals[0]
        return f"line {g['line']}:\n{g['full_goal']}" if g['full_goal'] else f"line {g['line']}: {g['goal']}"
    
    # 找出复杂度最高的索引
    max_idx = 0
    max_complexity = node_counts[0] if node_counts else 0
    for i, count in enumerate(node_counts):
        if count > max_complexity:
            max_complexity = count
            max_idx = i
    
    # 确保索引有效
    if max_idx < len(unsolved_goals):
        g = unsolved_goals[max_idx]
        # 使用 full_goal 输出完整的假设和目标
        if g['full_goal']:
            return f"[Hardest goal, complexity={max_complexity}] line {g['line']}:\n{g['full_goal']}"
        else:
            return f"[Hardest goal, complexity={max_complexity}] line {g['line']}: {g['goal']}"
    else:
        # 索引不匹配，返回第一个
        g = unsolved_goals[0]
        return f"line {g['line']}:\n{g['full_goal']}" if g['full_goal'] else f"line {g['line']}: {g['goal']}"


def revise_proof(code: str, error_messages: list) -> tuple:
    """
    修复编译失败的代码
    
    使用 LLM 智能修复或 sorry 替换。
    revise 模块已自带 Lean 验证功能，会自动验证修复结果。
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        
    Returns:
        (revised_code, success, revised_info)
    """
    try:
        # 使用智能修复模块（内置 Lean 验证）
        return smart_revise_proof(code, error_messages)
    except Exception as e:
        print(f"[Revise] Failed: {e}")
        return code, False, []
    

def evaluate(program_path: str) -> EvaluationResult:
    """
    评估 Lean 4 程序
    
    评分规则：
    - 证明完全成功（无 sorry）: 1.0
    - 作弊: 0.0
    - 一般情况 (combined_score = error_score + node_score):
      - error_score (占 0.2): 基于第一遍编译的错误数量
      - node_score (占 0.8): 基于 revise 后的代码计算
    
    报告规则：
    - 第一遍有语法错误：报告语法错误
    - 第一遍无语法错误：报告 sorries
    - revise 结果只用于打分，不用于报告
    
    Args:
        program_path: Lean 程序文件路径
        
    Returns:
        EvaluationResult 包含分数和 artifacts
    """
    try:
        with open(program_path, "r") as f:
            original_code = f.read()
        
        # ===== 第一遍验证（用于报告和 error_score） =====
        first_result = verify_lean_code(original_code)
        first_is_valid_no_sorry, first_is_valid_with_sorry, first_error_messages = check_verification_result(first_result)
        first_unsolved_goals = extract_unsolved_goals(first_result)
        
        # 计算 error_score（基于第一遍编译的错误数量）
        first_error_count = len(first_error_messages) if first_error_messages else 0
        error_score = compute_error_score(first_error_count, weight=0.1)
        
        # 用于报告的信息（基于第一遍验证）
        if not first_is_valid_with_sorry:
            # 第一遍有语法错误，提取编译错误
            compile_errors = []
            for msg in first_error_messages[:5]:
                if msg and isinstance(msg, str):
                    line_match = re.search(r"\{'line':\s*(\d+)", msg)
                    line_num = int(line_match.group(1)) if line_match else 0
                    error_desc = msg.split("}: ", 1)[-1][:150] if "}: " in msg else msg[:150]
                    compile_errors.append(f"- line {line_num}: {error_desc}")
            report_compile_errors_str = "\n".join(compile_errors) if compile_errors else "Unknown compilation error"
        else:
            # 第一遍无语法错误，报告 sorries (稍后会选择最难的那个)
            report_compile_errors_str = None
        
        # ===== 用于打分的代码（revise 后） =====
        scoring_code = original_code
        revised = False
        revised_nodes = []
        
        # 如果第一遍编译失败，尝试 revise（用于打分和保存）
        if not first_is_valid_with_sorry and first_error_messages:
            revised_code, revise_success, revised_nodes = revise_proof(original_code, first_error_messages)
            if revise_success and revised_code != original_code:
                scoring_code = revised_code
                revised = True
                print(f"[Revise] revised nodes: {revised_nodes}")
        
        # 用于打分的验证结果
        if revised:
            scoring_result = verify_lean_code(scoring_code)
            scoring_is_valid_no_sorry, scoring_is_valid_with_sorry, _ = check_verification_result(scoring_result)
            print(f"[Revise] Code fixed for scoring, is_valid_with_sorry={scoring_is_valid_with_sorry}")
            # 如果 revise 成功，error_score 应该是满分（代码已经没有语法错误了）
            if scoring_is_valid_with_sorry:
                error_score = 0.1  # 满分
        else:
            scoring_result = first_result
            scoring_is_valid_no_sorry = first_is_valid_no_sorry
            scoring_is_valid_with_sorry = first_is_valid_with_sorry
        
        # ===== 计算 node_score =====
        node_score = 0.0
        theorem_nc = 0
        lemma_ncs = []
        
        if scoring_is_valid_with_sorry:
            # revise 后编译成功，可以计算 node_score
            # 先检查作弊
            orig_theorem_name, _, _, _ = get_original_theorem_info()
            if orig_theorem_name:
                is_cheating, cheating_error, all_axioms = check_axiom_cheating(scoring_code, orig_theorem_name)
                if is_cheating:
                    print(f"[Cheating] {cheating_error}")
                    return EvaluationResult(
                        metrics={
                            "combined_score": 0.0,
                            "error_score": 0.0,
                            "node_score": 0.0,
                            "error_count": float(first_error_count),
                            "is_cheating": 1.0,
                            "has_counterexample": 0.0,
                        },
                        artifacts={
                            "error": f"Cheating detected: {cheating_error}",
                            "all_axioms": str(all_axioms),
                            "status": "cheating",
                            "suggestion": "Do not use 'axiom', change theorem type, or other cheating methods. Prove the original theorem properly.",
                        },
                    )
            
            # 检查是否完全成功（无 sorry）
            if scoring_is_valid_no_sorry:
                # 完全成功，无 sorry
                return EvaluationResult(
                    metrics={
                        "combined_score": 1.0,
                        "error_score": error_score,
                        "node_score": 0.8,  # 满分
                        "error_count": float(first_error_count),
                        "has_sorry": 0.0,
                        "has_counterexample": 0.0,
                    },
                    artifacts={
                        "message": "Proof complete! No sorry found.",
                        "status": "prove_no_sorry",
                    },
                )
            
            # 有 sorry，计算 node_score
            theorem_nc = get_theorem_node_count()
            try:
                lemma_ncs = get_node_counts(scoring_code)
            except CounterexampleFound as e:
                print(f"[Score] Counterexample found: {e.messages}")
                return EvaluationResult(
                    metrics={
                        "combined_score": 0.0,
                        "error_score": error_score,
                        "node_score": 0.0,
                        "error_count": float(first_error_count),
                        "has_sorry": 1.0,
                        "is_revised": 1.0 if revised else 0.0,
                        "has_counterexample": 1.0,
                    },
                    artifacts={"status": "counterexample_found", "counterexample": e.messages},
                )
            print(f"[Score] theorem_nc={theorem_nc}, lemma_ncs={lemma_ncs}")
            
            if theorem_nc > 0 and lemma_ncs:
                node_score = compute_node_score(theorem_nc, lemma_ncs, weight=0.9)
        
        # ===== 计算总分 =====
        combined_score = error_score + node_score
        
        # ===== 构建 artifacts =====
        # revise 成功时，artifacts 和无语法错误时一样，只是额外传递 revised_code 供 iteration.py 使用
        is_revised = revised and scoring_is_valid_with_sorry
        
        if is_revised:
            # revise 成功，报告 revise 后代码的 unsolved_goals（就像没有语法错误一样）
            scoring_unsolved_goals = extract_unsolved_goals(scoring_result)
            hardest_goal_str = get_hardest_unsolved_goal(scoring_unsolved_goals, lemma_ncs)
            artifacts = {
                "status": "prove_with_sorry",
                "unsolved_goals": hardest_goal_str,
                "revised_code": scoring_code,  # 供 iteration.py 保存到 database
            }
        elif report_compile_errors_str:
            # 第一遍有语法错误，revise 失败
            artifacts = {
                "status": "compile_failed",
                "compile_errors": report_compile_errors_str,
            }
        else:
            # 第一遍无语法错误
            hardest_goal_str = get_hardest_unsolved_goal(first_unsolved_goals, lemma_ncs)
            artifacts = {
                "status": "prove_with_sorry",
                "unsolved_goals": hardest_goal_str,
            }
        
        # 如果 revise 成功，error_count 应该是 0（revise 后的代码没有编译错误）
        final_error_count = 0 if is_revised else first_error_count
        
        return EvaluationResult(
            metrics={
                "combined_score": combined_score,
                "error_score": error_score,
                "node_score": node_score,
                "error_count": float(final_error_count),
                "has_sorry": 1.0 if not scoring_is_valid_no_sorry else 0.0,
                "is_revised": 1.0 if is_revised else 0.0,
                "has_counterexample": 0.0,
                "theorem_node_count": float(theorem_nc),
                "max_lemma_node_count": float(max(lemma_ncs)) if lemma_ncs else 0.0,
                "num_sorries": float(len(lemma_ncs)),
                "lemma_node_counts": lemma_ncs,
            },
            artifacts=artifacts,
        )
        
    except FileNotFoundError:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": f"File not found: {program_path}"},
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": f"Evaluation error: {str(e)}", "type": type(e).__name__},
        )


# 测试入口
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        program_path = sys.argv[1]
    else:
        program_path = Path(__file__).parent / "initial_program.lean"
    
    print(f"Evaluating: {program_path}")
    print("=" * 60)
    
    result = evaluate(str(program_path))
    metrics = result.metrics
    
    print(f"\nResults:")
    print(f"  Combined Score: {metrics.get('combined_score', 0.0):.4f}")
    print(f"  Error Score: {metrics.get('error_score', 0.0):.4f} (max 0.2)")
    print(f"  Node Score: {metrics.get('node_score', 0.0):.4f} (max 0.8)")
    print(f"  Error Count: {metrics.get('error_count', 0):.0f}")
    print(f"  Has Sorry: {metrics.get('has_sorry', 'N/A')}")
    print(f"  Theorem Node Count: {metrics.get('theorem_node_count', 'N/A')}")
    print(f"  Max Lemma Node Count: {metrics.get('max_lemma_node_count', 'N/A')}")
    print(f"  Num Sorries: {metrics.get('num_sorries', 'N/A')}")
    print(f"  Lemma Node Counts: {metrics.get('lemma_node_counts', 'N/A')}")
    
    if result.has_artifacts():
        print(f"\nArtifacts:")
        for key, value in result.artifacts.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
