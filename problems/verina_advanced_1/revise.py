"""
Simple revise module for Lean 4 code.

将编译失败的 lemma/theorem 的证明替换为 sorry，使代码能够编译通过。
不依赖 LeanCodeTree，直接处理所有 lemma/theorem。
"""

import re


def find_theorem_at_line(code: str, error_line: int) -> str | None:
    """
    从错误行向上查找最近的 lemma/theorem 名称
    
    Args:
        code: Lean 代码
        error_line: 错误行号（1-indexed）
        
    Returns:
        找到的 lemma/theorem 名称，或 None
    """
    lines = code.split('\n')
    # 从错误行向上查找
    for i in range(min(error_line - 1, len(lines) - 1), -1, -1):
        line = lines[i]
        # 匹配 lemma 或 theorem 声明（行首，可能有空格）
        # 支持多行声明：lemma name ... 可能在后续行
        match = re.match(r'^\s*(lemma|theorem)\s+(\w+)', line)
        if match:
            return match.group(2)
        # 也检查行内的声明（可能前面有注释等）
        match = re.search(r'\b(lemma|theorem)\s+(\w+)', line)
        if match and not line.strip().startswith('--') and not line.strip().startswith('/-'):
            return match.group(2)
    return None


def extract_error_lines(error_messages: list, line_offset: int = 0) -> list:
    """
    从错误消息列表中提取行号
    
    Args:
        error_messages: 错误消息列表，格式如 "{'line': 45, 'column': 40}: unsolved goals..."
        line_offset: 行号偏移（Lean Server 可能有行号偏移）
        
    Returns:
        行号列表（已应用偏移）
    """
    lines = []
    for msg in error_messages:
        if not msg:
            continue
        # 匹配 {'line': N, ...} 格式
        match = re.search(r"\{'line':\s*(\d+)", str(msg))
        if match:
            lines.append(int(match.group(1)) + line_offset)
    return lines


def find_all_theorems(code: str) -> list:
    """
    找到代码中所有的 lemma/theorem 声明
    
    Returns:
        [(name, start_line, by_line), ...]
    """
    lines = code.split('\n')
    theorems = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # 匹配 lemma 或 theorem 声明
        match = re.match(r'^\s*(lemma|theorem)\s+(\w+)', line)
        if match:
            name = match.group(2)
            start_line = i
            # 找到 := by 的位置
            by_line = None
            for j in range(i, min(i + 30, len(lines))):
                if ':= by' in lines[j] or ':=by' in lines[j]:
                    by_line = j
                    break
            if by_line is not None:
                theorems.append((name, start_line, by_line))
        i += 1
    
    return theorems


def find_theorem_containing_line(code: str, error_line: int, theorems: list = None) -> str | None:
    """
    找到包含错误行的 lemma/theorem
    
    Args:
        code: Lean 代码
        error_line: 错误行号（1-indexed）
        theorems: 预计算的定理列表
        
    Returns:
        找到的 lemma/theorem 名称，或 None
    """
    if theorems is None:
        theorems = find_all_theorems(code)
    
    error_idx = error_line - 1  # 转为 0-indexed
    
    # 找到包含这个行的定理（error_idx 在 [start_line, next_start_line) 范围内）
    for i, (name, start_line, by_line) in enumerate(theorems):
        # 确定这个定理的范围
        if i + 1 < len(theorems):
            next_start = theorems[i + 1][1]
        else:
            next_start = len(code.split('\n'))
        
        if start_line <= error_idx < next_start:
            return name
    
    return None


def find_proof_range(code: str, theorem_name: str) -> tuple | None:
    """
    找到指定 lemma/theorem 的证明范围
    
    Args:
        code: Lean 代码
        theorem_name: lemma/theorem 名称
        
    Returns:
        (start_pos, end_pos) 证明体的位置（包含 := by 之后到证明结束），或 None
    """
    # 匹配 lemma/theorem 声明，支持多行
    # 模式：(lemma|theorem) name ... := by ... (直到下一个顶层声明或文件结束)
    
    # 首先找到 theorem/lemma 的位置
    pattern = rf'\b(lemma|theorem)\s+{re.escape(theorem_name)}\b'
    match = re.search(pattern, code)
    if not match:
        return None
    
    decl_start = match.start()
    
    # 找到 := by 的位置
    by_pattern = r':=\s*by\b'
    by_match = re.search(by_pattern, code[decl_start:])
    if not by_match:
        return None
    
    proof_start = decl_start + by_match.end()
    
    # 找到证明的结束位置
    # 策略：找到下一个顶层声明（lemma, theorem, def, end, namespace 等）或文件结束
    remaining = code[proof_start:]
    
    # 顶层声明模式（行首或只有空格后）
    end_patterns = [
        r'^\s*(lemma|theorem|def|axiom|instance|class|structure|inductive|namespace|end|section|#)\b',
        r'^\s*-- EVOLVE-BLOCK-END',
    ]
    
    lines = remaining.split('\n')
    proof_end = len(remaining)
    
    for i, line in enumerate(lines):
        if i == 0:
            continue  # 跳过第一行（:= by 后面可能有内容）
        for pat in end_patterns:
            if re.match(pat, line):
                # 找到结束位置
                proof_end = sum(len(l) + 1 for l in lines[:i])
                return (proof_start, proof_start + proof_end)
    
    return (proof_start, proof_start + proof_end)


def replace_proof_with_sorry(code: str, theorem_name: str) -> tuple:
    """
    将指定 lemma/theorem 的证明替换为 sorry
    
    Args:
        code: Lean 代码
        theorem_name: lemma/theorem 名称
        
    Returns:
        (new_code, success)
    """
    lines = code.split('\n')
    
    # 找到 theorem/lemma 声明的起始行
    decl_start_idx = None
    for i, line in enumerate(lines):
        if re.match(rf'^(lemma|theorem)\s+{re.escape(theorem_name)}\b', line):
            decl_start_idx = i
            break
    
    if decl_start_idx is None:
        return code, False
    
    # 找到 := by 的位置
    by_line_idx = None
    for i in range(decl_start_idx, min(decl_start_idx + 20, len(lines))):
        if ':= by' in lines[i] or ':=by' in lines[i]:
            by_line_idx = i
            break
        # 也处理 := by 分在两行的情况
        if i > decl_start_idx and lines[i].strip().startswith('by') and ':=' in lines[i-1]:
            by_line_idx = i
            break
    
    if by_line_idx is None:
        return code, False
    
    # 找到证明的结束行（下一个顶层声明或 EVOLVE-BLOCK-END）
    proof_end_idx = len(lines)
    for i in range(by_line_idx + 1, len(lines)):
        line = lines[i]
        # 检查是否是新的顶层声明（行首无缩进）
        if re.match(r'^(lemma|theorem|def|axiom|instance|class|structure|inductive|namespace|end|section|#|/-)', line):
            proof_end_idx = i
            break
        # 检查 EVOLVE-BLOCK-END
        if line.strip().startswith('-- EVOLVE-BLOCK-END'):
            proof_end_idx = i
            break
    
    # 构造新代码
    # 保留声明部分（到 := by 所在行）
    decl_lines = lines[decl_start_idx:by_line_idx + 1]
    
    # 处理 := by 行，截取到 := by 结束
    last_decl_line = decl_lines[-1]
    by_pos = last_decl_line.find(':= by')
    if by_pos == -1:
        by_pos = last_decl_line.find(':=by')
    if by_pos != -1:
        # 保留到 := by 结束
        decl_lines[-1] = last_decl_line[:by_pos + 5]  # := by = 5 chars
    
    # 组合新代码
    new_lines = (
        lines[:decl_start_idx] +
        decl_lines +
        ['  sorry'] +
        [''] +
        lines[proof_end_idx:]
    )
    
    return '\n'.join(new_lines), True


def revise_proof(code: str, error_messages: list) -> tuple:
    """
    修复编译失败的代码
    
    将有错误的 lemma/theorem 的证明替换为 sorry，使代码能够编译通过。
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        
    Returns:
        (revised_code, success, revised_names)
    """
    if not error_messages:
        return code, False, []
    
    # 提取错误行号（Lean Server 有 2 行偏移）
    error_lines = extract_error_lines(error_messages, line_offset=2)
    if not error_lines:
        return code, False, []
    
    # 预计算所有定理
    theorems = find_all_theorems(code)
    
    # 找到需要修复的 lemma/theorem 名称
    names_to_revise = set()
    for line in error_lines:
        # 首先尝试新方法：找到包含错误行的定理
        name = find_theorem_containing_line(code, line, theorems)
        if not name:
            # 回退到旧方法：向上查找
            name = find_theorem_at_line(code, line)
        if name:
            names_to_revise.add(name)
    
    if not names_to_revise:
        print(f"[Revise] Could not find theorems for error lines: {error_lines}")
        return code, False, []
    
    # 逐个替换证明
    revised_code = code
    revised_names = []
    
    for name in names_to_revise:
        new_code, success = replace_proof_with_sorry(revised_code, name)
        if success:
            revised_code = new_code
            revised_names.append(name)
            print(f"[Revise] Replaced proof of '{name}' with sorry")
        else:
            print(f"[Revise] Failed to replace proof of '{name}'")
    
    return revised_code, len(revised_names) > 0, revised_names


# 测试
if __name__ == "__main__":
    test_code = """import Mathlib
set_option maxHeartbeats 0

namespace verina_advanced_1

-- EVOLVE-BLOCK-START

lemma lemma1_nonempty (nums : List Int) : nums.length > 0 := by
  have hlen : nums.length = 0 ∨ nums.length > 0 := by
    exact lt_or_eq_of_le (Nat.zero_le _ ) |> Or.symm |> Or_flip ?_
  sorry

theorem FindSingleNumber_spec_satisfied (nums: List Int) : True := by
  have h := lemma1_nonempty nums
  exact True.intro

-- EVOLVE-BLOCK-END

end verina_advanced_1
"""
    
    error_msgs = [
        "{'line': 10, 'column': 56}: unknown identifier 'Or_flip'",
        "{'line': 8, 'column': 23}: unsolved goals",
    ]
    
    revised, success, names = revise_proof(test_code, error_msgs)
    print("=" * 60)
    print(f"Success: {success}")
    print(f"Revised names: {names}")
    print("=" * 60)
    print(revised)

