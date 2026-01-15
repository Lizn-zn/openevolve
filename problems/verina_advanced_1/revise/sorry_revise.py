"""
Sorry 替换模块

将编译失败的 lemma/theorem 的证明替换为 sorry，使代码能够编译通过。
这是 LLM 修复失败时的回退方案。
"""

import re
from typing import Tuple, List, Optional


def find_theorem_at_line(code: str, error_line: int) -> Optional[str]:
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
        match = re.match(r'^\s*(lemma|theorem)\s+(\w+)', line)
        if match:
            return match.group(2)
        # 也检查行内的声明（可能前面有注释等）
        match = re.search(r'\b(lemma|theorem)\s+(\w+)', line)
        if match and not line.strip().startswith('--') and not line.strip().startswith('/-'):
            return match.group(2)
    return None


def extract_error_lines(error_messages: list, line_offset: int = 0) -> List[int]:
    """
    从错误消息列表中提取行号
    
    Args:
        error_messages: 错误消息列表
        line_offset: 行号偏移
        
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


def find_all_theorems(code: str) -> List[Tuple[str, int, int]]:
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


def find_theorem_containing_line(
    code: str, 
    error_line: int, 
    theorems: List[Tuple[str, int, int]] = None
) -> Optional[str]:
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
    
    # 找到包含这个行的定理
    for i, (name, start_line, by_line) in enumerate(theorems):
        if i + 1 < len(theorems):
            next_start = theorems[i + 1][1]
        else:
            next_start = len(code.split('\n'))
        
        if start_line <= error_idx < next_start:
            return name
    
    return None


def replace_proof_with_sorry(code: str, theorem_name: str) -> Tuple[str, bool]:
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
    
    # 找到证明的结束行
    proof_end_idx = len(lines)
    for i in range(by_line_idx + 1, len(lines)):
        line = lines[i]
        # 检查是否是新的顶层声明
        if re.match(r'^(lemma|theorem|def|axiom|instance|class|structure|inductive|namespace|end|section|#|/-)', line):
            proof_end_idx = i
            break
        # 检查 EVOLVE-BLOCK-END
        if line.strip().startswith('-- EVOLVE-BLOCK-END'):
            proof_end_idx = i
            break
    
    # 构造新代码
    decl_lines = lines[decl_start_idx:by_line_idx + 1]
    
    # 处理 := by 行
    last_decl_line = decl_lines[-1]
    by_pos = last_decl_line.find(':= by')
    if by_pos == -1:
        by_pos = last_decl_line.find(':=by')
    if by_pos != -1:
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


def sorry_revise(code: str, error_messages: list) -> Tuple[str, bool, List[str]]:
    """
    使用 sorry 替换修复编译失败的代码
    
    将有错误的 lemma/theorem 的证明替换为 sorry。
    
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
        name = find_theorem_containing_line(code, line, theorems)
        if not name:
            name = find_theorem_at_line(code, line)
        if name:
            names_to_revise.add(name)
    
    if not names_to_revise:
        return code, False, []
    
    # 逐个替换证明
    revised_code = code
    revised_names = []
    
    for name in names_to_revise:
        new_code, success = replace_proof_with_sorry(revised_code, name)
        if success:
            revised_code = new_code
            revised_names.append(name)
    
    return revised_code, len(revised_names) > 0, revised_names

