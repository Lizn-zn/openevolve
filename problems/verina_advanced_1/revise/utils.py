"""
Revise 模块的工具函数
"""

import re
from typing import Optional, Tuple


def extract_evolve_block(code: str) -> Tuple[Optional[str], int, int]:
    """
    从代码中提取 EVOLVE-BLOCK 内容
    
    Args:
        code: 完整的 Lean 代码
        
    Returns:
        (block_content, start_line, end_line) 或 (None, -1, -1) 如果未找到
    """
    lines = code.split('\n')
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if '-- EVOLVE-BLOCK-START' in line:
            start_idx = i
        elif '-- EVOLVE-BLOCK-END' in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        return None, -1, -1
    
    block_lines = lines[start_idx + 1:end_idx]
    return '\n'.join(block_lines), start_idx + 1, end_idx


def replace_evolve_block(original_code: str, new_block_content: str) -> str:
    """
    替换代码中的 EVOLVE-BLOCK 内容
    
    Args:
        original_code: 原始完整代码
        new_block_content: 新的 EVOLVE-BLOCK 内容
        
    Returns:
        替换后的完整代码
    """
    lines = original_code.split('\n')
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if '-- EVOLVE-BLOCK-START' in line:
            start_idx = i
        elif '-- EVOLVE-BLOCK-END' in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        # 没有 EVOLVE-BLOCK，返回原代码
        return original_code
    
    # 构建新代码
    before = lines[:start_idx + 1]  # 包含 EVOLVE-BLOCK-START
    after = lines[end_idx:]         # 包含 EVOLVE-BLOCK-END
    
    new_block_lines = new_block_content.split('\n')
    
    return '\n'.join(before + new_block_lines + after)


def extract_lean_code_from_response(response: str) -> Optional[str]:
    """
    从 LLM 响应中提取 Lean 代码块
    
    Args:
        response: LLM 的响应文本
        
    Returns:
        提取的 Lean 代码，或 None
    """
    # 尝试匹配 ```lean ... ``` 格式
    pattern = r"```lean\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配 ```lean4 ... ``` 格式
    pattern = r"```lean4\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试匹配通用 ``` ... ``` 格式
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def format_error_messages(error_messages: list) -> str:
    """
    格式化错误消息为可读的字符串
    
    Args:
        error_messages: 错误消息列表
        
    Returns:
        格式化后的错误字符串
    """
    if not error_messages:
        return "No error messages"
    
    formatted = []
    for i, msg in enumerate(error_messages[:10], 1):  # 最多显示 10 条
        if msg:
            # 尝试提取行号
            line_match = re.search(r"\{'line':\s*(\d+)", str(msg))
            if line_match:
                line_num = int(line_match.group(1))
                # 提取错误描述
                error_desc = str(msg).split("}: ", 1)[-1] if "}: " in str(msg) else str(msg)
                formatted.append(f"- Line {line_num}: {error_desc[:200]}")
            else:
                formatted.append(f"- {str(msg)[:200]}")
    
    return '\n'.join(formatted)


def validate_code_structure(original_code: str, fixed_code: str) -> Tuple[bool, Optional[str]]:
    """
    验证修复后的代码结构是否正确
    
    检查：
    1. EVOLVE-BLOCK 标记是否保留
    2. EVOLVE-BLOCK 外的代码是否未被修改（忽略空白行差异）
    
    Args:
        original_code: 原始代码
        fixed_code: 修复后的代码
        
    Returns:
        (is_valid, error_message)
    """
    # 检查 EVOLVE-BLOCK 标记
    if '-- EVOLVE-BLOCK-START' not in fixed_code:
        return False, "Missing EVOLVE-BLOCK-START marker"
    if '-- EVOLVE-BLOCK-END' not in fixed_code:
        return False, "Missing EVOLVE-BLOCK-END marker"
    
    # 提取 EVOLVE-BLOCK 外的代码
    orig_lines = original_code.split('\n')
    fixed_lines = fixed_code.split('\n')
    
    orig_start = -1
    orig_end = -1
    for i, line in enumerate(orig_lines):
        if '-- EVOLVE-BLOCK-START' in line:
            orig_start = i
        elif '-- EVOLVE-BLOCK-END' in line:
            orig_end = i
            break
    
    fixed_start = -1
    fixed_end = -1
    for i, line in enumerate(fixed_lines):
        if '-- EVOLVE-BLOCK-START' in line:
            fixed_start = i
        elif '-- EVOLVE-BLOCK-END' in line:
            fixed_end = i
            break
    
    if orig_start == -1 or orig_end == -1 or fixed_start == -1 or fixed_end == -1:
        return False, "Could not locate EVOLVE-BLOCK boundaries"
    
    # 比较函数：忽略空白行和尾部空格
    def normalize_lines(lines):
        return [l.rstrip() for l in lines if l.strip()]
    
    # 检查 EVOLVE-BLOCK 之前的代码（忽略空白差异）
    orig_before = normalize_lines(orig_lines[:orig_start])
    fixed_before = normalize_lines(fixed_lines[:fixed_start])
    if orig_before != fixed_before:
        return False, "Code before EVOLVE-BLOCK was modified"
    
    # 检查 EVOLVE-BLOCK 之后的代码（忽略空白差异）
    orig_after = normalize_lines(orig_lines[orig_end + 1:])
    fixed_after = normalize_lines(fixed_lines[fixed_end + 1:])
    if orig_after != fixed_after:
        return False, "Code after EVOLVE-BLOCK was modified"
    
    return True, None


def force_preserve_outside_block(original_code: str, fixed_code: str) -> str:
    """
    强制保留 EVOLVE-BLOCK 外的原始代码
    
    只提取 LLM 修复后的 EVOLVE-BLOCK 内容，与原始代码的外部结构合并。
    
    Args:
        original_code: 原始完整代码
        fixed_code: LLM 修复后的代码
        
    Returns:
        合并后的代码（保证 EVOLVE-BLOCK 外不变）
    """
    # 提取原始代码的结构
    orig_lines = original_code.split('\n')
    orig_start = -1
    orig_end = -1
    for i, line in enumerate(orig_lines):
        if '-- EVOLVE-BLOCK-START' in line:
            orig_start = i
        elif '-- EVOLVE-BLOCK-END' in line:
            orig_end = i
            break
    
    if orig_start == -1 or orig_end == -1:
        return fixed_code
    
    # 提取 LLM 修复后的 EVOLVE-BLOCK 内容
    fixed_lines = fixed_code.split('\n')
    fixed_start = -1
    fixed_end = -1
    for i, line in enumerate(fixed_lines):
        if '-- EVOLVE-BLOCK-START' in line:
            fixed_start = i
        elif '-- EVOLVE-BLOCK-END' in line:
            fixed_end = i
            break
    
    if fixed_start == -1 or fixed_end == -1:
        return fixed_code
    
    # 提取 LLM 修复的 BLOCK 内容（不包含 START/END 行）
    fixed_block = fixed_lines[fixed_start + 1:fixed_end]
    
    # 用原始结构 + LLM 修复的 BLOCK 内容重建代码
    result_lines = (
        orig_lines[:orig_start + 1] +  # 原始代码到 EVOLVE-BLOCK-START（包含）
        fixed_block +                   # LLM 修复的内容
        orig_lines[orig_end:]           # 原始代码从 EVOLVE-BLOCK-END（包含）到结束
    )
    
    return '\n'.join(result_lines)

