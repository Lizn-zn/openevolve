"""
Revise 模块的工具函数
"""

import re
from typing import Optional, Tuple, List


def find_evolve_block_bounds(code: str) -> Tuple[int, int]:
    """
    查找 EVOLVE-BLOCK 的边界行号
    
    Args:
        code: Lean 代码
        
    Returns:
        (start_idx, end_idx) 0-indexed 行号，未找到返回 (-1, -1)
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
    
    return start_idx, end_idx


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
    for msg in error_messages[:10]:  # 最多显示 10 条
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


def force_preserve_outside_block(original_code: str, fixed_code: str) -> str:
    """
    强制保留 EVOLVE-BLOCK 外的原始代码
    
    只提取修复后的 EVOLVE-BLOCK 内容，与原始代码的外部结构合并。
    
    Args:
        original_code: 原始完整代码
        fixed_code: 修复后的代码
        
    Returns:
        合并后的代码（保证 EVOLVE-BLOCK 外不变）
    """
    orig_lines = original_code.split('\n')
    orig_start, orig_end = find_evolve_block_bounds(original_code)
    
    if orig_start == -1 or orig_end == -1:
        return fixed_code
    
    fixed_lines = fixed_code.split('\n')
    fixed_start, fixed_end = find_evolve_block_bounds(fixed_code)
    
    if fixed_start == -1 or fixed_end == -1:
        return fixed_code
    
    # 提取修复后的 BLOCK 内容（不包含 START/END 行）
    fixed_block = fixed_lines[fixed_start + 1:fixed_end]
    
    # 用原始结构 + 修复的 BLOCK 内容重建代码
    result_lines = (
        orig_lines[:orig_start + 1] +  # 原始代码到 EVOLVE-BLOCK-START（包含）
        fixed_block +                   # 修复的内容
        orig_lines[orig_end:]           # 原始代码从 EVOLVE-BLOCK-END（包含）到结束
    )
    
    return '\n'.join(result_lines)
