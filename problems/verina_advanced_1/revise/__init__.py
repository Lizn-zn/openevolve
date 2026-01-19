"""
Revise 模块 - Lean 代码智能修复

提供三阶段修复策略：
1. Fix: 使用 LLM 尝试真正修复语法错误（不用 sorry）
2. Eliminate: 使用 LLM 进行最小化 sorry 填充
3. Sorry 硬替换: 直接将整个 proof body 替换为 sorry（回退方案）

使用方法：
    from revise import revise_proof
    
    revised_code, success, info = revise_proof(code, error_messages)
    
    # 也可以直接使用 Lean 验证
    from revise import verify_lean_code, is_code_valid
    
    result = verify_lean_code(code)
    valid = is_code_valid(code)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, List, Union

from .config import get_config
from .llm_revise import llm_revise
from .sorry_revise import sorry_revise
from .lean_verify import verify_lean_code, is_code_valid, check_verification_result
from .utils import force_preserve_outside_block

# 导入项目的 validate_changes_within_evolve_block
_OPENEVOLVE_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _OPENEVOLVE_ROOT not in sys.path:
    sys.path.insert(0, _OPENEVOLVE_ROOT)

from openevolve.utils.code_utils import validate_changes_within_evolve_block

logger = logging.getLogger("openevolve.revise")


def _ensure_changes_within_block(original_code: str, fixed_code: str) -> str:
    """
    确保修改只在 EVOLVE-BLOCK 内
    
    如果检测到 BLOCK 外有修改，强制只保留 BLOCK 内的修改。
    """
    is_valid, error_msg = validate_changes_within_evolve_block(original_code, fixed_code)
    
    if not is_valid:
        logger.warning(f"[Revise] {error_msg}")
        logger.info("[Revise] Forcing changes to stay within EVOLVE-BLOCK")
        return force_preserve_outside_block(original_code, fixed_code)
    
    return fixed_code


def revise_proof(
    code: str,
    error_messages: list,
    verify_func=None,
) -> Tuple[str, bool, Union[List[str], str], str]:
    """
    修复 Lean 代码中的编译错误
    
    三阶段修复策略：
    1. Fix: 尝试真正修复错误（不用 sorry）
    2. Eliminate: 使用最小化 sorry 填充
    3. Sorry 硬替换: 直接替换整个 proof body
    
    每次尝试失败后，会用新的代码和新的错误继续修复。
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        verify_func: 可选的验证函数，用于验证修复后的代码是否编译通过
                    签名: (code: str) -> dict with 'is_valid_with_sorry', 'error_messages'
                    如果不提供，默认使用内置的 verify_lean_code
        
    Returns:
        (revised_code, success, info, llm_responses)
        - revised_code: 修复后的代码
        - success: 是否修复成功
        - info: 修复信息
            - "fix": LLM 真正修复成功
            - "eliminate": LLM sorry 填充成功
            - [names...]: sorry 硬替换成功，返回修复的定理名列表
        - llm_responses: 所有 LLM response 的拼接（用 \\n---\\n 分隔）
    """
    if not error_messages:
        return code, False, [], ""
    
    config = get_config()
    
    # 如果没有提供验证函数，使用内置的
    if verify_func is None:
        def verify_func(c):
            result = verify_lean_code(c, config)
            _, is_valid_with_sorry, errors = check_verification_result(result)
            return {
                "is_valid_with_sorry": is_valid_with_sorry,
                "error_messages": errors,
            }
    
    # 保留所有 LLM response
    all_llm_responses = []
    
    # 当前的代码和错误（每次失败后会更新）
    current_code = code
    current_errors = error_messages
    
    # =========================================================================
    # Stage 1: Fix（尝试真正修复，不用 sorry）
    # =========================================================================
    if config.use_llm and config.fix_max_retries > 0:
        for attempt in range(config.fix_max_retries):
            fixed_code, success, llm_response = llm_revise(
                current_code, current_errors, config, strategy="fix"
            )
            
            if llm_response:
                all_llm_responses.append(f"[Fix attempt {attempt + 1}]\n{llm_response}")
            
            if success:
                # 确保只修改 EVOLVE-BLOCK 内的代码
                fixed_code = _ensure_changes_within_block(code, fixed_code)
                
                # 验证修复后的代码
                try:
                    result = verify_func(fixed_code)
                    if result.get('is_valid_with_sorry', False):
                        logger.info(f"[Revise] Fix successful (attempt {attempt + 1}/{config.fix_max_retries})")
                        return fixed_code, True, "fix", "\n---\n".join(all_llm_responses)
                    else:
                        # 验证失败，用新的代码和错误继续
                        new_errors = result.get('error_messages', [])
                        if new_errors and fixed_code != current_code:
                            current_code = fixed_code
                            current_errors = new_errors
                except Exception:
                    pass
    
    # =========================================================================
    # Stage 2: Eliminate（使用 sorry 消除错误）
    # =========================================================================
    if config.use_llm and config.eliminate_max_retries > 0:
        for attempt in range(config.eliminate_max_retries):
            fixed_code, success, llm_response = llm_revise(
                current_code, current_errors, config, strategy="eliminate"
            )
            
            if llm_response:
                all_llm_responses.append(f"[Eliminate attempt {attempt + 1}]\n{llm_response}")
            
            if success:
                # 确保只修改 EVOLVE-BLOCK 内的代码
                fixed_code = _ensure_changes_within_block(code, fixed_code)
                
                # 验证修复后的代码
                try:
                    result = verify_func(fixed_code)
                    if result.get('is_valid_with_sorry', False):
                        logger.info(f"[Revise] Eliminate successful (attempt {attempt + 1}/{config.eliminate_max_retries})")
                        return fixed_code, True, "eliminate", "\n---\n".join(all_llm_responses)
                    else:
                        # 验证失败，用新的代码和错误继续
                        new_errors = result.get('error_messages', [])
                        if new_errors and fixed_code != current_code:
                            current_code = fixed_code
                            current_errors = new_errors
                except Exception:
                    pass
    
    # =========================================================================
    # Stage 3: Sorry 硬替换（回退方案）
    # =========================================================================
    if config.fallback_to_sorry:
        # 用当前最新的代码和错误进行 sorry 替换
        revised_code, success, revised_names = sorry_revise(current_code, current_errors)
        
        if success:
            # 确保只修改 EVOLVE-BLOCK 内的代码
            revised_code = _ensure_changes_within_block(code, revised_code)
            
            logger.info(f"[Revise] Sorry replacement: {revised_names}")
            return revised_code, True, revised_names, "\n---\n".join(all_llm_responses)
    
    return code, False, [], "\n---\n".join(all_llm_responses)


# 导出主要接口
__all__ = [
    # 主函数
    'revise_proof',
    # 修复策略
    'llm_revise',
    'sorry_revise',
    # Lean 验证
    'verify_lean_code',
    'is_code_valid',
    'check_verification_result',
    # 配置
    'get_config',
]
