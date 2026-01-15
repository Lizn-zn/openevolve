"""
Revise 模块 - Lean 代码智能修复

提供两种修复策略：
1. LLM 智能修复：使用大语言模型理解错误并尝试修复
2. Sorry 替换：将失败的证明替换为 sorry（回退方案）

使用方法：
    from revise import revise_proof
    
    revised_code, success, info = revise_proof(code, error_messages)
    
    # 也可以直接使用 Lean 验证
    from revise import verify_lean_code, is_code_valid
    
    result = verify_lean_code(code)
    valid = is_code_valid(code)
"""

import logging
from typing import Tuple, List, Union

from .config import get_config
from .llm_revise import llm_revise
from .sorry_revise import sorry_revise
from .lean_verify import verify_lean_code, is_code_valid, check_verification_result

logger = logging.getLogger(__name__)


def revise_proof(
    code: str,
    error_messages: list,
    verify_func=None,
) -> Tuple[str, bool, Union[List[str], str]]:
    """
    修复 Lean 代码中的编译错误
    
    优先使用 LLM 智能修复，失败后回退到 sorry 替换。
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        verify_func: 可选的验证函数，用于验证修复后的代码是否编译通过
                    签名: (code: str) -> dict with 'is_valid_with_sorry'
                    如果不提供，默认使用内置的 verify_lean_code
        
    Returns:
        (revised_code, success, info)
        - revised_code: 修复后的代码
        - success: 是否修复成功
        - info: 修复信息（LLM 修复返回 "llm"，sorry 返回修复的定理名列表）
    """
    if not error_messages:
        return code, False, []
    
    config = get_config()
    
    # 如果没有提供验证函数，使用内置的
    if verify_func is None:
        def verify_func(c):
            result = verify_lean_code(c, config)
            return {"is_valid_with_sorry": result.get("is_valid_with_sorry", False)}
    
    # 策略 1：尝试 LLM 智能修复
    if config.use_llm:
        logger.info("[Revise] Attempting LLM-based repair...")
        
        for attempt in range(config.max_retries + 1):
            fixed_code, success = llm_revise(code, error_messages, config)
            
            if success:
                # 验证修复后的代码
                try:
                    result = verify_func(fixed_code)
                    if result.get('is_valid_with_sorry', False):
                        logger.info("[Revise] LLM repair successful and verified")
                        return fixed_code, True, "llm"
                    else:
                        logger.warning(f"[Revise] LLM repair failed verification (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"[Revise] Verification failed: {e}")
            
            if attempt < config.max_retries:
                logger.info(f"[Revise] LLM attempt {attempt + 1} failed, retrying...")
        
        logger.warning("[Revise] LLM repair failed after all attempts")
    
    # 策略 2：回退到 sorry 替换
    if config.fallback_to_sorry:
        logger.info("[Revise] Falling back to sorry replacement...")
        revised_code, success, revised_names = sorry_revise(code, error_messages)
        
        if success:
            logger.info(f"[Revise] Sorry replacement successful: {revised_names}")
            return revised_code, True, revised_names
        else:
            logger.warning("[Revise] Sorry replacement also failed")
    
    return code, False, []


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

