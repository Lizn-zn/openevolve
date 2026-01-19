"""
Lean 验证服务器接口

封装对 Lean 验证服务器的调用。
"""

import logging
from typing import Dict, Tuple, List, Optional

import requests

from .config import get_config, ReviseConfig

logger = logging.getLogger("openevolve.revise.verify")


def verify_lean_code(
    code: str,
    config: ReviseConfig = None,
    timeout: int = None,
) -> Dict:
    """
    向 Lean 验证服务器发送验证请求
    
    Args:
        code: Lean 代码
        config: 配置（可选，默认使用全局配置）
        timeout: 超时时间（可选，默认使用配置中的值）
        
    Returns:
        验证结果字典，包含:
        - is_valid_no_sorry: 是否无 sorry 且编译通过
        - is_valid_with_sorry: 是否允许 sorry 编译通过
        - error_message: (has_error, error_list)
        - response: 原始响应
    """
    if config is None:
        config = get_config()
    
    lean_cfg = config.lean_server
    url = f"{lean_cfg.url.rstrip('/')}/verify"
    
    if timeout is None:
        timeout = lean_cfg.timeout
    
    payload = {
        "code": code,
        "timeout": timeout,
        "max_retries": 1,
        "overflow_threshold": 2,
    }
    
    try:
        resp = requests.post(
            url, 
            json=payload, 
            timeout=timeout + lean_cfg.timeout_buffer
        )
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        logger.warning(f"[Lean Verify] Request timeout after {timeout}s")
        return {"error": "timeout", "is_valid": False}
    except requests.RequestException as e:
        logger.error(f"[Lean Verify] Request failed: {e}")
        return {"error": str(e), "is_valid": False}


def check_verification_result(result: Dict) -> Tuple[bool, bool, List[str]]:
    """
    检查验证结果
    
    Args:
        result: verify_lean_code 的返回值
        
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


def is_code_valid(code: str, config: ReviseConfig = None) -> bool:
    """
    检查代码是否可以编译通过（允许 sorry）
    
    Args:
        code: Lean 代码
        config: 配置
        
    Returns:
        True 如果代码可以编译通过
    """
    result = verify_lean_code(code, config)
    _, is_valid_with_sorry, _ = check_verification_result(result)
    return is_valid_with_sorry


def get_compile_errors(code: str, config: ReviseConfig = None) -> List[str]:
    """
    获取代码的编译错误列表
    
    Args:
        code: Lean 代码
        config: 配置
        
    Returns:
        错误消息列表
    """
    result = verify_lean_code(code, config)
    _, _, errors = check_verification_result(result)
    return errors

