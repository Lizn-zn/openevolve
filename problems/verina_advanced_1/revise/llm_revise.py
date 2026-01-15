"""
LLM 智能修复模块

使用 LLM 来理解错误并尝试修复 Lean 代码。
"""

import asyncio
import logging
from typing import Tuple

import openai

from .config import get_config, ReviseConfig
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .utils import (
    extract_lean_code_from_response,
    format_error_messages,
    validate_code_structure,
    force_preserve_outside_block,
)

logger = logging.getLogger(__name__)


def _create_client(config: ReviseConfig):
    """创建 OpenAI 客户端"""
    llm = config.llm
    
    if llm.use_azure_ad:
        # Azure AD 认证
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError:
            raise ImportError(
                "Azure identity libraries required. Install with: pip install azure-identity"
            )
        
        credential = DefaultAzureCredential(
            managed_identity_client_id=llm.managed_identity_client_id
        )
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        client = openai.AzureOpenAI(
            azure_endpoint=llm.api_base,
            azure_ad_token_provider=token_provider,
            api_version=llm.api_version,
            timeout=llm.timeout,
        )
    else:
        # 标准 OpenAI 或兼容 API
        client = openai.OpenAI(
            api_key=llm.api_key,
            base_url=llm.api_base if llm.api_base else None,
            timeout=llm.timeout,
        )
    
    return client


async def llm_revise_async(
    code: str,
    error_messages: list,
    config: ReviseConfig = None,
) -> Tuple[str, bool]:
    """
    使用 LLM 异步修复 Lean 代码
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        config: 配置（可选，默认使用全局配置）
        
    Returns:
        (fixed_code, success)
    """
    if config is None:
        config = get_config()
    
    if not error_messages:
        return code, False
    
    # 格式化错误信息
    errors_text = format_error_messages(error_messages)
    
    # 构建 prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        code=code,
        errors=errors_text,
    )
    
    try:
        # 创建客户端
        client = _create_client(config)
        
        # 获取模型名（去掉 azure- 前缀）
        model = config.llm.model
        if model.lower().startswith("azure-"):
            model = model.split("azure-", 1)[-1]
        
        # 判断是否是推理模型（需要使用不同的参数）
        REASONING_MODEL_PREFIXES = (
            "o1-", "o1", "o3-", "o3", "o4-",
            "gpt-5-", "gpt-5", "gpt-oss-",
        )
        is_reasoning_model = model.lower().startswith(REASONING_MODEL_PREFIXES)
        
        # 构建请求参数
        if is_reasoning_model:
            # 推理模型使用 max_completion_tokens，不支持 temperature
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": config.llm.max_tokens,
            }
        else:
            # 标准模型
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
            }
        
        # 调用 API（同步调用，在线程池中执行）
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**api_params)
        )
        
        result = response.choices[0].message.content
        
        if config.debug:
            logger.debug(f"[LLM Revise] Response:\n{result[:500]}...")
        
        # 提取代码
        fixed_code = extract_lean_code_from_response(result)
        
        if not fixed_code:
            logger.warning("[LLM Revise] Could not extract code from response")
            return code, False
        
        # 强制保留 EVOLVE-BLOCK 外的原始代码
        # 这样即使 LLM 修改了 BLOCK 外的内容，我们也只取 BLOCK 内的修改
        fixed_code = force_preserve_outside_block(code, fixed_code)
        
        # 验证代码结构（现在应该总是通过，因为我们强制保留了外部结构）
        is_valid, error_msg = validate_code_structure(code, fixed_code)
        if not is_valid:
            logger.warning(f"[LLM Revise] Invalid code structure: {error_msg}")
            return code, False
        
        # 检查是否有实际修改
        if fixed_code.strip() == code.strip():
            logger.info("[LLM Revise] No changes made by LLM")
            return code, False
        
        logger.info("[LLM Revise] Successfully fixed code")
        return fixed_code, True
        
    except Exception as e:
        logger.error(f"[LLM Revise] API call failed: {e}")
        return code, False


def llm_revise(
    code: str,
    error_messages: list,
    config: ReviseConfig = None,
) -> Tuple[str, bool]:
    """
    使用 LLM 同步修复 Lean 代码
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        config: 配置（可选）
        
    Returns:
        (fixed_code, success)
    """
    return asyncio.run(llm_revise_async(code, error_messages, config))

