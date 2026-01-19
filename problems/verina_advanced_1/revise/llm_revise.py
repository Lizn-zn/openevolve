"""
LLM 智能修复模块

使用 LLM 进行最小化 sorry 填充，通过 SEARCH/REPLACE 格式应用修改。
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Tuple

import openai

from .config import get_config, ReviseConfig
from .prompts import (
    FIX_SYSTEM_PROMPT,
    FIX_USER_PROMPT_TEMPLATE,
    ELIMINATE_SYSTEM_PROMPT,
    ELIMINATE_USER_PROMPT_TEMPLATE,
)
from .utils import format_error_messages

# 导入项目的 diff 工具
_OPENEVOLVE_ROOT = str(Path(__file__).parent.parent.parent.parent)
if _OPENEVOLVE_ROOT not in sys.path:
    sys.path.insert(0, _OPENEVOLVE_ROOT)

from openevolve.utils.code_utils import apply_diff, extract_diffs

logger = logging.getLogger("openevolve.revise.llm")

# SEARCH/REPLACE 的正则模式
DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"


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
    strategy: str = "eliminate",
) -> Tuple[str, bool, str]:
    """
    使用 LLM 异步修复 Lean 代码
    
    通过 SEARCH/REPLACE 格式进行最小化修改。
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        config: 配置（可选，默认使用全局配置）
        strategy: 修复策略
            - "fix": 尝试真正修复错误，不使用 sorry
            - "eliminate": 使用 sorry 消除错误（默认）
        
    Returns:
        (fixed_code, success, llm_response)
    """
    if config is None:
        config = get_config()
    
    if not error_messages:
        return code, False, ""
    
    # 根据策略选择 prompt
    if strategy == "fix":
        system_prompt = FIX_SYSTEM_PROMPT
        user_template = FIX_USER_PROMPT_TEMPLATE
    else:
        system_prompt = ELIMINATE_SYSTEM_PROMPT
        user_template = ELIMINATE_USER_PROMPT_TEMPLATE
    
    # 格式化错误信息
    errors_text = format_error_messages(error_messages)
    
    # 构建 prompt
    user_prompt = user_template.format(
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": config.llm.max_tokens,
            }
        else:
            # 标准模型
            api_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
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
        
        llm_response = response.choices[0].message.content
        
        if config.debug:
            logger.debug(f"[LLM Revise] Response:\n{llm_response}")
        
        # 提取 diff blocks
        diff_blocks = extract_diffs(llm_response, DIFF_PATTERN)
        
        if not diff_blocks:
            return code, False, llm_response
        
        # 应用 diffs
        fixed_code, all_applied, block_results = apply_diff(code, llm_response, DIFF_PATTERN)
        
        # 检查应用结果
        applied_count = sum(1 for r in block_results if r["applied"])
        
        if applied_count == 0:
            return code, False, llm_response
        
        # 检查是否有实际修改
        if fixed_code.strip() == code.strip():
            return code, False, llm_response
        
        return fixed_code, True, llm_response
        
    except Exception as e:
        logger.error(f"[LLM Revise] API call failed: {e}")
        return code, False, ""


def llm_revise(
    code: str,
    error_messages: list,
    config: ReviseConfig = None,
    strategy: str = "eliminate",
) -> Tuple[str, bool, str]:
    """
    使用 LLM 同步修复 Lean 代码
    
    Args:
        code: 原始 Lean 代码
        error_messages: 错误消息列表
        config: 配置（可选）
        strategy: 修复策略 ("fix" 或 "eliminate")
        
    Returns:
        (fixed_code, success, llm_response)
    """
    return asyncio.run(llm_revise_async(code, error_messages, config, strategy))
