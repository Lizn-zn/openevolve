"""
Revise 模块配置管理
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LeanServerConfig:
    """Lean 验证服务器配置"""
    url: str = "http://localhost:8000"
    timeout: int = 60
    timeout_buffer: int = 60


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_base: str = ""
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    use_azure_ad: bool = False
    managed_identity_client_id: Optional[str] = None
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: int = 120


@dataclass
class ReviseConfig:
    """Revise 模块配置"""
    lean_server: LeanServerConfig = field(default_factory=LeanServerConfig)
    use_llm: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    max_retries: int = 1
    fallback_to_sorry: bool = True
    debug: bool = False
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ReviseConfig":
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 解析 Lean Server 配置
        lean_data = data.get('lean_server', {})
        lean_config = LeanServerConfig(
            url=lean_data.get('url', 'http://localhost:8000'),
            timeout=lean_data.get('timeout', 60),
            timeout_buffer=lean_data.get('timeout_buffer', 60),
        )
        
        # 解析 LLM 配置
        llm_data = data.get('llm', {})
        
        # 处理环境变量引用（如 ${OPENAI_API_KEY}）
        api_key = llm_data.get('api_key')
        if api_key and api_key.startswith('${') and api_key.endswith('}'):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var)
        
        llm_config = LLMConfig(
            api_base=llm_data.get('api_base', ''),
            api_key=api_key,
            api_version=llm_data.get('api_version'),
            use_azure_ad=llm_data.get('use_azure_ad', False),
            managed_identity_client_id=llm_data.get('managed_identity_client_id'),
            model=llm_data.get('model', 'gpt-4o'),
            temperature=llm_data.get('temperature', 0.2),
            max_tokens=llm_data.get('max_tokens', 8192),
            timeout=llm_data.get('timeout', 120),
        )
        
        return cls(
            lean_server=lean_config,
            use_llm=data.get('use_llm', True),
            llm=llm_config,
            max_retries=data.get('max_retries', 1),
            fallback_to_sorry=data.get('fallback_to_sorry', True),
            debug=data.get('debug', False),
        )


# 全局配置实例（延迟加载）
_config: Optional[ReviseConfig] = None


def get_config() -> ReviseConfig:
    """获取配置实例（单例模式）"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            _config = ReviseConfig.from_yaml(str(config_path))
        else:
            _config = ReviseConfig()
    return _config


def reload_config() -> ReviseConfig:
    """重新加载配置"""
    global _config
    _config = None
    return get_config()

