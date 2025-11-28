"""
Adapted from SakanaAI/ShinkaEvolve (Apache-2.0 License)
Original source: https://github.com/SakanaAI/ShinkaEvolve/blob/main/shinka/llm/embedding.py
"""

import os
import openai
from typing import Union, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import Azure identity libraries (optional dependency)
try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False
    logger.warning(
        "Azure identity libraries not available. Install with: pip install azure-identity"
    )

M = 1_000_000

OPENAI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]

AZURE_EMBEDDING_MODELS = [
    "azure-text-embedding-3-small",
    "azure-text-embedding-3-large",
]

OPENROUTER_EMBEDDING_MODELS = [
    "openrouter-openai/text-embedding-3-small",
    "openrouter-openai/text-embedding-3-large",
]

GEMINI_EMBEDDING_MODELS = [
    "gemini-embedding-exp-03-07",
    "gemini-embedding-001",
]

OPENAI_EMBEDDING_COSTS = {
    "text-embedding-3-small": 0.02 / M,
    "text-embedding-3-large": 0.13 / M,
}

# Gemini embedding costs (approximate - check current pricing)
GEMINI_EMBEDDING_COSTS = {
    "gemini-embedding-exp-03-07": 0.0 / M,  # Experimental model, often free
    "gemini-embedding-001": 0.0 / M,  # Check current pricing
}


def get_client_model(model_name: str, llm_config=None) -> Tuple[Union[openai.OpenAI, openai.AzureOpenAI, str], str]:
    """
    Get the appropriate client and model name for the embedding model.
    
    Args:
        model_name: The embedding model name
        llm_config: Optional LLM configuration object to read Azure settings from
    
    Returns:
        Tuple of (client, model_to_use)
    """
    if model_name in OPENAI_EMBEDDING_MODELS:
        client = openai.OpenAI()
        model_to_use = model_name
    elif model_name in AZURE_EMBEDDING_MODELS:
        # Get rid of the azure- prefix
        model_to_use = model_name.split("azure-")[-1]
        
        api_base = getattr(llm_config, "api_base", None)
        managed_identity_client_id = getattr(llm_config, "managed_identity_client_id", None)
        api_version = getattr(llm_config, "api_version", None) 
        
        if not api_base or not managed_identity_client_id or not api_version:
            raise ValueError("Azure endpoint not found. Set api_base in config")
                
        # Use Azure AD authentication with managed identity
        credential = DefaultAzureCredential(managed_identity_client_id = managed_identity_client_id)
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        client = openai.AzureOpenAI(
            azure_endpoint=api_base,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        logger.info(f"Initialized Azure OpenAI client with Azure AD authentication (model: {model_to_use})")
    elif model_name in OPENROUTER_EMBEDDING_MODELS:
        # Get rid of the openrouter- prefix
        model_to_use = model_name.split("openrouter-")[-1]
        
        # Get OpenRouter configuration: 优先使用 llm_config，环境变量作为后备
        api_key = os.getenv("OPENROUTER_API_KEY") 
        
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        logger.info(f"Initialized OpenRouter client (model: {model_to_use})")
    elif model_name in GEMINI_EMBEDDING_MODELS:
        # Configure Gemini API
        try:
            import google.generativeai as genai
        except ImportError:
            raise ValueError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set for Gemini models")
        genai.configure(api_key=api_key)
        client = "gemini"  # Use string identifier for Gemini
        model_to_use = model_name
    else:
        raise ValueError(f"Invalid embedding model: {model_name}")

    return client, model_to_use


class EmbeddingClient:
    def __init__(
        self, model_name: str = "text-embedding-3-small", llm_config=None, verbose: bool = False
    ):
        """
        Initialize the EmbeddingClient.

        Args:
            model_name (str): The OpenAI, Azure, or Gemini embedding model name to use.
            llm_config: Optional LLM configuration object to read Azure settings from.
            verbose (bool): Whether to enable verbose logging.
        """
        self.client, self.model = get_client_model(model_name, llm_config)
        self.model_name = model_name
        self.verbose = verbose

    def get_embedding(
        self, code: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Computes the text embedding for a code string.

        Args:
            code (str, list[str]): The code as a string or list of strings.

        Returns:
            list: Embedding vector(s) for the code. Returns List[float] for single input,
                  List[List[float]] for multiple inputs. Returns empty list on error.
        """
        if isinstance(code, str):
            code = [code]
            single_code = True
        else:
            single_code = False
        
        # Handle Gemini models
        if self.model_name in GEMINI_EMBEDDING_MODELS:
            try:
                import google.generativeai as genai
                
                embeddings = []
                for text in code:
                    result = genai.embed_content(
                        model=f"models/{self.model}",
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                
                if single_code:
                    return embeddings[0] if embeddings else []
                else:
                    return embeddings
            except Exception as e:
                logger.error(f"Error getting Gemini embedding: {e}")
                if single_code:
                    return []
                else:
                    return [[]]
        
        # Handle OpenAI and Azure models (same interface)
        try:
            response = self.client.embeddings.create(
                model=self.model, input=code, encoding_format="float"
            )
            # Extract embedding from response
            if single_code:
                return response.data[0].embedding
            else:
                return [d.embedding for d in response.data]
        except Exception as e:
            logger.info(f"Error getting embedding: {e}")
            if single_code:
                return []
            else:
                return [[]]
