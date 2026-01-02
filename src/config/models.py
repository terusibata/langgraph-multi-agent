"""Model configurations and definitions."""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an LLM model."""

    model_id: str
    provider: Literal["bedrock", "openai", "anthropic"]
    input_cost_per_1k: float
    output_cost_per_1k: float
    cache_write_cost_per_1k: float  # Cost for creating cache (cache_creation)
    cache_read_cost_per_1k: float   # Cost for reading from cache (cache_read)
    max_tokens: int
    context_window: int
    supports_caching: bool = True   # Whether model supports prompt caching


# Bedrock model configurations
# Pricing based on AWS Bedrock pricing (as of 2025)
# Cache write cost: 1.25x standard input cost (5-minute TTL)
# Cache read cost: 0.1x standard input cost (90% discount)
# Reference: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
BEDROCK_MODELS: dict[str, ModelConfig] = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": ModelConfig(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cache_write_cost_per_1k=0.00375,  # 1.25x input cost
        cache_read_cost_per_1k=0.0003,    # 0.1x input cost
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cache_write_cost_per_1k=0.00375,  # 1.25x input cost
        cache_read_cost_per_1k=0.0003,    # 0.1x input cost
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelConfig(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        cache_write_cost_per_1k=0.001,    # 1.25x input cost
        cache_read_cost_per_1k=0.00008,   # 0.1x input cost
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-opus-20240229-v1:0": ModelConfig(
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        cache_write_cost_per_1k=0.01875,  # 1.25x input cost
        cache_read_cost_per_1k=0.0015,    # 0.1x input cost
        max_tokens=4096,
        context_window=200000,
    ),
    "anthropic.claude-3-sonnet-20240229-v1:0": ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        cache_write_cost_per_1k=0.00375,  # 1.25x input cost
        cache_read_cost_per_1k=0.0003,    # 0.1x input cost
        max_tokens=4096,
        context_window=200000,
    ),
    "anthropic.claude-3-haiku-20240307-v1:0": ModelConfig(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        cache_write_cost_per_1k=0.0003125,  # 1.25x input cost
        cache_read_cost_per_1k=0.000025,    # 0.1x input cost
        max_tokens=4096,
        context_window=200000,
    ),
}


def normalize_model_id(model_id: str) -> str:
    """
    Normalize model ID by removing cross-region inference profile prefixes.

    AWS Bedrock supports cross-region inference profiles with prefixes like:
    - us.anthropic.claude-3-7-sonnet-20250219-v1:0
    - eu.anthropic.claude-3-5-sonnet-20241022-v2:0

    This function removes the regional prefix to get the base model ID.

    Args:
        model_id: Original model ID (may include regional prefix)

    Returns:
        Normalized model ID without regional prefix
    """
    # Pattern: region-prefix.provider.model-name
    # Examples: us.anthropic.*, eu.anthropic.*, ap.anthropic.*
    pattern = r'^(?:us|eu|ap|apac)\.(anthropic\..+)$'
    match = re.match(pattern, model_id)

    if match:
        return match.group(1)

    return model_id


def get_model_config(model_id: str) -> ModelConfig | None:
    """
    Get model configuration by ID.

    Supports both standard model IDs and cross-region inference profile IDs.
    Cross-region IDs (e.g., us.anthropic.*) are automatically normalized.

    Args:
        model_id: Model ID (with or without regional prefix)

    Returns:
        ModelConfig if found, None otherwise
    """
    # First try exact match
    config = BEDROCK_MODELS.get(model_id)
    if config:
        return config

    # Try normalized model ID (remove regional prefix)
    normalized_id = normalize_model_id(model_id)
    return BEDROCK_MODELS.get(normalized_id)


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for token usage (legacy function, does not account for caching).

    For accurate caching cost calculation, use calculate_cost_with_cache().
    """
    config = get_model_config(model_id)
    if not config:
        return 0.0

    input_cost = (input_tokens / 1000) * config.input_cost_per_1k
    output_cost = (output_tokens / 1000) * config.output_cost_per_1k
    return input_cost + output_cost


def calculate_cost_with_cache(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> tuple[float, dict]:
    """
    Calculate cost for token usage with prompt caching support.

    Args:
        model_id: Model ID (with or without regional prefix)
        input_tokens: Total input tokens from LangChain response
        output_tokens: Output tokens
        cache_creation_tokens: Tokens written to cache (from input_token_details.cache_creation)
        cache_read_tokens: Tokens read from cache (from input_token_details.cache_read)

    Returns:
        Tuple of (total_cost_usd, breakdown_dict)
        breakdown_dict contains:
            - normal_input_tokens: Regular input tokens
            - cache_creation_tokens: Tokens written to cache
            - cache_read_tokens: Tokens read from cache
            - output_tokens: Output tokens
            - normal_input_cost: Cost for regular input
            - cache_creation_cost: Cost for cache writes
            - cache_read_cost: Cost for cache reads
            - output_cost: Cost for output
            - total_cost: Total cost in USD

    Note:
        According to LangChain documentation, the correct calculation is:
        normal_input = input_tokens - cache_creation - cache_read
    """
    config = get_model_config(model_id)
    if not config:
        return 0.0, {
            "normal_input_tokens": input_tokens,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": output_tokens,
            "normal_input_cost": 0.0,
            "cache_creation_cost": 0.0,
            "cache_read_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
        }

    # Calculate normal input tokens (exclude cached portions)
    normal_input_tokens = input_tokens - cache_creation_tokens - cache_read_tokens

    # Calculate costs for each token type
    normal_input_cost = (normal_input_tokens / 1000) * config.input_cost_per_1k
    cache_creation_cost = (cache_creation_tokens / 1000) * config.cache_write_cost_per_1k
    cache_read_cost = (cache_read_tokens / 1000) * config.cache_read_cost_per_1k
    output_cost = (output_tokens / 1000) * config.output_cost_per_1k

    total_cost = normal_input_cost + cache_creation_cost + cache_read_cost + output_cost

    breakdown = {
        "normal_input_tokens": normal_input_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "normal_input_cost": normal_input_cost,
        "cache_creation_cost": cache_creation_cost,
        "cache_read_cost": cache_read_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

    return total_cost, breakdown


def extract_usage_from_response(response: any) -> dict:
    """
    Extract token usage information from LangChain response.

    Supports both ChatBedrock and ChatBedrockConverse responses.

    Args:
        response: LangChain response object (AIMessage or similar)

    Returns:
        Dictionary with token usage information:
            - input_tokens: Total input tokens
            - output_tokens: Output tokens
            - cache_creation_tokens: Tokens written to cache
            - cache_read_tokens: Tokens read from cache
    """
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }

    # Try to get usage_metadata from response
    if hasattr(response, "usage_metadata"):
        metadata = response.usage_metadata
        usage["input_tokens"] = metadata.get("input_tokens", 0)
        usage["output_tokens"] = metadata.get("output_tokens", 0)

        # Extract cache information if available
        input_details = metadata.get("input_token_details", {})
        if input_details:
            usage["cache_creation_tokens"] = input_details.get("cache_creation", 0)
            usage["cache_read_tokens"] = input_details.get("cache_read", 0)

    # Fallback: try response_metadata
    elif hasattr(response, "response_metadata"):
        metadata = response.response_metadata.get("usage", {})
        usage["input_tokens"] = metadata.get("input_tokens", 0)
        usage["output_tokens"] = metadata.get("output_tokens", 0)

        # Some responses may have cache info in different location
        if "cache_creation_input_tokens" in metadata:
            usage["cache_creation_tokens"] = metadata.get("cache_creation_input_tokens", 0)
        if "cache_read_input_tokens" in metadata:
            usage["cache_read_tokens"] = metadata.get("cache_read_input_tokens", 0)

    return usage


def supports_prompt_caching(model_id: str) -> bool:
    """
    Check if a model supports prompt caching.

    Args:
        model_id: Model ID (with or without regional prefix)

    Returns:
        True if model supports prompt caching, False otherwise
    """
    config = get_model_config(model_id)
    if not config:
        return False
    return config.supports_caching


def should_use_prompt_caching(model_id: str, enable_caching: bool = True) -> bool:
    """
    Determine if prompt caching should be used for a given model.

    Args:
        model_id: Model ID (with or without regional prefix)
        enable_caching: Global flag to enable/disable caching (from settings)

    Returns:
        True if caching should be used, False otherwise
    """
    if not enable_caching:
        return False
    return supports_prompt_caching(model_id)


def get_available_models() -> list[dict]:
    """Get list of available models."""
    return [
        {
            "model_id": config.model_id,
            "provider": config.provider,
            "max_tokens": config.max_tokens,
            "context_window": config.context_window,
            "supports_caching": config.supports_caching,
        }
        for config in BEDROCK_MODELS.values()
    ]


# Prompt caching supported models (as of 2025)
# Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
PROMPT_CACHING_SUPPORTED_MODELS = {
    # Claude models (Generally Available)
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Preview only
    # Amazon Nova models (Generally Available)
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
    "amazon.nova-premier-v1:0",
}
