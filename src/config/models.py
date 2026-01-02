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
    max_tokens: int
    context_window: int


# Bedrock model configurations
BEDROCK_MODELS: dict[str, ModelConfig] = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": ModelConfig(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": ModelConfig(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-5-haiku-20241022-v1:0": ModelConfig(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        max_tokens=8192,
        context_window=200000,
    ),
    "anthropic.claude-3-opus-20240229-v1:0": ModelConfig(
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        max_tokens=4096,
        context_window=200000,
    ),
    "anthropic.claude-3-sonnet-20240229-v1:0": ModelConfig(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=4096,
        context_window=200000,
    ),
    "anthropic.claude-3-haiku-20240307-v1:0": ModelConfig(
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        provider="bedrock",
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
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
    """Calculate cost for token usage."""
    config = get_model_config(model_id)
    if not config:
        return 0.0

    input_cost = (input_tokens / 1000) * config.input_cost_per_1k
    output_cost = (output_tokens / 1000) * config.output_cost_per_1k
    return input_cost + output_cost


def get_available_models() -> list[dict]:
    """Get list of available models."""
    return [
        {
            "model_id": config.model_id,
            "provider": config.provider,
            "max_tokens": config.max_tokens,
            "context_window": config.context_window,
        }
        for config in BEDROCK_MODELS.values()
    ]
