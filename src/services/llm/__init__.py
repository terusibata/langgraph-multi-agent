"""LLM service module."""

from src.services.llm.bedrock import BedrockLLMService, get_llm

__all__ = ["BedrockLLMService", "get_llm"]
