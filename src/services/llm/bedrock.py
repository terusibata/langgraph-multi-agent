"""Amazon Bedrock LLM service."""

from typing import Any

from botocore.config import Config
from langchain_aws import ChatBedrock
import structlog

from src.config import get_settings
from src.config.models import get_model_config, calculate_cost

logger = structlog.get_logger()


class BedrockLLMService:
    """Service for managing Bedrock LLM instances."""

    def __init__(self):
        """Initialize the service."""
        self.settings = get_settings()
        self._models: dict[str, ChatBedrock] = {}
        self._boto_config = self._create_boto_config()

    def _create_boto_config(self) -> Config | None:
        """
        Create boto3 Config object with proxy settings if configured.

        Returns:
            Config object with proxy settings, or None if no proxy is configured
        """
        # Build proxies dict if any proxy is configured
        proxies = {}
        if self.settings.http_proxy:
            proxies["http"] = self.settings.http_proxy
        if self.settings.https_proxy:
            proxies["https"] = self.settings.https_proxy

        # If no proxies configured, return None
        if not proxies:
            return None

        # Build proxies_config for advanced settings
        proxies_config = {}
        if self.settings.proxy_ca_bundle:
            proxies_config["proxy_ca_bundle"] = self.settings.proxy_ca_bundle
        if self.settings.proxy_client_cert:
            proxies_config["proxy_client_cert"] = self.settings.proxy_client_cert
        proxies_config["proxy_use_forwarding_for_https"] = (
            self.settings.proxy_use_forwarding_for_https
        )

        config_kwargs: dict[str, Any] = {
            "proxies": proxies,
        }
        if proxies_config:
            config_kwargs["proxies_config"] = proxies_config

        logger.info(
            "boto3_proxy_configured",
            http_proxy=self.settings.http_proxy,
            https_proxy=self.settings.https_proxy,
            has_ca_bundle=bool(self.settings.proxy_ca_bundle),
            has_client_cert=bool(self.settings.proxy_client_cert),
        )

        return Config(**config_kwargs)

    def get_model(self, model_id: str | None = None) -> ChatBedrock:
        """
        Get or create a Bedrock model instance.

        Args:
            model_id: Model ID to use (defaults to settings)

        Returns:
            ChatBedrock instance
        """
        model_id = model_id or self.settings.default_model_id

        if model_id not in self._models:
            config = get_model_config(model_id)
            max_tokens = config.max_tokens if config else 4096

            # Prepare ChatBedrock kwargs
            bedrock_kwargs: dict[str, Any] = {
                "model_id": model_id,
                "region_name": self.settings.aws_region,
                "model_kwargs": {
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            }

            # Add boto config if proxy is configured
            if self._boto_config:
                bedrock_kwargs["config"] = self._boto_config

            self._models[model_id] = ChatBedrock(**bedrock_kwargs)

            logger.debug("bedrock_model_created", model_id=model_id)

        return self._models[model_id]

    def get_streaming_model(self, model_id: str | None = None) -> ChatBedrock:
        """
        Get a model configured for streaming.

        Args:
            model_id: Model ID to use

        Returns:
            ChatBedrock instance with streaming enabled
        """
        model_id = model_id or self.settings.default_model_id
        streaming_key = f"{model_id}_streaming"

        if streaming_key not in self._models:
            config = get_model_config(model_id)
            max_tokens = config.max_tokens if config else 4096

            # Prepare ChatBedrock kwargs
            bedrock_kwargs: dict[str, Any] = {
                "model_id": model_id,
                "region_name": self.settings.aws_region,
                "streaming": True,
                "model_kwargs": {
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            }

            # Add boto config if proxy is configured
            if self._boto_config:
                bedrock_kwargs["config"] = self._boto_config

            self._models[streaming_key] = ChatBedrock(**bedrock_kwargs)

            logger.debug("bedrock_streaming_model_created", model_id=model_id)

        return self._models[streaming_key]

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for token usage.

        Args:
            model_id: Model ID used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        return calculate_cost(model_id, input_tokens, output_tokens)


# Global service instance
_llm_service: BedrockLLMService | None = None


def get_llm_service() -> BedrockLLMService:
    """Get the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = BedrockLLMService()
    return _llm_service


def get_llm(model_id: str | None = None) -> ChatBedrock:
    """
    Convenience function to get an LLM instance.

    Args:
        model_id: Optional model ID

    Returns:
        ChatBedrock instance
    """
    return get_llm_service().get_model(model_id)


def get_streaming_llm(model_id: str | None = None) -> ChatBedrock:
    """
    Convenience function to get a streaming LLM instance.

    Args:
        model_id: Optional model ID

    Returns:
        ChatBedrock instance with streaming
    """
    return get_llm_service().get_streaming_model(model_id)
