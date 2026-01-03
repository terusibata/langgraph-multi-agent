"""Application settings configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/langgraph_agents",
        description="PostgreSQL connection URL (async)",
    )
    database_url_sync: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/langgraph_agents",
        description="PostgreSQL connection URL (sync, for LangGraph checkpoint)",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # AWS
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock",
    )

    # Proxy settings (optional)
    http_proxy: str | None = Field(
        default=None,
        description="HTTP proxy URL (e.g., http://proxy.example.com:8080)",
    )
    https_proxy: str | None = Field(
        default=None,
        description="HTTPS proxy URL (e.g., https://proxy.example.com:8080)",
    )
    proxy_ca_bundle: str | None = Field(
        default=None,
        description="Path to custom CA certificate bundle for proxy",
    )
    proxy_client_cert: str | None = Field(
        default=None,
        description="Path to client certificate for proxy authentication",
    )
    proxy_use_forwarding_for_https: bool = Field(
        default=True,
        description="Use CONNECT method for HTTPS proxying",
    )

    # Authentication
    access_key_secret: str = Field(
        default="dev-secret-key-change-in-production",
        description="Secret key for signing/verifying access keys",
    )
    access_key_algorithm: str = Field(
        default="HS256",
        description="Algorithm for access key signing",
    )

    # Models
    default_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        description="Default model ID for MainAgent",
    )
    sub_agent_model_id: str = Field(
        default="anthropic.claude-3-5-haiku-20241022-v1:0",
        description="Model ID for SubAgents",
    )

    # Context limits
    context_max_tokens: int = Field(
        default=200000,
        description="Maximum context tokens",
    )
    context_warning_threshold: int = Field(
        default=80,
        description="Context usage warning threshold (%)",
    )
    context_lock_threshold: int = Field(
        default=95,
        description="Context usage lock threshold (%)",
    )

    # Timeouts
    parallel_timeout_seconds: int = Field(
        default=30,
        description="Timeout for parallel agent execution",
    )
    sub_agent_timeout_seconds: int = Field(
        default=60,
        description="Timeout for individual SubAgent execution",
    )
    tool_timeout_seconds: int = Field(
        default=30,
        description="Timeout for tool calls",
    )
    session_timeout_seconds: int = Field(
        default=300,
        description="Overall session timeout",
    )

    # Retry settings
    default_max_retries: int = Field(
        default=3,
        description="Default maximum retry attempts",
    )
    retry_backoff_seconds: float = Field(
        default=1.0,
        description="Backoff time between retries",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format",
    )

    # API
    api_prefix: str = Field(
        default="/api/v1",
        description="API route prefix",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # Feature flags
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint",
    )
    enable_request_logging: bool = Field(
        default=True,
        description="Enable request/response logging",
    )

    @property
    def context_warning_tokens(self) -> int:
        """Calculate warning threshold in tokens."""
        return int(self.context_max_tokens * self.context_warning_threshold / 100)

    @property
    def context_lock_tokens(self) -> int:
        """Calculate lock threshold in tokens."""
        return int(self.context_max_tokens * self.context_lock_threshold / 100)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
