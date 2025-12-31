"""Base class for Tools."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
import structlog

from src.api.middleware.service_tokens import ServiceTokens

logger = structlog.get_logger()


class ToolContext(BaseModel):
    """Context passed to tools for execution."""

    service_tokens: ServiceTokens = Field(
        default_factory=ServiceTokens,
        description="Available service tokens",
    )
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(..., description="Request identifier")

    def get_token(self, service_name: str) -> str | None:
        """Get token value for a service."""
        token = self.service_tokens.get(service_name)
        return token.token if token else None

    def get_instance(self, service_name: str) -> str | None:
        """Get instance URL for a service."""
        token = self.service_tokens.get(service_name)
        return token.instance if token else None

    def require_token(self, service_name: str) -> str:
        """
        Get token value or raise an error.

        Raises:
            ValueError: If token is not available or expired
        """
        token = self.service_tokens.get(service_name)
        if not token:
            raise ValueError(f"Token for {service_name} is not available")
        if token.is_expired():
            raise ValueError(f"Token for {service_name} has expired")
        return token.token


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    data: Any = Field(default=None, description="Result data")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(default=0, description="Execution duration")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ToolParameters(BaseModel):
    """Base class for tool parameters."""

    pass


class ToolBase(ABC):
    """
    Base class for all Tools.

    Tools are used by SubAgents to interact with external services
    and data sources.
    """

    def __init__(
        self,
        name: str,
        description: str,
        required_service_token: str | None = None,
        parameters_schema: type[ToolParameters] | None = None,
        timeout_seconds: int = 30,
    ):
        """
        Initialize the Tool.

        Args:
            name: Unique identifier for the tool
            description: Description for LLM understanding
            required_service_token: Service token required (e.g., "servicenow")
            parameters_schema: Pydantic model for parameters
            timeout_seconds: Execution timeout
        """
        self.name = name
        self.description = description
        self.required_service_token = required_service_token
        self.parameters_schema = parameters_schema
        self.timeout_seconds = timeout_seconds

    def validate_token(self, context: ToolContext) -> tuple[bool, str | None]:
        """
        Validate that required service token is available.

        Args:
            context: Tool execution context

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.required_service_token:
            return True, None

        token = context.service_tokens.get(self.required_service_token)
        if not token:
            return False, f"{self.required_service_token}のトークンがありません"

        if token.is_expired():
            return False, f"{self.required_service_token}のトークンが期限切れです"

        return True, None

    async def execute_with_validation(
        self,
        params: dict,
        context: ToolContext,
    ) -> ToolResult:
        """
        Execute the tool with token validation.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution status
        """
        started_at = datetime.utcnow()

        # Validate token
        is_valid, error = self.validate_token(context)
        if not is_valid:
            return ToolResult(
                success=False,
                error=error,
                duration_ms=0,
            )

        try:
            # Execute the tool
            result = await self.execute(params, context)
            duration = datetime.utcnow() - started_at
            result.duration_ms = int(duration.total_seconds() * 1000)
            return result

        except Exception as e:
            logger.error(
                "tool_execution_error",
                tool=self.name,
                error=str(e),
                request_id=context.request_id,
            )
            duration = datetime.utcnow() - started_at
            return ToolResult(
                success=False,
                error=str(e),
                duration_ms=int(duration.total_seconds() * 1000),
            )

    @abstractmethod
    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute the tool.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution status and data
        """
        pass

    def get_langchain_tool_schema(self) -> dict:
        """
        Get the schema for LangChain tool definition.

        Returns:
            Dictionary with tool schema
        """
        schema = {
            "name": self.name,
            "description": self.description,
        }

        if self.parameters_schema:
            schema["parameters"] = self.parameters_schema.model_json_schema()

        return schema

    def to_langchain_tool(self):
        """
        Convert to a LangChain tool.

        Returns:
            LangChain Tool instance
        """
        from langchain_core.tools import StructuredTool

        async def _run(context: ToolContext, **params) -> str:
            result = await self.execute_with_validation(params, context)
            if result.success:
                return str(result.data)
            else:
                return f"Error: {result.error}"

        return StructuredTool(
            name=self.name,
            description=self.description,
            func=lambda **kwargs: None,  # Sync version not used
            coroutine=_run,
            args_schema=self.parameters_schema,
        )
