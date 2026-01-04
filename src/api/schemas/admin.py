"""Admin schemas for dynamic agent/tool management."""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Tool Definition Schemas
# =============================================================================


class ToolParameterSchema(BaseModel):
    """Schema for a single tool parameter."""

    name: str = Field(..., description="Parameter name")
    type: Literal["string", "integer", "number", "boolean", "array", "object"] = Field(
        ..., description="Parameter type"
    )
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value")
    enum: list[Any] | None = Field(default=None, description="Allowed values")
    items: dict | None = Field(default=None, description="Array item schema")
    properties: dict | None = Field(default=None, description="Object properties schema")
    # Additional field for OpenAPI 'in' parameter location
    in_: str | None = Field(default=None, alias="in", description="Parameter location (query, path, header, body)")

    model_config = {"populate_by_name": True}


class ToolExecutorConfig(BaseModel):
    """Configuration for tool execution."""

    type: Literal["http", "python", "mock"] = Field(
        ..., description="Executor type"
    )

    # HTTP executor config
    url: str | None = Field(default=None, description="HTTP endpoint URL")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = Field(
        default="POST", description="HTTP method"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="HTTP headers"
    )
    body_template: str | None = Field(
        default=None, description="Request body template (Jinja2)"
    )
    response_path: str | None = Field(
        default=None, description="JSONPath to extract result"
    )
    auth_type: Literal["none", "bearer", "api_key", "service_token"] = Field(
        default="none", description="Authentication type"
    )
    auth_config: dict = Field(
        default_factory=dict, description="Auth configuration"
    )

    # Python executor config
    module_path: str | None = Field(
        default=None, description="Python module path"
    )
    function_name: str | None = Field(
        default=None, description="Python function name"
    )
    class_name: str | None = Field(
        default=None, description="Python class name (optional)"
    )

    # Mock executor config
    mock_response: Any = Field(
        default=None, description="Mock response data"
    )
    mock_delay_ms: int = Field(
        default=0, description="Mock delay in milliseconds"
    )


class ToolDefinitionCreate(BaseModel):
    """Schema for creating a new tool definition."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique tool name (lowercase, underscores allowed)",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Tool description for LLM",
    )
    category: str = Field(
        default="general",
        description="Tool category for organization",
    )
    parameters: list[ToolParameterSchema] = Field(
        default_factory=list,
        description="Tool parameters",
    )
    executor: ToolExecutorConfig = Field(
        ..., description="Tool executor configuration"
    )
    required_service_token: str | None = Field(
        default=None,
        description="Required service token (e.g., 'servicenow')",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Execution timeout",
    )
    enabled: bool = Field(default=True, description="Whether tool is enabled")
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        reserved = ["system", "admin", "internal", "test"]
        if v in reserved:
            raise ValueError(f"Tool name '{v}' is reserved")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "weather_lookup",
                    "description": "Get current weather for a location",
                    "category": "external_api",
                    "parameters": [
                        {
                            "name": "location",
                            "type": "string",
                            "description": "City name or coordinates",
                            "required": True,
                        }
                    ],
                    "executor": {
                        "type": "http",
                        "url": "https://api.weather.com/v1/current",
                        "method": "GET",
                        "auth_type": "api_key",
                        "auth_config": {"header": "X-API-Key"},
                    },
                    "timeout_seconds": 10,
                    "enabled": True,
                }
            ]
        }
    }


class ToolDefinitionUpdate(BaseModel):
    """Schema for updating a tool definition."""

    description: str | None = Field(default=None, description="Tool description")
    category: str | None = Field(default=None, description="Tool category")
    parameters: list[ToolParameterSchema] | None = Field(
        default=None, description="Tool parameters"
    )
    executor: ToolExecutorConfig | None = Field(
        default=None, description="Tool executor configuration"
    )
    required_service_token: str | None = Field(
        default=None, description="Required service token"
    )
    timeout_seconds: int | None = Field(
        default=None, ge=1, le=300, description="Execution timeout"
    )
    enabled: bool | None = Field(default=None, description="Whether tool is enabled")
    metadata: dict | None = Field(default=None, description="Additional metadata")


class ToolDefinitionResponse(BaseModel):
    """Response schema for a tool definition."""

    id: str = Field(..., description="Tool definition ID")
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    parameters: list[ToolParameterSchema] = Field(default_factory=list, description="Tool parameters")
    executor: ToolExecutorConfig = Field(..., description="Executor configuration")
    required_service_token: str | None = Field(
        default=None, description="Required service token"
    )
    timeout_seconds: int = Field(..., description="Execution timeout")
    enabled: bool = Field(..., description="Whether tool is enabled")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")
    created_by: str | None = Field(default=None, description="Creator user ID")


class ToolListResponse(BaseModel):
    """Response for listing tools."""

    tools: list[ToolDefinitionResponse] = Field(..., description="List of tools")
    total: int = Field(..., description="Total number of tools")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")


# =============================================================================
# Agent Definition Schemas
# =============================================================================


class RetryStrategyConfig(BaseModel):
    """Configuration for agent retry behavior."""

    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    retry_conditions: list[str] = Field(
        default_factory=lambda: ["no_results"],
        description="Conditions that trigger retry",
    )
    query_modification: Literal["synonym", "broader", "narrower", "llm_rewrite", "none"] = Field(
        default="synonym",
        description="Query modification strategy",
    )
    backoff_seconds: float = Field(
        default=0.5, ge=0, le=10, description="Backoff between retries"
    )


class AgentExecutorConfig(BaseModel):
    """Configuration for agent execution."""

    type: Literal["llm", "rule_based", "hybrid"] = Field(
        default="llm", description="Executor type"
    )

    # LLM executor config
    model_id: str | None = Field(
        default=None, description="Override model ID for this agent"
    )
    system_prompt: str | None = Field(
        default=None, description="Custom system prompt"
    )
    temperature: float = Field(
        default=0.0, ge=0, le=1, description="LLM temperature"
    )
    max_tokens: int = Field(
        default=4096, ge=1, le=16384, description="Max output tokens"
    )

    # Rule-based executor config
    rules: list[dict] | None = Field(
        default=None, description="Rule definitions for rule-based execution"
    )

    # Common config
    output_format: Literal["text", "json", "structured"] = Field(
        default="text", description="Expected output format"
    )
    output_schema: dict | None = Field(
        default=None, description="JSON schema for structured output"
    )


class AgentDefinitionCreate(BaseModel):
    """Schema for creating a new agent definition."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Unique agent name",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Agent description",
    )
    capabilities: list[str] = Field(
        ...,
        min_length=1,
        description="Agent capabilities (used for task routing)",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tool names available to this agent",
    )
    executor: AgentExecutorConfig = Field(
        default_factory=AgentExecutorConfig,
        description="Agent executor configuration",
    )
    retry_strategy: RetryStrategyConfig = Field(
        default_factory=RetryStrategyConfig,
        description="Retry configuration",
    )
    priority: int = Field(
        default=0,
        ge=-100,
        le=100,
        description="Agent priority (higher = preferred)",
    )
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    response_format: Literal["text", "json"] | None = Field(
        default=None,
        description="Output response format (text or json)",
    )
    response_schema: dict | None = Field(
        default=None,
        description="JSON schema for response validation (required when response_format is json)",
    )
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        reserved = ["main", "supervisor", "system", "admin"]
        if v in reserved:
            raise ValueError(f"Agent name '{v}' is reserved")
        return v

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one capability is required")
        return list(set(v))  # Remove duplicates

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "document_search",
                    "description": "Search and analyze documents",
                    "capabilities": ["document_search", "text_analysis"],
                    "tools": ["vector_db_search", "document_fetch"],
                    "executor": {
                        "type": "llm",
                        "temperature": 0.0,
                        "system_prompt": "You are a document search specialist.",
                    },
                    "retry_strategy": {
                        "max_attempts": 3,
                        "retry_conditions": ["no_results"],
                        "query_modification": "synonym",
                    },
                    "priority": 10,
                    "enabled": True,
                }
            ]
        }
    }


class AgentDefinitionUpdate(BaseModel):
    """Schema for updating an agent definition."""

    description: str | None = Field(default=None, description="Agent description")
    capabilities: list[str] | None = Field(default=None, description="Agent capabilities")
    tools: list[str] | None = Field(default=None, description="Tool names")
    executor: AgentExecutorConfig | None = Field(
        default=None, description="Executor configuration"
    )
    retry_strategy: RetryStrategyConfig | None = Field(
        default=None, description="Retry configuration"
    )
    priority: int | None = Field(default=None, ge=-100, le=100, description="Agent priority")
    enabled: bool | None = Field(default=None, description="Whether agent is enabled")
    response_format: Literal["text", "json"] | None = Field(
        default=None,
        description="Output response format (text or json)",
    )
    response_schema: dict | None = Field(
        default=None,
        description="JSON schema for response validation",
    )
    metadata: dict | None = Field(default=None, description="Additional metadata")


class AgentDefinitionResponse(BaseModel):
    """Response schema for an agent definition."""

    id: str = Field(..., description="Agent definition ID")
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: list[str] = Field(..., description="Agent capabilities")
    tools: list[str] = Field(default_factory=list, description="Tool names")
    executor: AgentExecutorConfig = Field(..., description="Executor configuration")
    retry_strategy: RetryStrategyConfig = Field(..., description="Retry configuration")
    priority: int = Field(..., description="Agent priority")
    enabled: bool = Field(..., description="Whether agent is enabled")
    response_format: str | None = Field(default=None, description="Output response format")
    response_schema: dict | None = Field(default=None, description="JSON schema for response")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")
    created_by: str | None = Field(default=None, description="Creator user ID")


class AgentListResponse(BaseModel):
    """Response for listing agents."""

    agents: list[AgentDefinitionResponse] = Field(..., description="List of agents")
    total: int = Field(..., description="Total number of agents")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")


# =============================================================================
# Common Schemas
# =============================================================================


class ValidationResult(BaseModel):
    """Result of validating a definition."""

    valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class TestExecutionRequest(BaseModel):
    """Request to test a tool or agent execution."""

    parameters: dict = Field(
        default_factory=dict, description="Test parameters"
    )
    mock_context: dict = Field(
        default_factory=dict, description="Mock context data"
    )
    dry_run: bool = Field(
        default=True, description="If true, don't execute actual calls"
    )


class TestExecutionResponse(BaseModel):
    """Response from a test execution."""

    success: bool = Field(..., description="Whether execution succeeded")
    result: Any = Field(default=None, description="Execution result")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(default=0, description="Execution duration")
    logs: list[str] = Field(default_factory=list, description="Execution logs")


class BulkOperationRequest(BaseModel):
    """Request for bulk operations."""

    operation: Literal["enable", "disable", "delete"] = Field(
        ..., description="Operation to perform"
    )
    ids: list[str] = Field(
        ..., min_length=1, max_length=100, description="IDs to operate on"
    )


class BulkOperationResponse(BaseModel):
    """Response from a bulk operation."""

    success_count: int = Field(..., description="Number of successful operations")
    failure_count: int = Field(..., description="Number of failed operations")
    failures: list[dict] = Field(
        default_factory=list, description="Details of failures"
    )


class ImportExportRequest(BaseModel):
    """Request for import/export operations."""

    format: Literal["json", "yaml"] = Field(
        default="json", description="Export format"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in export"
    )


class ImportRequest(BaseModel):
    """Request for importing definitions."""

    data: dict = Field(..., description="Import data")
    overwrite: bool = Field(
        default=False, description="Overwrite existing definitions"
    )
    validate_only: bool = Field(
        default=False, description="Only validate, don't import"
    )


class ImportResponse(BaseModel):
    """Response from an import operation."""

    success: bool = Field(..., description="Whether import succeeded")
    imported_tools: int = Field(default=0, description="Number of tools imported")
    imported_agents: int = Field(default=0, description="Number of agents imported")
    errors: list[str] = Field(default_factory=list, description="Import errors")
    warnings: list[str] = Field(default_factory=list, description="Import warnings")


# =============================================================================
# OpenAPI Schemas
# =============================================================================


class OpenAPIAuthConfig(BaseModel):
    """Authentication configuration for OpenAPI tools."""

    type: Literal["none", "bearer", "api_key", "basic", "oauth2", "service_token"] = Field(
        default="none", description="Authentication type"
    )
    token_env: str | None = Field(
        default=None, description="Environment variable for token"
    )
    header_name: str | None = Field(
        default=None, description="Header name for API key"
    )
    service_name: str | None = Field(
        default=None, description="Service token name from context"
    )


class OpenAPIImportOptions(BaseModel):
    """Options for importing OpenAPI specification."""

    base_url: str | None = Field(
        default=None, description="Override base URL from spec"
    )
    prefix: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Prefix for generated tool names",
    )
    include_operations: list[str] | None = Field(
        default=None, description="Only include these operationIds"
    )
    exclude_operations: list[str] | None = Field(
        default=None, description="Exclude these operationIds"
    )
    include_tags: list[str] | None = Field(
        default=None, description="Only include operations with these tags"
    )
    exclude_tags: list[str] | None = Field(
        default=None, description="Exclude operations with these tags"
    )
    auth_config: OpenAPIAuthConfig | None = Field(
        default=None, description="Authentication configuration"
    )
    enabled: bool = Field(
        default=True, description="Whether generated tools should be enabled"
    )


class OpenAPIImportRequest(BaseModel):
    """Request for importing tools from OpenAPI specification."""

    spec: dict = Field(
        ..., description="OpenAPI 3.x specification as JSON object"
    )
    options: OpenAPIImportOptions = Field(
        default_factory=OpenAPIImportOptions,
        description="Import options",
    )
    validate_only: bool = Field(
        default=False, description="Only validate, don't register tools"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "spec": {
                        "openapi": "3.0.0",
                        "info": {
                            "title": "Weather API",
                            "version": "1.0.0"
                        },
                        "servers": [
                            {"url": "https://api.weather.com/v1"}
                        ],
                        "paths": {
                            "/weather": {
                                "get": {
                                    "operationId": "getCurrentWeather",
                                    "summary": "Get current weather for a location",
                                    "parameters": [
                                        {
                                            "name": "location",
                                            "in": "query",
                                            "required": True,
                                            "schema": {"type": "string"}
                                        }
                                    ],
                                    "responses": {
                                        "200": {
                                            "description": "Weather data"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "options": {
                        "prefix": "weather",
                        "enabled": True
                    }
                }
            ]
        }
    }


class OpenAPIImportFromURLRequest(BaseModel):
    """Request for importing tools from OpenAPI specification URL."""

    url: str = Field(
        ...,
        description="URL to OpenAPI specification (JSON or YAML)",
    )
    options: OpenAPIImportOptions = Field(
        default_factory=OpenAPIImportOptions,
        description="Import options",
    )
    validate_only: bool = Field(
        default=False, description="Only validate, don't register tools"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://api.example.com/openapi.json",
                    "options": {
                        "prefix": "example",
                        "include_tags": ["public"],
                        "auth_config": {
                            "type": "api_key",
                            "header_name": "X-API-Key",
                            "token_env": "EXAMPLE_API_KEY"
                        }
                    }
                }
            ]
        }
    }


class OpenAPIToolPreview(BaseModel):
    """Preview of a tool that will be generated from OpenAPI."""

    name: str = Field(..., description="Generated tool name")
    description: str = Field(..., description="Tool description")
    method: str = Field(..., description="HTTP method")
    path: str = Field(..., description="API path")
    parameters: list[dict] = Field(..., description="Tool parameters")
    tags: list[str] = Field(default_factory=list, description="OpenAPI tags")


class OpenAPIImportResponse(BaseModel):
    """Response from importing OpenAPI specification."""

    success: bool = Field(..., description="Whether import succeeded")
    tools_generated: int = Field(..., description="Number of tools generated")
    tools_registered: int = Field(..., description="Number of tools registered")
    tools: list[OpenAPIToolPreview] = Field(
        default_factory=list, description="Preview of generated tools"
    )
    errors: list[str] = Field(default_factory=list, description="Import errors")
    warnings: list[str] = Field(default_factory=list, description="Import warnings")
    source: str | None = Field(default=None, description="Source API title")
