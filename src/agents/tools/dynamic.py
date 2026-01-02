"""Dynamic tool execution based on runtime definitions."""

import asyncio
import importlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from jinja2 import Template
from jsonpath_ng import parse as jsonpath_parse

from src.agents.registry import ToolDefinition, get_tool_registry
from src.agents.tools.base import ToolBase, ToolContext, ToolResult, ToolParameters

logger = structlog.get_logger()


class DynamicToolParameters(ToolParameters):
    """Dynamic parameters that accept any fields."""

    class Config:
        extra = "allow"


class DynamicTool(ToolBase):
    """
    A tool that executes based on a dynamic definition.

    Supports multiple executor types:
    - http: Make HTTP requests to external APIs
    - python: Call Python functions
    - mock: Return mock responses (for testing)
    """

    def __init__(self, definition: ToolDefinition):
        """
        Initialize from a tool definition.

        Args:
            definition: The tool definition
        """
        super().__init__(
            name=definition.name,
            description=definition.description,
            required_service_token=definition.required_service_token,
            parameters_schema=DynamicToolParameters,
            timeout_seconds=definition.timeout_seconds,
        )
        self.definition = definition
        self.executor_type = definition.executor.get("type", "mock")

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute the tool based on its definition.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution status and data
        """
        started_at = datetime.now(timezone.utc)

        try:
            # Validate parameters
            validation_error = self._validate_params(params)
            if validation_error:
                return ToolResult(
                    success=False,
                    error=validation_error,
                    duration_ms=0,
                )

            # Execute based on type
            if self.executor_type == "http":
                result = await self._execute_http(params, context)
            elif self.executor_type == "python":
                result = await self._execute_python(params, context)
            elif self.executor_type == "mock":
                result = await self._execute_mock(params, context)
            else:
                result = ToolResult(
                    success=False,
                    error=f"Unknown executor type: {self.executor_type}",
                )

            # Calculate duration
            duration = datetime.now(timezone.utc) - started_at
            result.duration_ms = int(duration.total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(
                "dynamic_tool_error",
                tool=self.name,
                error=str(e),
                request_id=context.request_id,
            )
            duration = datetime.now(timezone.utc) - started_at
            return ToolResult(
                success=False,
                error=str(e),
                duration_ms=int(duration.total_seconds() * 1000),
            )

    def _validate_params(self, params: dict) -> str | None:
        """
        Validate parameters against the definition.

        Args:
            params: Parameters to validate

        Returns:
            Error message or None if valid
        """
        for param_def in self.definition.parameters:
            name = param_def.get("name")
            required = param_def.get("required", False)
            param_type = param_def.get("type", "string")
            enum_values = param_def.get("enum")

            if required and name not in params:
                return f"Required parameter '{name}' is missing"

            if name in params:
                value = params[name]

                # Type validation
                if param_type == "string" and not isinstance(value, str):
                    return f"Parameter '{name}' must be a string"
                elif param_type == "integer" and not isinstance(value, int):
                    return f"Parameter '{name}' must be an integer"
                elif param_type == "number" and not isinstance(value, (int, float)):
                    return f"Parameter '{name}' must be a number"
                elif param_type == "boolean" and not isinstance(value, bool):
                    return f"Parameter '{name}' must be a boolean"
                elif param_type == "array" and not isinstance(value, list):
                    return f"Parameter '{name}' must be an array"
                elif param_type == "object" and not isinstance(value, dict):
                    return f"Parameter '{name}' must be an object"

                # Enum validation
                if enum_values and value not in enum_values:
                    return f"Parameter '{name}' must be one of: {enum_values}"

        return None

    async def _execute_http(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute HTTP request.

        Args:
            params: Request parameters
            context: Execution context

        Returns:
            ToolResult with response data
        """
        executor = self.definition.executor
        url = executor.get("url")
        method = executor.get("method", "POST")
        headers = dict(executor.get("headers", {}))
        body_template = executor.get("body_template")
        response_path = executor.get("response_path")
        auth_type = executor.get("auth_type", "none")
        auth_config = executor.get("auth_config", {})

        if not url:
            return ToolResult(success=False, error="URL not configured")

        # Apply URL template substitution
        url = self._apply_template(url, params, context)

        # Build headers
        if auth_type == "bearer":
            token = context.get_token(self.required_service_token or "default")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif auth_type == "api_key":
            header_name = auth_config.get("header", "X-API-Key")
            key_env = auth_config.get("key_env")
            if key_env:
                import os
                api_key = os.getenv(key_env, "")
                headers[header_name] = api_key
            elif self.required_service_token:
                token = context.get_token(self.required_service_token)
                if token:
                    headers[header_name] = token
        elif auth_type == "service_token":
            if self.required_service_token:
                token = context.get_token(self.required_service_token)
                if token:
                    headers["Authorization"] = f"Bearer {token}"

        # Build request body
        body = None
        if body_template:
            body = self._apply_template(body_template, params, context)
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
        elif method in ["POST", "PUT", "PATCH"]:
            body = params

        # Make request
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if isinstance(body, dict) else None,
                content=body if isinstance(body, str) else None,
            )

            if response.status_code >= 400:
                return ToolResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:500]}",
                    metadata={"status_code": response.status_code},
                )

            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = response.text

            # Extract data using JSONPath if specified
            if response_path and isinstance(data, dict):
                try:
                    jsonpath_expr = jsonpath_parse(response_path)
                    matches = jsonpath_expr.find(data)
                    if matches:
                        data = [match.value for match in matches]
                        if len(data) == 1:
                            data = data[0]
                except Exception as e:
                    logger.warning(
                        "jsonpath_extraction_failed",
                        path=response_path,
                        error=str(e),
                    )

            return ToolResult(
                success=True,
                data=data,
                metadata={
                    "status_code": response.status_code,
                    "url": url,
                },
            )

    async def _execute_python(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute Python function.

        Args:
            params: Function parameters
            context: Execution context

        Returns:
            ToolResult with function result
        """
        executor = self.definition.executor
        module_path = executor.get("module_path")
        function_name = executor.get("function_name")
        class_name = executor.get("class_name")

        if not module_path or not function_name:
            return ToolResult(
                success=False,
                error="Python executor requires module_path and function_name",
            )

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the function or method
            if class_name:
                cls = getattr(module, class_name)
                instance = cls()
                func = getattr(instance, function_name)
            else:
                func = getattr(module, function_name)

            # Execute
            if asyncio.iscoroutinefunction(func):
                result = await func(params, context)
            else:
                result = func(params, context)

            # Handle ToolResult returns
            if isinstance(result, ToolResult):
                return result

            return ToolResult(
                success=True,
                data=result,
            )

        except ImportError as e:
            return ToolResult(
                success=False,
                error=f"Failed to import module '{module_path}': {str(e)}",
            )
        except AttributeError as e:
            return ToolResult(
                success=False,
                error=f"Function '{function_name}' not found: {str(e)}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Python execution failed: {str(e)}",
            )

    async def _execute_mock(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute mock response.

        Args:
            params: Parameters (unused)
            context: Execution context (unused)

        Returns:
            ToolResult with mock data
        """
        executor = self.definition.executor
        mock_response = executor.get("mock_response")
        mock_delay_ms = executor.get("mock_delay_ms", 0)

        # Simulate delay
        if mock_delay_ms > 0:
            await asyncio.sleep(mock_delay_ms / 1000)

        # Apply template to mock response if it's a string
        if isinstance(mock_response, str):
            mock_response = self._apply_template(mock_response, params, context)
            try:
                mock_response = json.loads(mock_response)
            except json.JSONDecodeError:
                pass

        return ToolResult(
            success=True,
            data=mock_response,
            metadata={"mock": True},
        )

    def _apply_template(
        self,
        template_str: str,
        params: dict,
        context: ToolContext,
    ) -> str:
        """
        Apply Jinja2 template substitution.

        Args:
            template_str: Template string
            params: Parameters to substitute
            context: Execution context

        Returns:
            Rendered string
        """
        try:
            template = Template(template_str)
            return template.render(
                params=params,
                context={
                    "tenant_id": context.tenant_id,
                    "user_id": context.user_id,
                    "request_id": context.request_id,
                },
            )
        except Exception as e:
            logger.warning(
                "template_render_failed",
                template=template_str[:100],
                error=str(e),
            )
            # Fall back to simple substitution
            result = template_str
            for key, value in params.items():
                result = result.replace(f"{{{{ params.{key} }}}}", str(value))
                result = result.replace(f"{{{{params.{key}}}}}", str(value))
            return result


class DynamicToolFactory:
    """Factory for creating DynamicTool instances from definitions."""

    @staticmethod
    def create(definition: ToolDefinition) -> DynamicTool:
        """
        Create a DynamicTool from a definition.

        Args:
            definition: Tool definition

        Returns:
            DynamicTool instance
        """
        return DynamicTool(definition)

    @staticmethod
    def create_from_name(name: str) -> DynamicTool | None:
        """
        Create a DynamicTool from a registered definition name.

        Args:
            name: Tool name

        Returns:
            DynamicTool instance or None if not found
        """
        registry = get_tool_registry()
        definition = registry.get_definition(name)
        if not definition:
            return None
        return DynamicTool(definition)


def get_dynamic_tool(name: str) -> DynamicTool | None:
    """
    Get a dynamic tool by name.

    Args:
        name: Tool name

    Returns:
        DynamicTool instance or None
    """
    return DynamicToolFactory.create_from_name(name)


def create_dynamic_tools_for_agent(tool_names: list[str]) -> list[DynamicTool]:
    """
    Create dynamic tools for an agent.

    Args:
        tool_names: List of tool names

    Returns:
        List of DynamicTool instances
    """
    tools = []
    for name in tool_names:
        tool = get_dynamic_tool(name)
        if tool:
            tools.append(tool)
        else:
            logger.warning("dynamic_tool_not_found", tool_name=name)
    return tools
