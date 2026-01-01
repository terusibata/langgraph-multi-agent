"""OpenAPI specification parser and tool generator."""

import re
from typing import Any
from urllib.parse import urljoin

import structlog
import yaml

from src.agents.registry import (
    ToolDefinition,
    get_tool_registry,
    create_tool_definition,
)

logger = structlog.get_logger()


class OpenAPIParseError(Exception):
    """Error parsing OpenAPI specification."""
    pass


class OpenAPIParser:
    """
    Parser for OpenAPI 3.x specifications.

    Converts OpenAPI operations to dynamic tool definitions.
    """

    SUPPORTED_VERSIONS = ["3.0", "3.1"]

    def __init__(self, spec: dict):
        """
        Initialize the parser.

        Args:
            spec: Parsed OpenAPI specification dictionary
        """
        self.spec = spec
        self.validate_version()

        # Extract common info
        self.info = spec.get("info", {})
        self.servers = spec.get("servers", [])
        self.paths = spec.get("paths", {})
        self.components = spec.get("components", {})
        self.security = spec.get("security", [])

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "OpenAPIParser":
        """
        Create parser from YAML string.

        Args:
            yaml_content: YAML string of OpenAPI spec

        Returns:
            OpenAPIParser instance
        """
        try:
            spec = yaml.safe_load(yaml_content)
            return cls(spec)
        except yaml.YAMLError as e:
            raise OpenAPIParseError(f"Invalid YAML: {str(e)}")

    @classmethod
    def from_json(cls, json_content: dict) -> "OpenAPIParser":
        """
        Create parser from JSON dict.

        Args:
            json_content: Dictionary of OpenAPI spec

        Returns:
            OpenAPIParser instance
        """
        return cls(json_content)

    def validate_version(self) -> None:
        """Validate OpenAPI version."""
        version = self.spec.get("openapi", "")
        if not any(version.startswith(v) for v in self.SUPPORTED_VERSIONS):
            raise OpenAPIParseError(
                f"Unsupported OpenAPI version: {version}. "
                f"Supported: {self.SUPPORTED_VERSIONS}"
            )

    def get_base_url(self, server_index: int = 0) -> str:
        """
        Get base URL from servers.

        Args:
            server_index: Index of server to use

        Returns:
            Base URL string
        """
        if not self.servers:
            return ""

        server = self.servers[min(server_index, len(self.servers) - 1)]
        url = server.get("url", "")

        # Handle server variables
        variables = server.get("variables", {})
        for var_name, var_config in variables.items():
            default_value = var_config.get("default", "")
            url = url.replace(f"{{{var_name}}}", default_value)

        return url

    def parse_all_operations(self) -> list[dict]:
        """
        Parse all operations from the spec.

        Returns:
            List of operation dictionaries
        """
        operations = []

        for path, path_item in self.paths.items():
            # Handle $ref in path item
            if "$ref" in path_item:
                path_item = self._resolve_ref(path_item["$ref"])

            # Common parameters for all operations in this path
            common_params = path_item.get("parameters", [])

            for method in ["get", "post", "put", "delete", "patch", "options", "head"]:
                if method in path_item:
                    operation = path_item[method]
                    operations.append({
                        "path": path,
                        "method": method.upper(),
                        "operation": operation,
                        "common_parameters": common_params,
                    })

        return operations

    def operation_to_tool_definition(
        self,
        path: str,
        method: str,
        operation: dict,
        common_parameters: list[dict] | None = None,
        base_url: str | None = None,
        auth_config: dict | None = None,
    ) -> ToolDefinition:
        """
        Convert an OpenAPI operation to a ToolDefinition.

        Args:
            path: API path
            method: HTTP method
            operation: Operation object
            common_parameters: Parameters from path item
            base_url: Override base URL
            auth_config: Authentication configuration

        Returns:
            ToolDefinition instance
        """
        # Generate tool name from operationId or path
        operation_id = operation.get("operationId")
        if operation_id:
            name = self._sanitize_name(operation_id)
        else:
            name = self._generate_name_from_path(path, method)

        # Get description
        description = operation.get("summary") or operation.get("description") or f"{method} {path}"

        # Parse parameters
        all_params = (common_parameters or []) + operation.get("parameters", [])
        parameters = self._parse_parameters(all_params)

        # Parse request body if present
        request_body = operation.get("requestBody")
        if request_body:
            body_params = self._parse_request_body(request_body)
            parameters.extend(body_params)

        # Build URL
        url = base_url or self.get_base_url()
        full_url = urljoin(url + "/", path.lstrip("/"))

        # Determine auth type from security
        auth_type, auth_cfg = self._determine_auth(operation, auth_config)

        # Build executor config
        executor = {
            "type": "http",
            "url": full_url,
            "method": method,
            "headers": {},
            "auth_type": auth_type,
            "auth_config": auth_cfg,
        }

        # Add path parameter handling
        path_params = [p["name"] for p in parameters if p.get("in") == "path"]
        if path_params:
            executor["path_parameters"] = path_params

        # Create tool definition
        return create_tool_definition(
            name=name,
            description=description,
            category="openapi",
            parameters=parameters,
            executor=executor,
            metadata={
                "openapi_source": self.info.get("title", "Unknown API"),
                "openapi_version": self.info.get("version", ""),
                "original_path": path,
                "original_method": method,
                "tags": operation.get("tags", []),
            },
        )

    def _parse_parameters(self, params: list[dict]) -> list[dict]:
        """Parse OpenAPI parameters to tool parameter format."""
        result = []

        for param in params:
            # Handle $ref
            if "$ref" in param:
                param = self._resolve_ref(param["$ref"])

            schema = param.get("schema", {})
            if "$ref" in schema:
                schema = self._resolve_ref(schema["$ref"])

            tool_param = {
                "name": param.get("name"),
                "type": self._map_type(schema.get("type", "string")),
                "description": param.get("description", ""),
                "required": param.get("required", False),
                "in": param.get("in", "query"),  # query, path, header, cookie
            }

            # Add default if present
            if "default" in schema:
                tool_param["default"] = schema["default"]

            # Add enum if present
            if "enum" in schema:
                tool_param["enum"] = schema["enum"]

            result.append(tool_param)

        return result

    def _parse_request_body(self, request_body: dict) -> list[dict]:
        """Parse request body to parameters."""
        # Handle $ref
        if "$ref" in request_body:
            request_body = self._resolve_ref(request_body["$ref"])

        content = request_body.get("content", {})
        required = request_body.get("required", False)

        # Prefer application/json
        media_type = content.get("application/json", {})
        if not media_type:
            # Try other content types
            for ct in content.values():
                media_type = ct
                break

        schema = media_type.get("schema", {})
        if "$ref" in schema:
            schema = self._resolve_ref(schema["$ref"])

        # If it's an object, flatten properties to parameters
        if schema.get("type") == "object":
            return self._schema_to_parameters(schema, required)

        # Otherwise, create a single body parameter
        return [{
            "name": "body",
            "type": self._map_type(schema.get("type", "object")),
            "description": request_body.get("description", "Request body"),
            "required": required,
            "in": "body",
        }]

    def _schema_to_parameters(
        self,
        schema: dict,
        parent_required: bool = False,
    ) -> list[dict]:
        """Convert JSON Schema to flat parameters."""
        params = []
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            if "$ref" in prop_schema:
                prop_schema = self._resolve_ref(prop_schema["$ref"])

            param = {
                "name": prop_name,
                "type": self._map_type(prop_schema.get("type", "string")),
                "description": prop_schema.get("description", ""),
                "required": parent_required and prop_name in required_fields,
                "in": "body",
            }

            if "default" in prop_schema:
                param["default"] = prop_schema["default"]

            if "enum" in prop_schema:
                param["enum"] = prop_schema["enum"]

            params.append(param)

        return params

    def _resolve_ref(self, ref: str) -> dict:
        """Resolve a $ref pointer."""
        if not ref.startswith("#/"):
            logger.warning("external_ref_not_supported", ref=ref)
            return {}

        parts = ref[2:].split("/")
        current = self.spec

        try:
            for part in parts:
                # Handle URL encoding
                part = part.replace("~1", "/").replace("~0", "~")
                current = current[part]
            return current
        except (KeyError, TypeError):
            logger.warning("ref_resolution_failed", ref=ref)
            return {}

    def _map_type(self, openapi_type: str) -> str:
        """Map OpenAPI type to tool parameter type."""
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        return type_map.get(openapi_type, "string")

    def _sanitize_name(self, name: str) -> str:
        """Sanitize operation ID to valid tool name."""
        # Convert camelCase to snake_case
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
        name = name.lower()

        # Replace invalid characters
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")

        # Ensure starts with letter
        if name and not name[0].isalpha():
            name = "op_" + name

        return name or "unnamed_operation"

    def _generate_name_from_path(self, path: str, method: str) -> str:
        """Generate tool name from path and method."""
        # Remove path parameters
        clean_path = re.sub(r"\{[^}]+\}", "", path)
        # Convert to snake_case
        name = clean_path.replace("/", "_").replace("-", "_")
        name = re.sub(r"_+", "_", name).strip("_")

        return f"{method.lower()}_{name}" if name else method.lower()

    def _determine_auth(
        self,
        operation: dict,
        override_config: dict | None = None,
    ) -> tuple[str, dict]:
        """
        Determine authentication type and config.

        Returns:
            Tuple of (auth_type, auth_config)
        """
        if override_config:
            return override_config.get("type", "none"), override_config

        # Check operation-level security first
        security = operation.get("security", self.security)
        if not security:
            return "none", {}

        # Get first security requirement
        for sec_req in security:
            for scheme_name in sec_req:
                scheme = self.components.get("securitySchemes", {}).get(scheme_name, {})

                if scheme.get("type") == "http":
                    if scheme.get("scheme") == "bearer":
                        return "bearer", {"scheme_name": scheme_name}
                    elif scheme.get("scheme") == "basic":
                        return "basic", {"scheme_name": scheme_name}

                elif scheme.get("type") == "apiKey":
                    return "api_key", {
                        "scheme_name": scheme_name,
                        "in": scheme.get("in", "header"),
                        "name": scheme.get("name", "X-API-Key"),
                    }

                elif scheme.get("type") == "oauth2":
                    return "oauth2", {
                        "scheme_name": scheme_name,
                        "flows": scheme.get("flows", {}),
                    }

        return "none", {}


class OpenAPIToolGenerator:
    """
    Generator for creating and registering tools from OpenAPI specs.
    """

    def __init__(self):
        self.registry = get_tool_registry()

    async def generate_tools_from_spec(
        self,
        spec: dict | str,
        options: dict | None = None,
    ) -> list[ToolDefinition]:
        """
        Generate and register tools from an OpenAPI spec.

        Args:
            spec: OpenAPI spec as dict or YAML string
            options: Generation options
                - base_url: Override base URL
                - prefix: Prefix for tool names
                - include_operations: List of operationIds to include
                - exclude_operations: List of operationIds to exclude
                - include_tags: List of tags to include
                - exclude_tags: List of tags to exclude
                - auth_config: Authentication configuration
                - enabled: Whether tools should be enabled
                - register: Whether to register tools (default: True)

        Returns:
            List of generated ToolDefinitions
        """
        options = options or {}

        # Parse spec
        if isinstance(spec, str):
            parser = OpenAPIParser.from_yaml(spec)
        else:
            parser = OpenAPIParser.from_json(spec)

        # Get all operations
        operations = parser.parse_all_operations()

        # Filter operations
        filtered = self._filter_operations(operations, options)

        # Generate tools
        tools = []
        base_url = options.get("base_url")
        prefix = options.get("prefix", "")
        auth_config = options.get("auth_config")
        enabled = options.get("enabled", True)
        should_register = options.get("register", True)

        for op_data in filtered:
            try:
                definition = parser.operation_to_tool_definition(
                    path=op_data["path"],
                    method=op_data["method"],
                    operation=op_data["operation"],
                    common_parameters=op_data.get("common_parameters"),
                    base_url=base_url,
                    auth_config=auth_config,
                )

                # Apply prefix
                if prefix:
                    definition.name = f"{prefix}_{definition.name}"

                # Set enabled state
                definition.enabled = enabled

                # Register if requested
                if should_register:
                    try:
                        await self.registry.register_definition(definition)
                        logger.info(
                            "openapi_tool_registered",
                            tool_name=definition.name,
                            path=op_data["path"],
                            method=op_data["method"],
                        )
                    except ValueError as e:
                        logger.warning(
                            "openapi_tool_registration_skipped",
                            tool_name=definition.name,
                            reason=str(e),
                        )

                tools.append(definition)

            except Exception as e:
                logger.error(
                    "openapi_tool_generation_failed",
                    path=op_data["path"],
                    method=op_data["method"],
                    error=str(e),
                )

        return tools

    def _filter_operations(
        self,
        operations: list[dict],
        options: dict,
    ) -> list[dict]:
        """Filter operations based on options."""
        include_ops = set(options.get("include_operations", []))
        exclude_ops = set(options.get("exclude_operations", []))
        include_tags = set(options.get("include_tags", []))
        exclude_tags = set(options.get("exclude_tags", []))

        result = []
        for op_data in operations:
            operation = op_data["operation"]
            op_id = operation.get("operationId", "")
            op_tags = set(operation.get("tags", []))

            # Check operation ID filters
            if include_ops and op_id not in include_ops:
                continue
            if op_id in exclude_ops:
                continue

            # Check tag filters
            if include_tags and not op_tags.intersection(include_tags):
                continue
            if op_tags.intersection(exclude_tags):
                continue

            result.append(op_data)

        return result

    async def generate_from_url(
        self,
        url: str,
        options: dict | None = None,
    ) -> list[ToolDefinition]:
        """
        Fetch and generate tools from an OpenAPI spec URL.

        Args:
            url: URL to OpenAPI spec (JSON or YAML)
            options: Generation options

        Returns:
            List of generated ToolDefinitions
        """
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.text
            content_type = response.headers.get("content-type", "")

            # Parse based on content type
            if "json" in content_type:
                spec = response.json()
            else:
                spec = content  # Will be parsed as YAML

            return await self.generate_tools_from_spec(spec, options)


def parse_openapi_spec(spec: dict | str) -> OpenAPIParser:
    """
    Parse an OpenAPI specification.

    Args:
        spec: OpenAPI spec as dict or YAML/JSON string

    Returns:
        OpenAPIParser instance
    """
    if isinstance(spec, str):
        return OpenAPIParser.from_yaml(spec)
    return OpenAPIParser.from_json(spec)


async def generate_tools_from_openapi(
    spec: dict | str,
    options: dict | None = None,
) -> list[ToolDefinition]:
    """
    Generate tools from an OpenAPI specification.

    Args:
        spec: OpenAPI spec
        options: Generation options

    Returns:
        List of generated ToolDefinitions
    """
    generator = OpenAPIToolGenerator()
    return await generator.generate_tools_from_spec(spec, options)
