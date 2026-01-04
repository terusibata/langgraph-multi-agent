"""Tools module.

All Tools are now dynamically created via:
- Dynamic Tool Definitions (stored in database)
- OpenAPI-generated Tools (auto-generated from OpenAPI specs)

For examples of static tool implementations, see the examples/tools/ directory.
"""

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters
from src.agents.tools.dynamic import (
    DynamicTool,
    DynamicToolFactory,
    get_dynamic_tool,
    create_dynamic_tools_for_agent,
)
from src.agents.tools.openapi import (
    OpenAPIParser,
    OpenAPIParseError,
    OpenAPIToolGenerator,
    parse_openapi_spec,
    generate_tools_from_openapi,
)

__all__ = [
    "ToolBase",
    "ToolResult",
    "ToolContext",
    "ToolParameters",
    "DynamicTool",
    "DynamicToolFactory",
    "get_dynamic_tool",
    "create_dynamic_tools_for_agent",
    "OpenAPIParser",
    "OpenAPIParseError",
    "OpenAPIToolGenerator",
    "parse_openapi_spec",
    "generate_tools_from_openapi",
]
