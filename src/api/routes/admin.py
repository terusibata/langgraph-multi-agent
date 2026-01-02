"""Admin API routes for dynamic agent and tool management.

This module serves as the main router that aggregates all admin sub-routers:
- admin_tools: Tool CRUD operations
- admin_agents: Agent CRUD operations
- admin_openapi: OpenAPI import functionality

For bulk operations and advanced features, additional endpoints are included here.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.middleware.access_key import RequestContext, check_permission
from src.api.schemas.admin import (
    BulkOperationRequest,
    BulkOperationResponse,
    ToolDefinitionResponse,
    AgentDefinitionResponse,
)
from src.agents.registry import (
    get_tool_registry,
    get_agent_registry,
)

# Import sub-routers
from src.api.routes.admin_tools import router as tools_router
from src.api.routes.admin_agents import router as agents_router
from src.api.routes.admin_openapi import router as openapi_router

logger = structlog.get_logger()

# Main admin router
router = APIRouter(prefix="/admin", tags=["Admin"])

# Include sub-routers
router.include_router(tools_router)
router.include_router(agents_router)
router.include_router(openapi_router)


# =============================================================================
# Bulk Operations
# =============================================================================


@router.post(
    "/bulk/tools/enable",
    response_model=BulkOperationResponse,
    summary="Bulk enable/disable tools",
)
async def bulk_toggle_tools(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: BulkOperationRequest,
):
    """Enable or disable multiple tools at once."""
    registry = get_tool_registry()

    succeeded = []
    failed = []

    for name in body.names:
        try:
            definition = await registry.update_definition(
                name,
                {"enabled": body.enabled}
            )
            if definition:
                succeeded.append(name)
            else:
                # Check if it's a static tool
                if registry.get(name):
                    failed.append({
                        "name": name,
                        "error": "Cannot modify static tools",
                    })
                else:
                    failed.append({
                        "name": name,
                        "error": "Tool not found",
                    })
        except Exception as e:
            failed.append({
                "name": name,
                "error": str(e),
            })

    logger.info(
        "bulk_toggle_tools",
        enabled=body.enabled,
        succeeded_count=len(succeeded),
        failed_count=len(failed),
        user_id=context.user_id,
    )

    return BulkOperationResponse(
        success=len(failed) == 0,
        succeeded=succeeded,
        failed=failed,
        message=f"Toggled {len(succeeded)} tools, {len(failed)} failed",
    )


@router.post(
    "/bulk/agents/enable",
    response_model=BulkOperationResponse,
    summary="Bulk enable/disable agents",
)
async def bulk_toggle_agents(
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    body: BulkOperationRequest,
):
    """Enable or disable multiple agents at once."""
    registry = get_agent_registry()

    succeeded = []
    failed = []

    for name in body.names:
        try:
            definition = await registry.update_definition(
                name,
                {"enabled": body.enabled}
            )
            if definition:
                succeeded.append(name)
            else:
                # Check if it's a static agent
                if registry.get(name):
                    failed.append({
                        "name": name,
                        "error": "Cannot modify static agents",
                    })
                else:
                    failed.append({
                        "name": name,
                        "error": "Agent not found",
                    })
        except Exception as e:
            failed.append({
                "name": name,
                "error": str(e),
            })

    logger.info(
        "bulk_toggle_agents",
        enabled=body.enabled,
        succeeded_count=len(succeeded),
        failed_count=len(failed),
        user_id=context.user_id,
    )

    return BulkOperationResponse(
        success=len(failed) == 0,
        succeeded=succeeded,
        failed=failed,
        message=f"Toggled {len(succeeded)} agents, {len(failed)} failed",
    )


@router.delete(
    "/bulk/tools",
    response_model=BulkOperationResponse,
    summary="Bulk delete tools",
)
async def bulk_delete_tools(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    names: list[str] = Query(..., description="Tool names to delete"),
):
    """Delete multiple tools at once."""
    registry = get_tool_registry()

    succeeded = []
    failed = []

    for name in names:
        try:
            # Check if it's a static tool
            if registry.get(name):
                failed.append({
                    "name": name,
                    "error": "Cannot delete static tools",
                })
                continue

            deleted = await registry.delete_definition(name)
            if deleted:
                succeeded.append(name)
            else:
                failed.append({
                    "name": name,
                    "error": "Tool not found",
                })
        except Exception as e:
            failed.append({
                "name": name,
                "error": str(e),
            })

    logger.info(
        "bulk_delete_tools",
        succeeded_count=len(succeeded),
        failed_count=len(failed),
        user_id=context.user_id,
    )

    return BulkOperationResponse(
        success=len(failed) == 0,
        succeeded=succeeded,
        failed=failed,
        message=f"Deleted {len(succeeded)} tools, {len(failed)} failed",
    )


@router.delete(
    "/bulk/agents",
    response_model=BulkOperationResponse,
    summary="Bulk delete agents",
)
async def bulk_delete_agents(
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    names: list[str] = Query(..., description="Agent names to delete"),
):
    """Delete multiple agents at once."""
    registry = get_agent_registry()

    succeeded = []
    failed = []

    for name in names:
        try:
            # Check if it's a static agent
            if registry.get(name):
                failed.append({
                    "name": name,
                    "error": "Cannot delete static agents",
                })
                continue

            deleted = await registry.delete_definition(name)
            if deleted:
                succeeded.append(name)
            else:
                failed.append({
                    "name": name,
                    "error": "Agent not found",
                })
        except Exception as e:
            failed.append({
                "name": name,
                "error": str(e),
            })

    logger.info(
        "bulk_delete_agents",
        succeeded_count=len(succeeded),
        failed_count=len(failed),
        user_id=context.user_id,
    )

    return BulkOperationResponse(
        success=len(failed) == 0,
        succeeded=succeeded,
        failed=failed,
        message=f"Deleted {len(succeeded)} agents, {len(failed)} failed",
    )


# =============================================================================
# Statistics and Overview
# =============================================================================


@router.get(
    "/stats",
    summary="Get admin statistics",
)
async def get_admin_stats(
    context: Annotated[RequestContext, Depends(check_permission("admin:read"))],
):
    """Get overview statistics for the admin dashboard."""
    tool_registry = get_tool_registry()
    agent_registry = get_agent_registry()

    # Tool stats
    static_tools = len(tool_registry.list_all())
    dynamic_tools = len(tool_registry.list_all_definitions())
    enabled_dynamic_tools = len(tool_registry.list_enabled_definitions())

    # Agent stats
    static_agents = len(agent_registry.list_all())
    dynamic_agents = len(agent_registry.list_all_definitions())
    enabled_dynamic_agents = len(agent_registry.list_enabled_definitions())

    # Category breakdown for tools
    tool_categories = {}
    for tool in tool_registry.list_all_definitions():
        category = tool.category or "general"
        tool_categories[category] = tool_categories.get(category, 0) + 1

    # Capability breakdown for agents
    agent_capabilities = {}
    for agent in agent_registry.list_all_definitions():
        for cap in agent.capabilities:
            agent_capabilities[cap] = agent_capabilities.get(cap, 0) + 1

    return {
        "tools": {
            "static": static_tools,
            "dynamic": dynamic_tools,
            "dynamic_enabled": enabled_dynamic_tools,
            "dynamic_disabled": dynamic_tools - enabled_dynamic_tools,
            "total": static_tools + dynamic_tools,
            "categories": tool_categories,
        },
        "agents": {
            "static": static_agents,
            "dynamic": dynamic_agents,
            "dynamic_enabled": enabled_dynamic_agents,
            "dynamic_disabled": dynamic_agents - enabled_dynamic_agents,
            "total": static_agents + dynamic_agents,
            "capabilities": agent_capabilities,
        },
    }


@router.get(
    "/export",
    summary="Export all dynamic definitions",
)
async def export_definitions(
    context: Annotated[RequestContext, Depends(check_permission("admin:read"))],
    include_tools: bool = Query(default=True, description="Include tool definitions"),
    include_agents: bool = Query(default=True, description="Include agent definitions"),
):
    """Export all dynamic tool and agent definitions for backup or migration."""
    result = {}

    if include_tools:
        tool_registry = get_tool_registry()
        result["tools"] = [
            t.to_dict() for t in tool_registry.list_all_definitions()
        ]

    if include_agents:
        agent_registry = get_agent_registry()
        result["agents"] = [
            a.to_dict() for a in agent_registry.list_all_definitions()
        ]

    logger.info(
        "definitions_exported",
        tools_count=len(result.get("tools", [])),
        agents_count=len(result.get("agents", [])),
        user_id=context.user_id,
    )

    return result
