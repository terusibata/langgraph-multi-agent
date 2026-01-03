"""Admin API routes for agent management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.middleware.access_key import RequestContext, check_permission
from src.api.schemas.admin import (
    AgentDefinitionCreate,
    AgentDefinitionUpdate,
    AgentDefinitionResponse,
    AgentListResponse as AdminAgentListResponse,
)
from src.agents.registry import (
    get_agent_registry,
    create_agent_definition,
)

logger = structlog.get_logger()

router = APIRouter(tags=["Admin - Agents"])


@router.get(
    "/agents",
    response_model=AdminAgentListResponse,
    summary="List all agent definitions",
)
async def list_agent_definitions(
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:read"))],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    enabled_only: bool = Query(default=False, description="Only show enabled agents"),
    capability: str | None = Query(default=None, description="Filter by capability"),
):
    """
    List all registered agent definitions (dynamic only).

    Static agents can be viewed via /agent/agents endpoint.
    """
    registry = get_agent_registry()

    # Get all definitions
    if enabled_only:
        definitions = await registry.list_enabled_definitions()
    else:
        definitions = await registry.list_all_definitions()

    # Filter by capability
    if capability:
        definitions = [d for d in definitions if capability in d.capabilities]

    # Pagination
    total = len(definitions)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = definitions[start:end]

    # Convert to response
    agents = [
        AgentDefinitionResponse(**d.to_dict())
        for d in paginated
    ]

    return AdminAgentListResponse(
        agents=agents,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/agents",
    response_model=AgentDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new agent definition",
)
async def create_agent(
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    body: AgentDefinitionCreate,
):
    """Create a new dynamic agent definition."""
    registry = get_agent_registry()

    # Check if name already exists
    if registry.get(body.name) or await registry.get_definition(body.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{body.name}' already exists",
        )

    # Create definition
    definition = create_agent_definition(
        name=body.name,
        description=body.description,
        capabilities=body.capabilities,
        tools=body.tools,
        executor=body.executor.model_dump() if body.executor else {"type": "llm"},
        retry_strategy=body.retry_strategy.model_dump() if body.retry_strategy else None,
        priority=body.priority,
        enabled=body.enabled,
        metadata=body.metadata,
    )

    # Register
    try:
        await registry.register_definition(definition, user_id=context.user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    logger.info(
        "agent_created",
        agent_name=body.name,
        agent_id=definition.id,
        created_by=context.user_id,
    )

    return AgentDefinitionResponse(**definition.to_dict())


@router.get(
    "/agents/{agent_name}",
    response_model=AgentDefinitionResponse,
    summary="Get an agent definition",
)
async def get_agent(
    agent_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:read"))],
):
    """Get a specific agent definition by name."""
    registry = get_agent_registry()

    definition = await registry.get_definition(agent_name)
    if not definition:
        # Check if it's a static agent
        static_agent = registry.get(agent_name)
        if static_agent:
            info = await registry.get_agent_info(agent_name)
            return AgentDefinitionResponse(**info)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    return AgentDefinitionResponse(**definition.to_dict())


@router.put(
    "/agents/{agent_name}",
    response_model=AgentDefinitionResponse,
    summary="Update an agent definition",
)
async def update_agent(
    agent_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    body: AgentDefinitionUpdate,
):
    """Update an existing agent definition."""
    registry = get_agent_registry()

    # Check if it's a static agent (can't modify)
    if registry.get(agent_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify static agents",
        )

    # Build updates
    updates = body.model_dump(exclude_none=True)
    if "executor" in updates and hasattr(updates["executor"], "model_dump"):
        updates["executor"] = updates["executor"].model_dump()
    if "retry_strategy" in updates and hasattr(updates["retry_strategy"], "model_dump"):
        updates["retry_strategy"] = updates["retry_strategy"].model_dump()

    # Update
    definition = await registry.update_definition(agent_name, updates)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    logger.info(
        "agent_updated",
        agent_name=agent_name,
        agent_id=definition.id,
        updated_by=context.user_id,
    )

    return AgentDefinitionResponse(**definition.to_dict())


@router.delete(
    "/agents/{agent_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an agent definition",
)
async def delete_agent(
    agent_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
):
    """Delete an agent definition."""
    registry = get_agent_registry()

    # Check if it's a static agent (can't delete)
    if registry.get(agent_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete static agents",
        )

    # Delete
    deleted = await registry.delete_definition(agent_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    logger.info(
        "agent_deleted",
        agent_name=agent_name,
        deleted_by=context.user_id,
    )


@router.patch(
    "/agents/{agent_name}/toggle",
    response_model=AgentDefinitionResponse,
    summary="Toggle agent enabled status",
)
async def toggle_agent(
    agent_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    enabled: bool = Query(..., description="Whether to enable or disable"),
):
    """Enable or disable an agent."""
    registry = get_agent_registry()

    # Check if it's a static agent (can't modify)
    if registry.get(agent_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify static agents via this endpoint",
        )

    # Update
    definition = await registry.update_definition(agent_name, {"enabled": enabled})
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    logger.info(
        "agent_toggled",
        agent_name=agent_name,
        enabled=enabled,
        toggled_by=context.user_id,
    )

    return AgentDefinitionResponse(**definition.to_dict())
