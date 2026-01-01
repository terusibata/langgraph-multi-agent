"""Admin API routes for dynamic agent and tool management."""

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.middleware.access_key import verify_access_key, RequestContext, check_permission
from src.api.schemas.admin import (
    # Tool schemas
    ToolDefinitionCreate,
    ToolDefinitionUpdate,
    ToolDefinitionResponse,
    ToolListResponse as AdminToolListResponse,
    # Agent schemas
    AgentDefinitionCreate,
    AgentDefinitionUpdate,
    AgentDefinitionResponse,
    AgentListResponse as AdminAgentListResponse,
    # Common schemas
    ValidationResult,
    TestExecutionRequest,
    TestExecutionResponse,
    BulkOperationRequest,
    BulkOperationResponse,
    ImportRequest,
    ImportResponse,
)
from src.api.schemas.response import ErrorResponse
from src.agents.registry import (
    get_tool_registry,
    get_agent_registry,
    ToolDefinition,
    AgentDefinition,
    create_tool_definition,
    create_agent_definition,
)
from src.agents.tools.dynamic import DynamicToolFactory
from src.agents.sub_agents.dynamic import DynamicAgentFactory
from src.agents.tools.base import ToolContext

logger = structlog.get_logger()

router = APIRouter(prefix="/admin", tags=["Admin"])


# =============================================================================
# Tool Management Endpoints
# =============================================================================


@router.get(
    "/tools",
    response_model=AdminToolListResponse,
    summary="List all tool definitions",
)
async def list_tool_definitions(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:read"))],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    enabled_only: bool = Query(default=False, description="Only show enabled tools"),
    category: str | None = Query(default=None, description="Filter by category"),
):
    """
    List all registered tool definitions (dynamic only).

    Static tools can be viewed via /agent/tools endpoint.
    """
    registry = get_tool_registry()

    # Get all definitions
    if enabled_only:
        definitions = registry.list_enabled_definitions()
    else:
        definitions = registry.list_all_definitions()

    # Filter by category
    if category:
        definitions = [d for d in definitions if d.category == category]

    # Pagination
    total = len(definitions)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = definitions[start:end]

    # Convert to response
    tools = [
        ToolDefinitionResponse(**d.to_dict())
        for d in paginated
    ]

    return AdminToolListResponse(
        tools=tools,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post(
    "/tools",
    response_model=ToolDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new tool definition",
)
async def create_tool(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: ToolDefinitionCreate,
):
    """Create a new dynamic tool definition."""
    registry = get_tool_registry()

    # Check if name already exists
    if registry.get(body.name) or registry.get_definition(body.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool '{body.name}' already exists",
        )

    # Create definition
    definition = create_tool_definition(
        name=body.name,
        description=body.description,
        category=body.category,
        parameters=[p.model_dump() for p in body.parameters],
        executor=body.executor.model_dump(),
        required_service_token=body.required_service_token,
        timeout_seconds=body.timeout_seconds,
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
        "tool_created",
        tool_name=body.name,
        tool_id=definition.id,
        created_by=context.user_id,
    )

    return ToolDefinitionResponse(**definition.to_dict())


@router.get(
    "/tools/{tool_name}",
    response_model=ToolDefinitionResponse,
    summary="Get a tool definition",
)
async def get_tool(
    tool_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:read"))],
):
    """Get a specific tool definition by name."""
    registry = get_tool_registry()

    definition = registry.get_definition(tool_name)
    if not definition:
        # Check if it's a static tool
        static_tool = registry.get(tool_name)
        if static_tool:
            info = registry.get_tool_info(tool_name)
            return ToolDefinitionResponse(**info)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    return ToolDefinitionResponse(**definition.to_dict())


@router.put(
    "/tools/{tool_name}",
    response_model=ToolDefinitionResponse,
    summary="Update a tool definition",
)
async def update_tool(
    tool_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: ToolDefinitionUpdate,
):
    """Update an existing tool definition."""
    registry = get_tool_registry()

    # Check if it's a static tool (can't modify)
    if registry.get(tool_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify static tools",
        )

    # Build updates
    updates = body.model_dump(exclude_none=True)
    if "parameters" in updates:
        updates["parameters"] = [p.model_dump() if hasattr(p, "model_dump") else p for p in updates["parameters"]]
    if "executor" in updates:
        updates["executor"] = updates["executor"].model_dump() if hasattr(updates["executor"], "model_dump") else updates["executor"]

    # Update
    definition = await registry.update_definition(tool_name, updates)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    logger.info(
        "tool_updated",
        tool_name=tool_name,
        tool_id=definition.id,
        updated_by=context.user_id,
    )

    return ToolDefinitionResponse(**definition.to_dict())


@router.delete(
    "/tools/{tool_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a tool definition",
)
async def delete_tool(
    tool_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
):
    """Delete a tool definition."""
    registry = get_tool_registry()

    # Check if it's a static tool (can't delete)
    if registry.get(tool_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete static tools",
        )

    # Delete
    deleted = await registry.delete_definition(tool_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    logger.info(
        "tool_deleted",
        tool_name=tool_name,
        deleted_by=context.user_id,
    )


@router.post(
    "/tools/{tool_name}/test",
    response_model=TestExecutionResponse,
    summary="Test a tool execution",
)
async def test_tool(
    tool_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: TestExecutionRequest,
):
    """Test execute a tool with sample parameters."""
    registry = get_tool_registry()

    # Get tool definition
    definition = registry.get_definition(tool_name)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    # Create dynamic tool
    tool = DynamicToolFactory.create(definition)

    # Create test context
    tool_context = ToolContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        request_id=context.request_id,
        **body.mock_context,
    )

    # Execute (with dry_run if specified)
    logs = []
    if body.dry_run:
        # Validate only
        logs.append(f"Dry run mode - validating parameters")
        validation_error = tool._validate_params(body.parameters)
        if validation_error:
            return TestExecutionResponse(
                success=False,
                error=validation_error,
                logs=logs,
            )
        logs.append("Parameters validated successfully")
        return TestExecutionResponse(
            success=True,
            result={"message": "Validation passed (dry run)"},
            logs=logs,
        )

    # Actual execution
    logs.append(f"Executing tool '{tool_name}'")
    result = await tool.execute_with_validation(body.parameters, tool_context)
    logs.append(f"Execution completed in {result.duration_ms}ms")

    return TestExecutionResponse(
        success=result.success,
        result=result.data,
        error=result.error,
        duration_ms=result.duration_ms,
        logs=logs,
    )


# =============================================================================
# Agent Management Endpoints
# =============================================================================


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
        definitions = registry.list_enabled_definitions()
    else:
        definitions = registry.list_all_definitions()

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
    tool_registry = get_tool_registry()

    # Check if name already exists
    if registry.get(body.name) or registry.get_definition(body.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent '{body.name}' already exists",
        )

    # Validate tools exist
    for tool_name in body.tools:
        if not tool_registry.get(tool_name) and not tool_registry.get_definition(tool_name):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool '{tool_name}' does not exist",
            )

    # Create definition
    definition = create_agent_definition(
        name=body.name,
        description=body.description,
        capabilities=body.capabilities,
        tools=body.tools,
        executor=body.executor.model_dump(),
        retry_strategy=body.retry_strategy.model_dump(),
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

    definition = registry.get_definition(agent_name)
    if not definition:
        # Check if it's a static agent
        static_agent = registry.get(agent_name)
        if static_agent:
            info = registry.get_agent_info(agent_name)
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
    tool_registry = get_tool_registry()

    # Check if it's a static agent (can't modify)
    if registry.get(agent_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify static agents",
        )

    # Validate tools exist if updating tools
    if body.tools is not None:
        for tool_name in body.tools:
            if not tool_registry.get(tool_name) and not tool_registry.get_definition(tool_name):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tool '{tool_name}' does not exist",
                )

    # Build updates
    updates = body.model_dump(exclude_none=True)
    if "executor" in updates:
        updates["executor"] = updates["executor"].model_dump() if hasattr(updates["executor"], "model_dump") else updates["executor"]
    if "retry_strategy" in updates:
        updates["retry_strategy"] = updates["retry_strategy"].model_dump() if hasattr(updates["retry_strategy"], "model_dump") else updates["retry_strategy"]

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


@router.post(
    "/agents/{agent_name}/test",
    response_model=TestExecutionResponse,
    summary="Test an agent execution",
)
async def test_agent(
    agent_name: str,
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    body: TestExecutionRequest,
):
    """Test execute an agent with sample parameters."""
    from src.agents.state import create_initial_state, RequestContext as AgentRequestContext

    registry = get_agent_registry()

    # Get agent definition
    definition = registry.get_definition(agent_name)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_name}' not found",
        )

    # Create dynamic agent
    agent = DynamicAgentFactory.create(definition)

    # Create test state
    agent_context = AgentRequestContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        permissions=context.permissions,
        request_metadata={"request_id": context.request_id},
    )

    query = body.parameters.get("query", "test query")
    state = create_initial_state(
        user_input=query,
        request_context=agent_context,
    )

    logs = []
    logs.append(f"Testing agent '{agent_name}' with query: {query}")

    if body.dry_run:
        # Validate only
        logs.append("Dry run mode - skipping actual execution")
        return TestExecutionResponse(
            success=True,
            result={"message": "Validation passed (dry run)"},
            logs=logs,
        )

    # Actual execution
    logs.append(f"Executing agent '{agent_name}'")
    result = await agent.execute(state, body.parameters)
    logs.append(f"Execution completed in {result.duration_ms}ms")

    return TestExecutionResponse(
        success=result.status == "success",
        result=result.data,
        error=result.error,
        duration_ms=result.duration_ms,
        logs=logs,
    )


# =============================================================================
# Bulk Operations
# =============================================================================


@router.post(
    "/tools/bulk",
    response_model=BulkOperationResponse,
    summary="Bulk operations on tools",
)
async def bulk_tool_operation(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: BulkOperationRequest,
):
    """Perform bulk operations on tools (enable, disable, delete)."""
    registry = get_tool_registry()
    success_count = 0
    failures = []

    for tool_id in body.ids:
        try:
            # Find definition by ID
            definition = registry.get_definition_by_id(tool_id)
            if not definition:
                failures.append({"id": tool_id, "error": "Not found"})
                continue

            if body.operation == "enable":
                await registry.update_definition(definition.name, {"enabled": True})
                success_count += 1
            elif body.operation == "disable":
                await registry.update_definition(definition.name, {"enabled": False})
                success_count += 1
            elif body.operation == "delete":
                await registry.delete_definition(definition.name)
                success_count += 1

        except Exception as e:
            failures.append({"id": tool_id, "error": str(e)})

    return BulkOperationResponse(
        success_count=success_count,
        failure_count=len(failures),
        failures=failures,
    )


@router.post(
    "/agents/bulk",
    response_model=BulkOperationResponse,
    summary="Bulk operations on agents",
)
async def bulk_agent_operation(
    context: Annotated[RequestContext, Depends(check_permission("admin:agents:write"))],
    body: BulkOperationRequest,
):
    """Perform bulk operations on agents (enable, disable, delete)."""
    registry = get_agent_registry()
    success_count = 0
    failures = []

    for agent_id in body.ids:
        try:
            # Find definition by ID
            definition = registry.get_definition_by_id(agent_id)
            if not definition:
                failures.append({"id": agent_id, "error": "Not found"})
                continue

            if body.operation == "enable":
                await registry.update_definition(definition.name, {"enabled": True})
                success_count += 1
            elif body.operation == "disable":
                await registry.update_definition(definition.name, {"enabled": False})
                success_count += 1
            elif body.operation == "delete":
                await registry.delete_definition(definition.name)
                success_count += 1

        except Exception as e:
            failures.append({"id": agent_id, "error": str(e)})

    return BulkOperationResponse(
        success_count=success_count,
        failure_count=len(failures),
        failures=failures,
    )


# =============================================================================
# Import/Export
# =============================================================================


@router.get(
    "/export",
    summary="Export all definitions",
)
async def export_definitions(
    context: Annotated[RequestContext, Depends(check_permission("admin:export"))],
    include_metadata: bool = Query(default=True, description="Include metadata"),
):
    """Export all tool and agent definitions."""
    tool_registry = get_tool_registry()
    agent_registry = get_agent_registry()

    tools = []
    for definition in tool_registry.list_all_definitions():
        data = definition.to_dict()
        if not include_metadata:
            data.pop("created_at", None)
            data.pop("updated_at", None)
            data.pop("created_by", None)
        tools.append(data)

    agents = []
    for definition in agent_registry.list_all_definitions():
        data = definition.to_dict()
        if not include_metadata:
            data.pop("created_at", None)
            data.pop("updated_at", None)
            data.pop("created_by", None)
        agents.append(data)

    return {
        "version": "1.0",
        "tools": tools,
        "agents": agents,
    }


@router.post(
    "/import",
    response_model=ImportResponse,
    summary="Import definitions",
)
async def import_definitions(
    context: Annotated[RequestContext, Depends(check_permission("admin:import"))],
    body: ImportRequest,
):
    """Import tool and agent definitions."""
    tool_registry = get_tool_registry()
    agent_registry = get_agent_registry()

    errors = []
    warnings = []
    imported_tools = 0
    imported_agents = 0

    data = body.data
    tools_data = data.get("tools", [])
    agents_data = data.get("agents", [])

    # Import tools first (agents may depend on them)
    for tool_data in tools_data:
        try:
            name = tool_data.get("name")

            # Check if exists
            if tool_registry.get(name) or tool_registry.get_definition(name):
                if body.overwrite:
                    await tool_registry.delete_definition(name)
                    warnings.append(f"Overwriting existing tool '{name}'")
                else:
                    warnings.append(f"Skipping existing tool '{name}'")
                    continue

            # Create definition
            definition = ToolDefinition.from_dict({
                "id": f"tool_{uuid4().hex[:12]}",
                **tool_data,
            })

            if not body.validate_only:
                await tool_registry.register_definition(definition, user_id=context.user_id)
                imported_tools += 1

        except Exception as e:
            errors.append(f"Failed to import tool '{tool_data.get('name', 'unknown')}': {str(e)}")

    # Import agents
    for agent_data in agents_data:
        try:
            name = agent_data.get("name")

            # Check if exists
            if agent_registry.get(name) or agent_registry.get_definition(name):
                if body.overwrite:
                    await agent_registry.delete_definition(name)
                    warnings.append(f"Overwriting existing agent '{name}'")
                else:
                    warnings.append(f"Skipping existing agent '{name}'")
                    continue

            # Create definition
            definition = AgentDefinition.from_dict({
                "id": f"agent_{uuid4().hex[:12]}",
                **agent_data,
            })

            if not body.validate_only:
                await agent_registry.register_definition(definition, user_id=context.user_id)
                imported_agents += 1

        except Exception as e:
            errors.append(f"Failed to import agent '{agent_data.get('name', 'unknown')}': {str(e)}")

    return ImportResponse(
        success=len(errors) == 0,
        imported_tools=imported_tools,
        imported_agents=imported_agents,
        errors=errors,
        warnings=warnings,
    )
