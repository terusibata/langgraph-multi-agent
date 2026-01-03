"""Admin API routes for tool management."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
import structlog

from src.api.middleware.access_key import RequestContext, check_permission
from src.api.schemas.admin import (
    ToolDefinitionCreate,
    ToolDefinitionUpdate,
    ToolDefinitionResponse,
    ToolListResponse as AdminToolListResponse,
    TestExecutionRequest,
    TestExecutionResponse,
)
from src.agents.registry import (
    get_tool_registry,
    create_tool_definition,
)
from src.agents.tools.dynamic import DynamicToolFactory
from src.agents.tools.base import ToolContext

logger = structlog.get_logger()

router = APIRouter(tags=["Admin - Tools"])


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
        definitions = await registry.list_enabled_definitions()
    else:
        definitions = await registry.list_all_definitions()

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
    existing_def = await registry.get_definition(body.name)
    if registry.get(body.name) or existing_def:
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

    definition = await registry.get_definition(tool_name)
    if not definition:
        # Check if it's a static tool
        static_tool = registry.get(tool_name)
        if static_tool:
            info = await registry.get_tool_info(tool_name)
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
    definition = await registry.get_definition(tool_name)
    if not definition:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )

    # Create dynamic tool
    tool = DynamicToolFactory.create(definition)

    # Create test context (only allow service_tokens from mock_context)
    mock_service_tokens = body.mock_context.get("service_tokens", {})
    tool_context = ToolContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        request_id=context.request_id,
        service_tokens=mock_service_tokens,
    )

    # Execute (with dry_run if specified)
    logs = []
    if body.dry_run:
        # Validate only
        logs.append("Dry run mode - validating parameters")
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
