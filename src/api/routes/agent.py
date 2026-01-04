"""Agent API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse
import structlog

from src.api.middleware.access_key import verify_access_key, RequestContext, check_permission
from src.api.middleware.service_tokens import extract_service_tokens
from src.api.schemas.request import AgentStreamRequest, AgentInvokeRequest
from src.api.schemas.response import (
    ErrorResponse,
    ModelListResponse,
    ModelInfo,
    AgentListResponse,
    AgentInfo,
    ToolListResponse,
    ToolInfo,
)
from src.agents.graph import create_graph
from src.agents.state import RequestContext as AgentRequestContext, CompanyContext as AgentCompanyContext
from src.agents.registry import get_agent_registry, get_tool_registry, initialize_registries
from src.services.streaming import SSEManager
from src.services.error import get_error_handler, AgentError
from src.config import get_settings
from src.config.models import get_available_models

logger = structlog.get_logger()

router = APIRouter(prefix="/agent", tags=["Agent"])


@router.post(
    "/stream",
    summary="Execute agent with SSE streaming",
    response_class=EventSourceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        403: {"model": ErrorResponse, "description": "Authorization error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def stream_agent(
    request: Request,
    body: AgentStreamRequest,
    context: Annotated[RequestContext, Depends(check_permission("agent:invoke"))],
):
    """
    Execute the multi-agent system with SSE streaming.

    Returns Server-Sent Events with:
    - session_start: When processing begins
    - plan_created: Execution plan details
    - agent_start/end: SubAgent execution status
    - tool_call/result: Tool execution details
    - evaluation: Intermediate evaluation results
    - token: Generated response tokens
    - session_complete: Final results and metrics
    - error: Error information if failure occurs
    """
    # Extract service tokens
    service_tokens = await extract_service_tokens(request)

    # Convert company context if provided
    agent_company_context = None
    if body.company_context:
        agent_company_context = AgentCompanyContext(
            company_id=body.company_context.company_id,
            company_name=body.company_context.company_name,
            vision=body.company_context.vision,
            terminology=body.company_context.terminology,
            reference_info=body.company_context.reference_info,
            metadata=body.company_context.metadata,
        )

    # Build agent request context
    agent_context = AgentRequestContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        permissions=context.permissions,
        service_tokens=service_tokens,
        company_context=agent_company_context,
        request_metadata={
            "request_id": context.request_id,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
        },
    )

    logger.info(
        "agent_stream_request",
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        thread_id=body.thread_id,
        message_length=len(body.message),
        model_id=body.model_id,
        fast_response=body.fast_response,
        direct_tool_mode=body.direct_tool_mode,
    )

    # Create graph and SSE manager
    graph = create_graph(body.model_id)
    sse_manager = SSEManager(
        session_id=f"sess_{context.request_id}",
        thread_id=body.thread_id or "new",
    )

    # Run graph with streaming
    async def generate():
        async for event in graph.stream(
            user_input=body.message,
            request_context=agent_context,
            sse_manager=sse_manager,
            thread_id=body.thread_id,
            fast_response=body.fast_response,
            direct_tool_mode=body.direct_tool_mode,
            response_format=body.response_format,
            response_schema=body.response_schema,
        ):
            yield event

    return EventSourceResponse(generate())


@router.post(
    "/invoke",
    summary="Execute agent synchronously",
    responses={
        401: {"model": ErrorResponse, "description": "Authentication error"},
        403: {"model": ErrorResponse, "description": "Authorization error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def invoke_agent(
    request: Request,
    body: AgentInvokeRequest,
    context: Annotated[RequestContext, Depends(check_permission("agent:invoke"))],
):
    """
    Execute the multi-agent system synchronously.

    Returns the complete result after processing finishes.
    Use /stream for real-time updates during processing.
    """
    # Extract service tokens
    service_tokens = await extract_service_tokens(request)

    # Convert company context if provided
    agent_company_context = None
    if body.company_context:
        agent_company_context = AgentCompanyContext(
            company_id=body.company_context.company_id,
            company_name=body.company_context.company_name,
            vision=body.company_context.vision,
            terminology=body.company_context.terminology,
            reference_info=body.company_context.reference_info,
            metadata=body.company_context.metadata,
        )

    # Build agent request context
    agent_context = AgentRequestContext(
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        permissions=context.permissions,
        service_tokens=service_tokens,
        company_context=agent_company_context,
        request_metadata={
            "request_id": context.request_id,
            "client_ip": context.client_ip,
            "user_agent": context.user_agent,
        },
    )

    logger.info(
        "agent_invoke_request",
        tenant_id=context.tenant_id,
        user_id=context.user_id,
        thread_id=body.thread_id,
        fast_response=body.fast_response,
        direct_tool_mode=body.direct_tool_mode,
    )

    try:
        # Create and run graph
        graph = create_graph(body.model_id)
        final_state = await graph.run(
            user_input=body.message,
            request_context=agent_context,
            thread_id=body.thread_id,
            fast_response=body.fast_response,
            direct_tool_mode=body.direct_tool_mode,
        )

        # Build response
        return {
            "session_id": final_state["session_id"],
            "thread_id": final_state["thread_id"],
            "response": final_state["final_response"],
            "metrics": {
                "duration_ms": final_state["metrics"].duration_ms,
                "total_tokens": (
                    final_state["metrics"].total_input_tokens +
                    final_state["metrics"].total_output_tokens
                ),
                "total_cost_usd": final_state["metrics"].total_cost_usd,
            },
        }

    except AgentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.to_dict(),
        )
    except Exception as e:
        error_handler = get_error_handler()
        error = error_handler.handle_exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.to_dict(),
        )


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available models",
)
async def list_models(
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Get list of available LLM models."""
    settings = get_settings()
    models = get_available_models()

    return ModelListResponse(
        models=[
            ModelInfo(
                model_id=m["model_id"],
                provider=m["provider"],
                max_tokens=m["max_tokens"],
                context_window=m["context_window"],
                is_default=(m["model_id"] == settings.default_model_id),
            )
            for m in models
        ],
        default_model_id=settings.default_model_id,
    )


@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List available SubAgents",
)
async def list_agents(
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Get list of available SubAgents."""
    await initialize_registries()
    registry = get_agent_registry()

    agents = []
    for agent in registry.list_all():
        info = await registry.get_agent_info(agent.name)
        if info:
            agents.append(AgentInfo(**info))

    return AgentListResponse(agents=agents)


@router.get(
    "/tools",
    response_model=ToolListResponse,
    summary="List available tools",
)
async def list_tools(
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Get list of available tools."""
    await initialize_registries()
    registry = get_tool_registry()

    tools = []
    for tool in registry.list_all():
        info = await registry.get_tool_info(tool.name)
        if info:
            tools.append(ToolInfo(**info))

    return ToolListResponse(tools=tools)
