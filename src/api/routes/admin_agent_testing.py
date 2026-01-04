"""Admin API routes for agent testing."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.services.agent_testing import get_agent_tester
from src.agents.state import AdHocAgentSpec

router = APIRouter()


# ============================================================================
# Request/Response schemas
# ============================================================================


class AgentTestRequest(BaseModel):
    """Request to test an agent."""

    test_input: str = Field(..., description="Test input for the agent")
    task_params: dict = Field(default_factory=dict, description="Optional task parameters")


class AdHocAgentTestRequest(BaseModel):
    """Request to test an ad-hoc agent."""

    spec: dict = Field(..., description="Ad-hoc agent specification")
    test_input: str = Field(..., description="Test input for the agent")
    task_params: dict = Field(default_factory=dict, description="Optional task parameters")


class AgentTestResponse(BaseModel):
    """Response from agent test."""

    test_id: str
    agent_name: str
    agent_type: str
    status: str
    data: dict | list | None = None
    error: str | None = None
    duration_ms: int
    tool_calls: list[dict] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


# ============================================================================
# Agent testing endpoints
# ============================================================================


@router.post(
    "/agents/static/{agent_name}/test",
    response_model=AgentTestResponse,
    summary="Test a static agent",
    description="Execute a static agent in a sandbox environment with test input",
)
async def test_static_agent(
    agent_name: str,
    request: AgentTestRequest,
):
    """Test a static agent with given input."""
    tester = get_agent_tester()
    result = await tester.test_static_agent(
        agent_name=agent_name,
        test_input=request.test_input,
        task_params=request.task_params,
    )

    return AgentTestResponse(**result.to_dict())


@router.post(
    "/agents/dynamic/{agent_name}/test",
    response_model=AgentTestResponse,
    summary="Test a dynamic agent",
    description="Execute a dynamic agent in a sandbox environment with test input",
)
async def test_dynamic_agent(
    agent_name: str,
    request: AgentTestRequest,
):
    """Test a dynamic agent with given input."""
    tester = get_agent_tester()
    result = await tester.test_dynamic_agent(
        agent_name=agent_name,
        test_input=request.test_input,
        task_params=request.task_params,
    )

    if result.status == "failed" and "not found" in (result.error or ""):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.error,
        )

    return AgentTestResponse(**result.to_dict())


@router.post(
    "/agents/adhoc/test",
    response_model=AgentTestResponse,
    summary="Test an ad-hoc agent specification",
    description="Execute an ad-hoc agent specification in a sandbox environment",
)
async def test_adhoc_agent(
    request: AdHocAgentTestRequest,
):
    """Test an ad-hoc agent specification."""
    tester = get_agent_tester()

    try:
        # Validate and create spec
        spec = AdHocAgentSpec(**request.spec)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ad-hoc agent specification: {str(e)}",
        )

    result = await tester.test_adhoc_agent(
        spec=spec,
        test_input=request.test_input,
        task_params=request.task_params,
    )

    return AgentTestResponse(**result.to_dict())
