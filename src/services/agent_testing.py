"""Agent testing service for sandbox execution and validation."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from src.agents.state import AgentState, RequestContext, SubAgentResult, AdHocAgentSpec, create_initial_state
from src.agents.sub_agents.adhoc import AdHocAgentFactory
from src.agents.sub_agents.dynamic import DynamicAgentFactory
from src.agents.registry import get_agent_registry

logger = structlog.get_logger()


class AgentTestResult:
    """Result of an agent test execution."""

    def __init__(
        self,
        test_id: str,
        agent_name: str,
        agent_type: str,  # "static", "dynamic", "adhoc"
        status: str,  # "success", "partial", "failed"
        data: Any | None = None,
        error: str | None = None,
        duration_ms: int = 0,
        tool_calls: list[dict] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ):
        """Initialize test result."""
        self.test_id = test_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.status = status
        self.data = data
        self.error = error
        self.duration_ms = duration_ms
        self.tool_calls = tool_calls or []
        self.started_at = started_at or datetime.now(timezone.utc)
        self.completed_at = completed_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "tool_calls": self.tool_calls,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AgentTester:
    """Service for testing agents in a sandbox environment."""

    async def test_static_agent(
        self,
        agent_name: str,
        test_input: str,
        task_params: dict | None = None,
    ) -> AgentTestResult:
        """
        Test a static agent with given input.

        Args:
            agent_name: Name of the static agent
            test_input: Test input string
            task_params: Optional task parameters

        Returns:
            AgentTestResult
        """
        test_id = f"test_{uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)

        logger.info(
            "agent_test_start",
            test_id=test_id,
            agent_name=agent_name,
            agent_type="static",
        )

        try:
            # Get static agent from registry
            registry = get_agent_registry()
            agent = registry.get(agent_name)

            if not agent:
                return AgentTestResult(
                    test_id=test_id,
                    agent_name=agent_name,
                    agent_type="static",
                    status="failed",
                    error=f"Static agent '{agent_name}' not found",
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

            # Create test state
            state = self._create_test_state(test_input)

            # Execute agent
            result = await agent.execute_with_retry(state, task_params or {})

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.info(
                "agent_test_complete",
                test_id=test_id,
                agent_name=agent_name,
                status=result.status,
                duration_ms=duration_ms,
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=agent_name,
                agent_type="static",
                status=result.status,
                data=result.data,
                error=result.error,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.error(
                "agent_test_error",
                test_id=test_id,
                agent_name=agent_name,
                error=str(e),
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=agent_name,
                agent_type="static",
                status="failed",
                error=str(e),
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

    async def test_dynamic_agent(
        self,
        agent_name: str,
        test_input: str,
        task_params: dict | None = None,
    ) -> AgentTestResult:
        """
        Test a dynamic agent with given input.

        Args:
            agent_name: Name of the dynamic agent
            test_input: Test input string
            task_params: Optional task parameters

        Returns:
            AgentTestResult
        """
        test_id = f"test_{uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)

        logger.info(
            "agent_test_start",
            test_id=test_id,
            agent_name=agent_name,
            agent_type="dynamic",
        )

        try:
            # Get dynamic agent definition from registry
            registry = get_agent_registry()
            definition = await registry.get_definition(agent_name)

            if not definition:
                return AgentTestResult(
                    test_id=test_id,
                    agent_name=agent_name,
                    agent_type="dynamic",
                    status="failed",
                    error=f"Dynamic agent '{agent_name}' not found",
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

            # Create dynamic agent instance
            agent = DynamicAgentFactory.create(definition)

            # Create test state
            state = self._create_test_state(test_input)

            # Execute agent
            result = await agent.execute_with_retry(state, task_params or {})

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.info(
                "agent_test_complete",
                test_id=test_id,
                agent_name=agent_name,
                status=result.status,
                duration_ms=duration_ms,
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=agent_name,
                agent_type="dynamic",
                status=result.status,
                data=result.data,
                error=result.error,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.error(
                "agent_test_error",
                test_id=test_id,
                agent_name=agent_name,
                error=str(e),
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=agent_name,
                agent_type="dynamic",
                status="failed",
                error=str(e),
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

    async def test_adhoc_agent(
        self,
        spec: AdHocAgentSpec | dict,
        test_input: str,
        task_params: dict | None = None,
    ) -> AgentTestResult:
        """
        Test an ad-hoc agent specification.

        Args:
            spec: Ad-hoc agent specification (dict or AdHocAgentSpec)
            test_input: Test input string
            task_params: Optional task parameters

        Returns:
            AgentTestResult
        """
        test_id = f"test_{uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)

        # Convert dict to AdHocAgentSpec if needed
        if isinstance(spec, dict):
            spec = AdHocAgentSpec(**spec)

        logger.info(
            "agent_test_start",
            test_id=test_id,
            agent_name=spec.name,
            agent_type="adhoc",
            purpose=spec.purpose,
            tools=spec.tools,
        )

        try:
            # Create ad-hoc agent
            agent = AdHocAgentFactory.create(spec)

            # Create test state
            state = self._create_test_state(test_input)

            # Execute agent
            result = await agent.execute_with_retry(state, task_params or {})

            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.info(
                "agent_test_complete",
                test_id=test_id,
                agent_name=spec.name,
                status=result.status,
                duration_ms=duration_ms,
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=spec.name,
                agent_type="adhoc",
                status=result.status,
                data=result.data,
                error=result.error,
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

        except Exception as e:
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.error(
                "agent_test_error",
                test_id=test_id,
                agent_name=spec.name,
                error=str(e),
            )

            return AgentTestResult(
                test_id=test_id,
                agent_name=spec.name,
                agent_type="adhoc",
                status="failed",
                error=str(e),
                duration_ms=duration_ms,
                started_at=started_at,
                completed_at=completed_at,
            )

    def _create_test_state(self, test_input: str) -> AgentState:
        """Create a test agent state."""
        # Create minimal test request context
        request_context = RequestContext(
            tenant_id="test_tenant",
            user_id="test_user",
            request_id=f"test_{uuid4().hex[:8]}",
            service_tokens={},
            permissions=[],
            company_context=None,
        )

        # Create initial state
        state = create_initial_state(
            user_input=test_input,
            request_context=request_context,
            thread_id=None,
        )

        return state


def get_agent_tester() -> AgentTester:
    """Get the agent tester instance."""
    return AgentTester()
