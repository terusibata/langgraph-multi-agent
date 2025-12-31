"""Parallel execution service."""

import asyncio
from datetime import datetime
from typing import Callable, Any

import structlog

from src.agents.state import AgentState, SubAgentResult
from src.agents.registry import get_agent_registry
from src.config import get_settings

logger = structlog.get_logger()


class ParallelExecutor:
    """
    Service for executing SubAgents in parallel.

    Manages concurrent execution of multiple agents with timeout handling.
    """

    def __init__(self, timeout_seconds: int | None = None):
        """
        Initialize the executor.

        Args:
            timeout_seconds: Default timeout for parallel execution
        """
        settings = get_settings()
        self.default_timeout = timeout_seconds or settings.parallel_timeout_seconds

    async def execute_agents(
        self,
        state: AgentState,
        agent_names: list[str],
        timeout_seconds: int | None = None,
    ) -> dict[str, SubAgentResult]:
        """
        Execute multiple agents in parallel.

        Args:
            state: Current agent state
            agent_names: List of agent names to execute
            timeout_seconds: Optional timeout override

        Returns:
            Dictionary mapping agent names to results
        """
        timeout = timeout_seconds or self.default_timeout
        registry = get_agent_registry()

        logger.info(
            "parallel_execution_start",
            session_id=state["session_id"],
            agents=agent_names,
            timeout=timeout,
        )

        # Create tasks for each agent
        tasks = {}
        for agent_name in agent_names:
            agent = registry.get(agent_name)
            if agent:
                # Get task parameters from plan
                task_params = self._get_task_params(state, agent_name)
                tasks[agent_name] = asyncio.create_task(
                    self._execute_with_timeout(
                        agent.execute_with_retry,
                        state,
                        task_params,
                        agent_name,
                        timeout,
                    )
                )
            else:
                logger.warning(
                    "agent_not_found",
                    agent_name=agent_name,
                    session_id=state["session_id"],
                )

        # Wait for all tasks with overall timeout
        results = {}
        try:
            done, pending = await asyncio.wait(
                tasks.values(),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Collect results
            for agent_name, task in tasks.items():
                if task in done:
                    try:
                        results[agent_name] = task.result()
                    except Exception as e:
                        results[agent_name] = SubAgentResult(
                            agent_name=agent_name,
                            status="failed",
                            error=str(e),
                        )
                else:
                    results[agent_name] = SubAgentResult(
                        agent_name=agent_name,
                        status="failed",
                        error="Execution timeout",
                    )

        except Exception as e:
            logger.error(
                "parallel_execution_error",
                session_id=state["session_id"],
                error=str(e),
            )
            # Return failure results for all agents
            for agent_name in agent_names:
                if agent_name not in results:
                    results[agent_name] = SubAgentResult(
                        agent_name=agent_name,
                        status="failed",
                        error=str(e),
                    )

        logger.info(
            "parallel_execution_complete",
            session_id=state["session_id"],
            results={name: r.status for name, r in results.items()},
        )

        return results

    async def execute_single(
        self,
        state: AgentState,
        agent_name: str,
        timeout_seconds: int | None = None,
    ) -> SubAgentResult:
        """
        Execute a single agent.

        Args:
            state: Current agent state
            agent_name: Agent name to execute
            timeout_seconds: Optional timeout

        Returns:
            SubAgentResult
        """
        results = await self.execute_agents(
            state,
            [agent_name],
            timeout_seconds,
        )
        return results.get(agent_name, SubAgentResult(
            agent_name=agent_name,
            status="failed",
            error="Agent not found",
        ))

    async def _execute_with_timeout(
        self,
        func: Callable,
        state: AgentState,
        task_params: dict,
        agent_name: str,
        timeout: int,
    ) -> SubAgentResult:
        """Execute a function with timeout handling."""
        started_at = datetime.utcnow()
        try:
            result = await asyncio.wait_for(
                func(state, task_params),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            return SubAgentResult(
                agent_name=agent_name,
                status="failed",
                error=f"Execution timeout after {timeout}s",
                duration_ms=duration,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        except Exception as e:
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            return SubAgentResult(
                agent_name=agent_name,
                status="failed",
                error=str(e),
                duration_ms=duration,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    def _get_task_params(self, state: AgentState, agent_name: str) -> dict:
        """Get task parameters for an agent from the execution plan."""
        plan = state["execution_plan"]
        for task in plan.tasks:
            if task.agent_name == agent_name:
                return task.parameters
        return {"query": state["user_input"]}
