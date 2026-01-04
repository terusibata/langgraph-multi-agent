"""Parallel execution service with ad-hoc agent support."""

import asyncio
from datetime import datetime, timezone
from typing import Callable, Any

import structlog

from src.agents.state import AgentState, SubAgentResult, Task
from src.agents.registry import get_agent_registry
from src.agents.sub_agents.dynamic import DynamicAgentFactory
from src.agents.sub_agents.adhoc import AdHocAgentFactory
from src.config import get_settings

logger = structlog.get_logger()


class ParallelExecutor:
    """
    Service for executing SubAgents in parallel.

    Supports both pre-defined agents (static/dynamic) and ad-hoc agents
    generated at runtime by the Planner.
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
        Execute multiple agents in parallel by name.

        This method is for backwards compatibility with the simple mode.

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

            # If not a static agent, check for dynamic agent definition
            if not agent:
                definition = await registry.get_definition(agent_name)
                if definition:
                    agent = DynamicAgentFactory.create(definition)
                    logger.debug(
                        "using_dynamic_agent",
                        agent_name=agent_name,
                        session_id=state["session_id"],
                    )

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

        return await self._wait_for_tasks(tasks, agent_names, timeout, state)

    async def execute_tasks(
        self,
        state: AgentState,
        task_list: list[Task],
        timeout_seconds: int | None = None,
    ) -> dict[str, SubAgentResult]:
        """
        Execute multiple tasks in parallel.

        This method supports both pre-defined and ad-hoc agents.

        Args:
            state: Current agent state
            task_list: List of Task objects to execute
            timeout_seconds: Optional timeout override

        Returns:
            Dictionary mapping agent names to results
        """
        timeout = timeout_seconds or self.default_timeout
        registry = get_agent_registry()

        logger.info(
            "parallel_task_execution_start",
            session_id=state["session_id"],
            num_tasks=len(task_list),
            adhoc_count=sum(1 for t in task_list if t.is_adhoc),
            timeout=timeout,
        )

        # Create async tasks for each task
        async_tasks = {}

        for task in task_list:
            agent_name = task.effective_agent_name

            if task.is_adhoc:
                # Create ad-hoc agent from specification
                agent = AdHocAgentFactory.create(task.adhoc_spec)
                logger.info(
                    "created_adhoc_agent",
                    agent_name=agent_name,
                    purpose=task.adhoc_spec.purpose,
                    tools=task.adhoc_spec.tools,
                    session_id=state["session_id"],
                )
            else:
                # Get pre-defined agent (static or dynamic)
                agent = registry.get(task.agent_name)

                if not agent:
                    definition = await registry.get_definition(task.agent_name)
                    if definition:
                        agent = DynamicAgentFactory.create(definition)

            if agent:
                async_tasks[agent_name] = asyncio.create_task(
                    self._execute_with_timeout(
                        agent.execute_with_retry,
                        state,
                        task.parameters,
                        agent_name,
                        timeout,
                    )
                )
            else:
                logger.warning(
                    "agent_not_found_for_task",
                    task_id=task.id,
                    agent_name=agent_name,
                    is_adhoc=task.is_adhoc,
                    session_id=state["session_id"],
                )

        agent_names = [t.effective_agent_name for t in task_list]
        return await self._wait_for_tasks(async_tasks, agent_names, timeout, state)

    async def _wait_for_tasks(
        self,
        tasks: dict[str, asyncio.Task],
        agent_names: list[str],
        timeout: int,
        state: AgentState,
    ) -> dict[str, SubAgentResult]:
        """Wait for all tasks to complete and collect results."""
        results = {}

        if not tasks:
            return results

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
        Execute a single agent by name.

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

    async def execute_single_task(
        self,
        state: AgentState,
        task: Task,
        timeout_seconds: int | None = None,
    ) -> SubAgentResult:
        """
        Execute a single task.

        Args:
            state: Current agent state
            task: Task to execute
            timeout_seconds: Optional timeout

        Returns:
            SubAgentResult
        """
        results = await self.execute_tasks(
            state,
            [task],
            timeout_seconds,
        )
        return results.get(task.effective_agent_name, SubAgentResult(
            agent_name=task.effective_agent_name,
            status="failed",
            error="Task execution failed",
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
        started_at = datetime.now(timezone.utc)
        try:
            result = await asyncio.wait_for(
                func(state, task_params),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            duration = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            return SubAgentResult(
                agent_name=agent_name,
                status="failed",
                error=f"Execution timeout after {timeout}s",
                duration_ms=duration,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            duration = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            return SubAgentResult(
                agent_name=agent_name,
                status="failed",
                error=str(e),
                duration_ms=duration,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

    def _get_task_params(self, state: AgentState, agent_name: str) -> dict:
        """Get task parameters for an agent from the execution plan."""
        plan = state["execution_plan"]
        for task in plan.tasks:
            if task.effective_agent_name == agent_name:
                return task.parameters
        return {"query": state["user_input"]}


def get_tasks_for_group(
    plan,
    group_id: str,
) -> list[Task]:
    """Get all tasks belonging to a parallel group."""
    for group in plan.parallel_groups:
        if group.group_id == group_id:
            return [
                task for task in plan.tasks
                if task.id in group.task_ids
            ]
    return []


def get_next_tasks_to_execute(
    plan,
    completed_agents: set[str],
) -> list[Task]:
    """
    Get the next batch of tasks to execute.

    This considers the execution order and returns either:
    - All tasks in the next parallel group, or
    - The next individual task
    """
    for item in plan.execution_order:
        # Check if it's a group
        for group in plan.parallel_groups:
            if group.group_id == item:
                group_tasks = [
                    task for task in plan.tasks
                    if task.id in group.task_ids
                    and task.effective_agent_name not in completed_agents
                ]
                if group_tasks:
                    return group_tasks
                break
        else:
            # It's an individual task ID
            for task in plan.tasks:
                if task.id == item and task.effective_agent_name not in completed_agents:
                    return [task]

    return []
