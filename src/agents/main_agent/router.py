"""Router component for MainAgent."""

import structlog

from src.agents.state import AgentState

logger = structlog.get_logger()


class Router:
    """
    Router component that determines the next step in the execution flow.
    """

    def __init__(self):
        """Initialize the Router."""
        pass

    def get_next_step(self, state: AgentState) -> str:
        """
        Determine the next step based on current state.

        Args:
            state: Current agent state

        Returns:
            Name of the next node to execute
        """
        plan = state["execution_plan"]
        current_phase = plan.current_phase

        logger.debug(
            "routing_decision",
            session_id=state["session_id"],
            current_phase=current_phase,
            should_continue=state["should_continue"],
        )

        # Check if we should stop
        if not state["should_continue"]:
            return "synthesize"

        # Route based on current phase
        if current_phase == "planning":
            return "plan"

        if current_phase == "planned":
            return "execute_agents"

        if current_phase == "executing":
            return "execute_agents"

        if current_phase == "evaluating":
            return "evaluate"

        if current_phase == "needs_more_info":
            # Check if there are more agents to run
            next_action = state["intermediate_evaluation"].next_action
            if next_action == "catalog":
                return "execute_catalog"
            return "synthesize"

        if current_phase == "complete":
            return "synthesize"

        # Default to synthesis
        return "synthesize"

    def get_agents_to_execute(self, state: AgentState) -> list[str]:
        """
        Get the list of agents to execute in the current step.

        Args:
            state: Current agent state

        Returns:
            List of agent names to execute
        """
        plan = state["execution_plan"]
        executed_agents = set(state["sub_agent_results"].keys())

        # Find next execution group
        for order_item in plan.execution_order:
            # Check if it's a parallel group
            group = next(
                (g for g in plan.parallel_groups if g.group_id == order_item),
                None
            )

            if group:
                # Get agents from the group that haven't been executed
                task_ids = group.task_ids
                agents = [
                    next(
                        (t.agent_name for t in plan.tasks if t.id == tid),
                        None
                    )
                    for tid in task_ids
                ]
                pending = [a for a in agents if a and a not in executed_agents]
                if pending:
                    return pending
            else:
                # It's a single task
                task = next(
                    (t for t in plan.tasks if t.id == order_item),
                    None
                )
                if task and task.agent_name not in executed_agents:
                    return [task.agent_name]

        return []

    def should_evaluate(self, state: AgentState) -> bool:
        """
        Determine if evaluation is needed.

        Args:
            state: Current agent state

        Returns:
            True if evaluation should be performed
        """
        plan = state["execution_plan"]
        executed_agents = set(state["sub_agent_results"].keys())

        # Evaluate after search agents complete
        search_agents = {"knowledge_search", "vector_search"}
        if search_agents.issubset(executed_agents):
            return True

        # Evaluate after catalog agent completes
        if "catalog" in executed_agents:
            return True

        return False

    def get_task_params(self, state: AgentState, agent_name: str) -> dict:
        """
        Get parameters for a specific agent task.

        Args:
            state: Current agent state
            agent_name: Name of the agent

        Returns:
            Dictionary of task parameters
        """
        plan = state["execution_plan"]

        for task in plan.tasks:
            if task.agent_name == agent_name:
                return task.parameters

        # Default parameters
        return {"query": state["user_input"]}

    def is_parallel_group(self, state: AgentState, agents: list[str]) -> bool:
        """
        Check if a list of agents should be executed in parallel.

        Args:
            state: Current agent state
            agents: List of agent names

        Returns:
            True if agents are in a parallel group
        """
        plan = state["execution_plan"]
        agent_set = set(agents)

        for group in plan.parallel_groups:
            group_agents = set()
            for task_id in group.task_ids:
                task = next(
                    (t for t in plan.tasks if t.id == task_id),
                    None
                )
                if task:
                    group_agents.add(task.agent_name)

            if agent_set == group_agents:
                return True

        return False

    def get_group_timeout(self, state: AgentState, agents: list[str]) -> int:
        """
        Get timeout for a group of agents.

        Args:
            state: Current agent state
            agents: List of agent names

        Returns:
            Timeout in seconds
        """
        plan = state["execution_plan"]
        agent_set = set(agents)

        for group in plan.parallel_groups:
            group_agents = set()
            for task_id in group.task_ids:
                task = next(
                    (t for t in plan.tasks if t.id == task_id),
                    None
                )
                if task:
                    group_agents.add(task.agent_name)

            if agent_set == group_agents:
                return group.timeout_seconds

        return 30  # Default timeout
