"""MainAgent - Supervisor agent that orchestrates the multi-agent system."""

from typing import Any, AsyncIterator

import structlog

from src.agents.state import AgentState
from src.agents.main_agent.planner import Planner
from src.agents.main_agent.router import Router
from src.agents.main_agent.evaluator import Evaluator
from src.agents.main_agent.synthesizer import Synthesizer
from src.agents.main_agent.direct_tool_executor import DirectToolExecutor
from src.config import get_settings

logger = structlog.get_logger()


class MainAgent:
    """
    MainAgent serves as the supervisor in the hierarchical multi-agent system.

    Responsibilities:
    - Analyze user intent
    - Create execution plans (with dynamic ad-hoc agent generation in dynamic mode)
    - Route to appropriate SubAgents
    - Evaluate intermediate results
    - Generate final responses

    The MainAgent supports two modes:
    - Simple mode: Uses only pre-defined agents
    - Dynamic mode: Can generate ad-hoc agents on-the-fly based on available tools
    """

    def __init__(
        self,
        llm: Any,
        model_id: str | None = None,
        dynamic_mode: bool = True,
    ):
        """
        Initialize the MainAgent.

        Args:
            llm: Language model instance
            model_id: Optional model ID override
            dynamic_mode: Enable dynamic ad-hoc agent generation
        """
        self.llm = llm
        self.model_id = model_id or get_settings().default_model_id
        self.dynamic_mode = dynamic_mode

        # Initialize components
        self.planner = Planner(llm, dynamic_mode=dynamic_mode)
        self.router = Router()
        self.evaluator = Evaluator(llm)
        self.synthesizer = Synthesizer(llm)
        self.direct_tool_executor = DirectToolExecutor(llm)

        logger.info(
            "main_agent_initialized",
            model_id=self.model_id,
            dynamic_mode=self.dynamic_mode,
        )

    async def plan(self, state: AgentState) -> AgentState:
        """
        Create an execution plan based on user input.

        Args:
            state: Current agent state

        Returns:
            Updated state with execution plan
        """
        logger.info(
            "main_agent_planning",
            session_id=state["session_id"],
            user_input=state["user_input"][:100],
        )

        # Create execution plan
        plan = await self.planner.create_plan(state)

        # Update state
        state["execution_plan"] = plan
        state["execution_plan"].current_phase = "planned"

        return state

    def route(self, state: AgentState) -> str:
        """
        Determine the next step in the execution flow.

        Args:
            state: Current agent state

        Returns:
            Name of the next node
        """
        return self.router.get_next_step(state)

    async def evaluate(self, state: AgentState) -> AgentState:
        """
        Evaluate intermediate results and decide next action.

        Args:
            state: Current agent state

        Returns:
            Updated state with evaluation
        """
        logger.info(
            "main_agent_evaluating",
            session_id=state["session_id"],
            agents_completed=list(state["sub_agent_results"].keys()),
        )

        # Perform evaluation
        evaluation = await self.evaluator.evaluate(state)

        # Update state
        state["intermediate_evaluation"] = evaluation

        # Update phase based on evaluation
        if evaluation.next_action == "catalog":
            state["execution_plan"].current_phase = "needs_more_info"
        elif evaluation.next_action == "respond":
            state["execution_plan"].current_phase = "complete"
        else:
            state["execution_plan"].current_phase = "evaluating"

        return state

    async def synthesize(self, state: AgentState) -> AgentState:
        """
        Generate the final response.

        Args:
            state: Current agent state

        Returns:
            Updated state with final response
        """
        logger.info(
            "main_agent_synthesizing",
            session_id=state["session_id"],
        )

        # Generate response
        response = await self.synthesizer.synthesize(state)

        # Update state
        state["final_response"] = response
        state["should_continue"] = False
        state["execution_plan"].current_phase = "complete"

        return state

    async def synthesize_stream(self, state: AgentState) -> AsyncIterator[str]:
        """
        Generate the final response with streaming.

        Args:
            state: Current agent state

        Yields:
            Response tokens
        """
        logger.info(
            "main_agent_synthesizing_stream",
            session_id=state["session_id"],
        )

        full_response = ""
        async for token in self.synthesizer.synthesize_stream(state):
            full_response += token
            yield token

        # Update state after streaming completes
        state["final_response"] = full_response
        state["should_continue"] = False
        state["execution_plan"].current_phase = "complete"

    def get_agents_to_execute(self, state: AgentState) -> list[str]:
        """
        Get list of agents to execute.

        Args:
            state: Current agent state

        Returns:
            List of agent names
        """
        return self.router.get_agents_to_execute(state)

    def get_task_params(self, state: AgentState, agent_name: str) -> dict:
        """
        Get task parameters for an agent.

        Args:
            state: Current agent state
            agent_name: Name of the agent

        Returns:
            Task parameters dictionary
        """
        return self.router.get_task_params(state, agent_name)

    def is_parallel_execution(self, state: AgentState, agents: list[str]) -> bool:
        """
        Check if agents should be executed in parallel.

        Args:
            state: Current agent state
            agents: List of agent names

        Returns:
            True if parallel execution
        """
        return self.router.is_parallel_group(state, agents)

    def get_execution_timeout(self, state: AgentState, agents: list[str]) -> int:
        """
        Get timeout for agent execution.

        Args:
            state: Current agent state
            agents: List of agent names

        Returns:
            Timeout in seconds
        """
        return self.router.get_group_timeout(state, agents)

    def get_plan_summary(self, state: AgentState) -> dict:
        """
        Get execution plan summary.

        Args:
            state: Current agent state

        Returns:
            Plan summary dictionary
        """
        return self.planner.get_plan_summary(state["execution_plan"])

    def get_evaluation_summary(self, state: AgentState) -> dict:
        """
        Get evaluation summary.

        Args:
            state: Current agent state

        Returns:
            Evaluation summary dictionary
        """
        return self.evaluator.get_evaluation_summary(state["intermediate_evaluation"])
