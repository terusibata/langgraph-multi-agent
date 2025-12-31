"""Planner component for MainAgent."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import structlog

from src.agents.state import AgentState, ExecutionPlan, Task, ParallelGroup
from src.agents.registry import get_agent_registry

logger = structlog.get_logger()


PLANNER_SYSTEM_PROMPT = """あなたはマルチエージェントシステムの計画立案者です。
ユーザーの質問を分析し、適切な実行計画を立ててください。

利用可能なエージェント:
{available_agents}

実行計画のルール:
1. 検索系タスク（knowledge_search, vector_search）は並列実行可能
2. カタログ調査（catalog）は検索結果を評価した後に実行
3. 各タスクには適切なパラメータを設定

出力形式（JSON）:
{{
    "intent": "ユーザーの意図の要約",
    "tasks": [
        {{
            "agent_name": "エージェント名",
            "priority": 1,
            "parameters": {{"query": "検索クエリ"}}
        }}
    ],
    "parallel_groups": [
        {{
            "task_indices": [0, 1],
            "timeout_seconds": 30
        }}
    ]
}}
"""


class Planner:
    """
    Planner component that analyzes user intent and creates execution plans.
    """

    def __init__(self, llm: Any):
        """
        Initialize the Planner.

        Args:
            llm: Language model instance for planning
        """
        self.llm = llm
        self.output_parser = JsonOutputParser()

    def get_available_agents_description(self) -> str:
        """Get description of available agents."""
        registry = get_agent_registry()
        agents = registry.list_enabled()

        descriptions = []
        for agent in agents:
            config = registry.get_agent_config(agent.name)
            capabilities = ", ".join(agent.capabilities)
            descriptions.append(
                f"- {agent.name}: {agent.description}\n  能力: {capabilities}"
            )

        return "\n".join(descriptions)

    async def create_plan(self, state: AgentState) -> ExecutionPlan:
        """
        Create an execution plan based on user input.

        Args:
            state: Current agent state

        Returns:
            ExecutionPlan with tasks and parallel groups
        """
        user_input = state["user_input"]

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(
                available_agents=self.get_available_agents_description()
            )),
            HumanMessage(content=f"ユーザーの質問: {user_input}"),
        ])

        try:
            # Get LLM response
            chain = prompt | self.llm | self.output_parser
            result = await chain.ainvoke({})

            # Parse result into ExecutionPlan
            plan = self._parse_plan_result(result)

            logger.info(
                "plan_created",
                session_id=state["session_id"],
                num_tasks=len(plan.tasks),
                num_parallel_groups=len(plan.parallel_groups),
            )

            return plan

        except Exception as e:
            logger.error(
                "plan_creation_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            # Return a default plan
            return self._create_default_plan(user_input)

    def _parse_plan_result(self, result: dict) -> ExecutionPlan:
        """Parse LLM result into ExecutionPlan."""
        tasks = []
        for i, task_data in enumerate(result.get("tasks", [])):
            task = Task(
                id=f"task_{i}",
                agent_name=task_data["agent_name"],
                priority=task_data.get("priority", 0),
                parameters=task_data.get("parameters", {}),
            )
            tasks.append(task)

        parallel_groups = []
        for group_data in result.get("parallel_groups", []):
            task_indices = group_data.get("task_indices", [])
            task_ids = [tasks[i].id for i in task_indices if i < len(tasks)]
            group = ParallelGroup(
                task_ids=task_ids,
                timeout_seconds=group_data.get("timeout_seconds", 30),
            )
            parallel_groups.append(group)

        # Build execution order
        execution_order = []
        grouped_task_ids = set()
        for group in parallel_groups:
            execution_order.append(group.group_id)
            grouped_task_ids.update(group.task_ids)

        # Add ungrouped tasks
        for task in tasks:
            if task.id not in grouped_task_ids:
                execution_order.append(task.id)

        return ExecutionPlan(
            tasks=tasks,
            parallel_groups=parallel_groups,
            execution_order=execution_order,
            current_phase="planned",
        )

    def _create_default_plan(self, user_input: str) -> ExecutionPlan:
        """Create a default plan when LLM planning fails."""
        # Default: parallel search with knowledge and vector search
        tasks = [
            Task(
                id="task_0",
                agent_name="knowledge_search",
                priority=1,
                parameters={"query": user_input},
            ),
            Task(
                id="task_1",
                agent_name="vector_search",
                priority=1,
                parameters={"query": user_input},
            ),
        ]

        parallel_groups = [
            ParallelGroup(
                group_id="search_phase",
                task_ids=["task_0", "task_1"],
                timeout_seconds=30,
            ),
        ]

        return ExecutionPlan(
            tasks=tasks,
            parallel_groups=parallel_groups,
            execution_order=["search_phase"],
            current_phase="planned",
        )

    def get_plan_summary(self, plan: ExecutionPlan) -> dict:
        """
        Get a summary of the execution plan.

        Args:
            plan: The execution plan

        Returns:
            Dictionary with plan summary
        """
        initial_agents = [task.agent_name for task in plan.tasks]
        parallel_groups_summary = [
            [
                next(
                    (t.agent_name for t in plan.tasks if t.id == tid),
                    tid
                )
                for tid in group.task_ids
            ]
            for group in plan.parallel_groups
        ]

        return {
            "initial_agents": initial_agents,
            "parallel_groups": parallel_groups_summary,
            "estimated_steps": len(plan.execution_order),
        }
