"""Planner component for MainAgent with dynamic ad-hoc agent generation."""

from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import structlog

from src.agents.state import (
    AgentState,
    ExecutionPlan,
    Task,
    ParallelGroup,
    AdHocAgentSpec,
)
from src.agents.registry import get_agent_registry, get_tool_registry
from src.config.models import should_use_prompt_caching

logger = structlog.get_logger()


# =============================================================================
# System prompts for different planning modes
# =============================================================================

TOOL_ANALYSIS_PROMPT = """あなたはマルチエージェントシステムのインテリジェントプランナーです。
ユーザーのリクエストを分析し、利用可能なツールを使って最適な実行計画を立案してください。

## 利用可能なツール
{available_tools}

## テンプレートエージェント（事前定義済み）
{template_agents}

## 計画立案の原則

1. **ツールファースト思考**: まずどのツールが必要かを考え、次にそれらをどうグループ化するか決定
2. **並列実行の最大化**: 依存関係がないタスクは並列実行グループにまとめる
3. **テンプレート活用**: 適切なテンプレートエージェントがあれば優先的に使用
4. **Ad-hoc生成**: テンプレートが不適切な場合、必要なツールを組み合わせてad-hocエージェントを生成

## 出力形式（JSON）

{{
    "intent": "ユーザーの意図の要約",
    "analysis": {{
        "required_capabilities": ["必要な能力のリスト"],
        "relevant_tools": ["関連するツール名のリスト"],
        "can_parallelize": true/false,
        "reasoning": "分析の理由"
    }},
    "tasks": [
        {{
            "type": "template",
            "agent_name": "テンプレートエージェント名",
            "priority": 1,
            "parameters": {{"query": "検索クエリ"}}
        }},
        {{
            "type": "adhoc",
            "name": "adhoc_search_agent",
            "purpose": "このエージェントの目的",
            "tools": ["tool1", "tool2"],
            "expected_output": "期待される出力形式",
            "reasoning": "このツール組み合わせを選んだ理由",
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

## 重要なポイント

- 検索系タスクは通常並列実行可能
- 結果を使う後続タスクは依存関係を考慮
- 1つのad-hocエージェントには関連性の高いツールのみを含める
- ツールの説明をよく読み、適切な組み合わせを選択
"""

SIMPLE_PLANNER_PROMPT = """あなたはマルチエージェントシステムの計画立案者です。
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

    The planner can operate in two modes:
    1. Simple mode: Uses pre-defined agents only
    2. Dynamic mode: Analyzes tools and generates ad-hoc agents as needed
    """

    def __init__(self, llm: Any, dynamic_mode: bool = True):
        """
        Initialize the Planner.

        Args:
            llm: Language model instance for planning
            dynamic_mode: If True, enables ad-hoc agent generation
        """
        self.llm = llm
        self.dynamic_mode = dynamic_mode
        self.output_parser = JsonOutputParser()

    def get_available_tools_description(self) -> str:
        """Get description of all available tools (static and dynamic)."""
        tool_registry = get_tool_registry()
        descriptions = []

        # Add static tools
        for tool in tool_registry.list_enabled():
            descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  必要なサービストークン: {tool.required_service_token or 'なし'}"
            )

        # Add dynamic tool definitions
        for definition in tool_registry.list_enabled_definitions():
            descriptions.append(
                f"- {definition.name}: {definition.description}\n"
                f"  カテゴリ: {definition.category}\n"
                f"  必要なサービストークン: {definition.required_service_token or 'なし'}"
            )

        return "\n".join(descriptions) if descriptions else "ツールが登録されていません"

    def get_template_agents_description(self) -> str:
        """Get description of template (pre-defined) agents."""
        registry = get_agent_registry()
        descriptions = []

        # Add static agents
        for agent in registry.list_enabled():
            capabilities = ", ".join(agent.capabilities)
            tools = ", ".join([t.name for t in agent.tools])
            descriptions.append(
                f"- {agent.name}: {agent.description}\n"
                f"  能力: {capabilities}\n"
                f"  ツール: {tools}"
            )

        # Add dynamic agents (also considered templates)
        for definition in registry.list_enabled_definitions():
            capabilities = ", ".join(definition.capabilities)
            tools = ", ".join(definition.tools)
            descriptions.append(
                f"- {definition.name}: {definition.description}\n"
                f"  能力: {capabilities}\n"
                f"  ツール: {tools}"
            )

        return "\n".join(descriptions) if descriptions else "テンプレートエージェントなし"

    def get_available_agents_description(self) -> str:
        """Get description of available agents (for simple mode)."""
        registry = get_agent_registry()

        descriptions = []

        # Add static agents
        for agent in registry.list_enabled():
            capabilities = ", ".join(agent.capabilities)
            descriptions.append(
                f"- {agent.name}: {agent.description}\n  能力: {capabilities}"
            )

        # Add dynamic agents
        for definition in registry.list_enabled_definitions():
            capabilities = ", ".join(definition.capabilities)
            descriptions.append(
                f"- {definition.name}: {definition.description}\n  能力: {capabilities}"
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
        if self.dynamic_mode:
            return await self._create_dynamic_plan(state)
        else:
            return await self._create_simple_plan(state)

    async def _create_dynamic_plan(self, state: AgentState) -> ExecutionPlan:
        """Create a plan with dynamic ad-hoc agent generation."""
        user_input = state["user_input"]

        # Build system prompt with caching for better performance
        # The tool and agent descriptions are cached since they change infrequently
        system_prompt_content = TOOL_ANALYSIS_PROMPT.format(
            available_tools=self.get_available_tools_description(),
            template_agents=self.get_template_agents_description(),
        )

        # Check if we should use prompt caching based on model support
        # Get model_id from LLM instance if available
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with prompt caching enabled on system message if supported
        # This caches the tool/agent descriptions which are static across requests
        if use_caching:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(
                    content=[
                        {"type": "text", "text": system_prompt_content},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                ),
                HumanMessage(content=f"ユーザーのリクエスト: {user_input}"),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=f"ユーザーのリクエスト: {user_input}"),
            ])

        try:
            # Get LLM response
            chain = prompt | self.llm | self.output_parser
            result = await chain.ainvoke({})

            # Parse result into ExecutionPlan
            plan = self._parse_dynamic_plan_result(result)

            logger.info(
                "dynamic_plan_created",
                session_id=state["session_id"],
                num_tasks=len(plan.tasks),
                num_adhoc=sum(1 for t in plan.tasks if t.is_adhoc),
                num_parallel_groups=len(plan.parallel_groups),
                analysis=result.get("analysis", {}),
            )

            return plan

        except Exception as e:
            logger.error(
                "dynamic_plan_creation_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            # Fall back to simple mode
            return await self._create_simple_plan(state)

    async def _create_simple_plan(self, state: AgentState) -> ExecutionPlan:
        """Create a simple plan using pre-defined agents only."""
        user_input = state["user_input"]

        # Build system prompt with caching
        system_prompt_content = SIMPLE_PLANNER_PROMPT.format(
            available_agents=self.get_available_agents_description()
        )

        # Check if we should use prompt caching based on model support
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with prompt caching enabled if supported
        if use_caching:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(
                    content=[
                        {"type": "text", "text": system_prompt_content},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                ),
                HumanMessage(content=f"ユーザーの質問: {user_input}"),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=f"ユーザーの質問: {user_input}"),
            ])

        try:
            # Get LLM response
            chain = prompt | self.llm | self.output_parser
            result = await chain.ainvoke({})

            # Parse result into ExecutionPlan
            plan = self._parse_simple_plan_result(result)

            logger.info(
                "simple_plan_created",
                session_id=state["session_id"],
                num_tasks=len(plan.tasks),
                num_parallel_groups=len(plan.parallel_groups),
            )

            return plan

        except Exception as e:
            logger.error(
                "simple_plan_creation_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            # Return a default plan
            return self._create_default_plan(user_input)

    def _parse_dynamic_plan_result(self, result: dict) -> ExecutionPlan:
        """Parse LLM result into ExecutionPlan with ad-hoc agents."""
        tasks = []

        for i, task_data in enumerate(result.get("tasks", [])):
            task_type = task_data.get("type", "template")

            if task_type == "adhoc":
                # Create ad-hoc agent specification
                adhoc_spec = AdHocAgentSpec(
                    id=f"adhoc_{uuid4().hex[:8]}",
                    name=task_data.get("name", f"adhoc_agent_{i}"),
                    purpose=task_data.get("purpose", ""),
                    tools=task_data.get("tools", []),
                    expected_output=task_data.get("expected_output", ""),
                    reasoning=task_data.get("reasoning", ""),
                )

                task = Task(
                    id=f"task_{i}",
                    adhoc_spec=adhoc_spec,
                    priority=task_data.get("priority", 0),
                    parameters=task_data.get("parameters", {}),
                )
            else:
                # Template/pre-defined agent
                task = Task(
                    id=f"task_{i}",
                    agent_name=task_data.get("agent_name"),
                    priority=task_data.get("priority", 0),
                    parameters=task_data.get("parameters", {}),
                )

            tasks.append(task)

        # Parse parallel groups
        parallel_groups = self._parse_parallel_groups(result, tasks)

        # Build execution order
        execution_order = self._build_execution_order(tasks, parallel_groups)

        return ExecutionPlan(
            tasks=tasks,
            parallel_groups=parallel_groups,
            execution_order=execution_order,
            current_phase="planned",
        )

    def _parse_simple_plan_result(self, result: dict) -> ExecutionPlan:
        """Parse simple LLM result into ExecutionPlan."""
        tasks = []
        for i, task_data in enumerate(result.get("tasks", [])):
            task = Task(
                id=f"task_{i}",
                agent_name=task_data["agent_name"],
                priority=task_data.get("priority", 0),
                parameters=task_data.get("parameters", {}),
            )
            tasks.append(task)

        parallel_groups = self._parse_parallel_groups(result, tasks)
        execution_order = self._build_execution_order(tasks, parallel_groups)

        return ExecutionPlan(
            tasks=tasks,
            parallel_groups=parallel_groups,
            execution_order=execution_order,
            current_phase="planned",
        )

    def _parse_parallel_groups(
        self,
        result: dict,
        tasks: list[Task],
    ) -> list[ParallelGroup]:
        """Parse parallel groups from result."""
        parallel_groups = []

        for group_data in result.get("parallel_groups", []):
            task_indices = group_data.get("task_indices", [])
            task_ids = [tasks[i].id for i in task_indices if i < len(tasks)]

            if task_ids:
                group = ParallelGroup(
                    task_ids=task_ids,
                    timeout_seconds=group_data.get("timeout_seconds", 30),
                )
                parallel_groups.append(group)

        return parallel_groups

    def _build_execution_order(
        self,
        tasks: list[Task],
        parallel_groups: list[ParallelGroup],
    ) -> list[str]:
        """Build execution order considering parallel groups."""
        execution_order = []
        grouped_task_ids = set()

        for group in parallel_groups:
            execution_order.append(group.group_id)
            grouped_task_ids.update(group.task_ids)

        # Add ungrouped tasks
        for task in tasks:
            if task.id not in grouped_task_ids:
                execution_order.append(task.id)

        return execution_order

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
        tasks_summary = []
        for task in plan.tasks:
            if task.is_adhoc:
                tasks_summary.append({
                    "type": "adhoc",
                    "name": task.adhoc_spec.name,
                    "purpose": task.adhoc_spec.purpose,
                    "tools": task.adhoc_spec.tools,
                })
            else:
                tasks_summary.append({
                    "type": "template",
                    "name": task.agent_name,
                })

        parallel_groups_summary = [
            {
                "group_id": group.group_id,
                "tasks": [
                    next(
                        (t.effective_agent_name for t in plan.tasks if t.id == tid),
                        tid
                    )
                    for tid in group.task_ids
                ],
                "timeout_seconds": group.timeout_seconds,
            }
            for group in plan.parallel_groups
        ]

        return {
            "tasks": tasks_summary,
            "parallel_groups": parallel_groups_summary,
            "estimated_steps": len(plan.execution_order),
            "has_adhoc_agents": any(t.is_adhoc for t in plan.tasks),
        }
