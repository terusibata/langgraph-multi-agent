"""LangGraph workflow definition for multi-agent system with dynamic agents."""

from typing import Any, AsyncIterator

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
import structlog

from src.agents.state import AgentState, create_initial_state, RequestContext, Task
from src.agents.main_agent import MainAgent
from src.agents.registry import (
    get_agent_registry,
    initialize_registries,
    is_dynamic_mode,
)
from src.services.llm import get_llm
from src.services.execution import ParallelExecutor
from src.services.execution.parallel import get_next_tasks_to_execute
from src.services.streaming import SSEManager
from src.services.thread import get_thread_manager
from src.services.error import get_error_handler, AgentError
from src.config import get_settings
from src.models.base import get_session_factory
from src.repositories.execution import ExecutionRepository

logger = structlog.get_logger()


class MultiAgentGraph:
    """
    LangGraph-based multi-agent workflow with dynamic agent support.

    Orchestrates the MainAgent and SubAgents (both pre-defined and ad-hoc)
    through a state machine. Uses database for execution result persistence.
    """

    def __init__(
        self,
        model_id: str | None = None,
        dynamic_mode: bool | None = None,
        _skip_async_init: bool = False,
    ):
        """
        Initialize the graph.

        Args:
            model_id: Optional model ID for MainAgent
            dynamic_mode: Override dynamic mode setting (None uses config)
            _skip_async_init: Skip async initialization (for internal use)
        """
        self.settings = get_settings()
        self.model_id = model_id or self.settings.default_model_id
        self._initialized = False
        self._dynamic_mode_override = dynamic_mode

        # Initialize components
        self.llm = get_llm(self.model_id)
        self.executor = ParallelExecutor()
        self.thread_manager = get_thread_manager()
        self.error_handler = get_error_handler()

        # Build the graph
        self.graph = self._build_graph()

        if not _skip_async_init:
            # Dynamic mode will be determined at runtime
            self.dynamic_mode = dynamic_mode if dynamic_mode is not None else True
            self.main_agent = MainAgent(self.llm, self.model_id, self.dynamic_mode)

    async def initialize(self) -> None:
        """Initialize async components (registries, dynamic mode from DB)."""
        if self._initialized:
            return

        # Initialize registries
        await initialize_registries()

        # Determine dynamic mode from database if not overridden
        if self._dynamic_mode_override is None:
            self.dynamic_mode = await is_dynamic_mode()
        else:
            self.dynamic_mode = self._dynamic_mode_override

        # Reinitialize main agent with correct dynamic mode
        self.main_agent = MainAgent(self.llm, self.model_id, self.dynamic_mode)

        self._initialized = True

        logger.info(
            "multi_agent_graph_initialized",
            model_id=self.model_id,
            dynamic_mode=self.dynamic_mode,
        )

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        # Create state graph
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute_agents", self._execute_agents_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("synthesize", self._synthesize_node)

        # Set entry point using START constant
        graph.add_edge(START, "plan")

        # Add conditional edges
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "execute_agents": "execute_agents",
                "synthesize": "synthesize",
            },
        )

        graph.add_conditional_edges(
            "execute_agents",
            self._route_after_execution,
            {
                "evaluate": "evaluate",
                "execute_agents": "execute_agents",
                "synthesize": "synthesize",
            },
        )

        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "execute_agents": "execute_agents",
                "synthesize": "synthesize",
            },
        )

        # Synthesize always ends
        graph.add_edge("synthesize", END)

        return graph

    async def _plan_node(self, state: AgentState) -> AgentState:
        """Planning node - creates execution plan with dynamic agents."""
        logger.info(
            "node_plan",
            session_id=state["session_id"],
            dynamic_mode=self.dynamic_mode,
        )

        # Add user message to history
        state["messages"].append(
            HumanMessage(content=state["user_input"])
        )

        # Create execution plan (may include ad-hoc agents)
        state = await self.main_agent.plan(state)

        # Log plan summary
        plan = state["execution_plan"]
        adhoc_count = sum(1 for t in plan.tasks if t.is_adhoc)
        logger.info(
            "plan_created",
            session_id=state["session_id"],
            total_tasks=len(plan.tasks),
            adhoc_agents=adhoc_count,
            template_agents=len(plan.tasks) - adhoc_count,
            parallel_groups=len(plan.parallel_groups),
        )

        # Save execution plan to database
        await self._save_execution_plan(state)

        return state

    async def _execute_agents_node(self, state: AgentState) -> AgentState:
        """Execute SubAgents node - supports both pre-defined and ad-hoc agents."""
        logger.info("node_execute_agents", session_id=state["session_id"])

        # Get completed agents
        completed_agents = set(state["sub_agent_results"].keys())

        # Get next tasks to execute
        tasks_to_run = get_next_tasks_to_execute(
            state["execution_plan"],
            completed_agents,
        )

        if not tasks_to_run:
            logger.warning(
                "no_tasks_to_execute",
                session_id=state["session_id"],
                completed=list(completed_agents),
            )
            return state

        # Update phase
        state["execution_plan"].current_phase = "executing"

        # Get timeout
        timeout = self.main_agent.get_execution_timeout(
            state,
            [t.effective_agent_name for t in tasks_to_run],
        )

        # Log execution info
        task_info = [
            {
                "name": t.effective_agent_name,
                "is_adhoc": t.is_adhoc,
                "tools": t.adhoc_spec.tools if t.is_adhoc else None,
            }
            for t in tasks_to_run
        ]

        logger.info(
            "executing_tasks",
            session_id=state["session_id"],
            tasks=task_info,
            parallel=len(tasks_to_run) > 1,
            timeout=timeout,
        )

        # Execute tasks (supports both pre-defined and ad-hoc agents)
        results = await self.executor.execute_tasks(
            state,
            tasks_to_run,
            timeout,
        )

        # Store results and save to database
        for agent_name, result in results.items():
            state["sub_agent_results"][agent_name] = result

            # Update metrics
            state["metrics"].tool_call_count += 1

            # Find task for this agent
            task = self._find_task_by_agent(state, agent_name)

            # Save execution result to database
            await self._save_execution_result(state, agent_name, result, task)

        return state

    async def _evaluate_node(self, state: AgentState) -> AgentState:
        """Evaluation node - assesses results."""
        logger.info("node_evaluate", session_id=state["session_id"])

        state["execution_plan"].current_phase = "evaluating"
        state = await self.main_agent.evaluate(state)

        return state

    async def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesis node - generates final response."""
        logger.info("node_synthesize", session_id=state["session_id"])

        # Check if direct tool mode is enabled
        if state.get("direct_tool_mode", False) and not state.get("fast_response", False):
            # Use direct tool executor
            logger.info("using_direct_tool_executor", session_id=state["session_id"])
            response = await self.main_agent.direct_tool_executor.execute(state)
            state["final_response"] = response
        else:
            # Normal synthesis (includes fast response mode handling)
            state = await self.main_agent.synthesize(state)

        # Finalize metrics
        state["metrics"].finalize()

        # Complete execution session in database
        await self._complete_execution_session(state)

        return state

    def _route_after_plan(self, state: AgentState) -> str:
        """Route after planning."""
        # Fast response mode: skip execution and go directly to synthesis
        if state.get("fast_response", False):
            logger.info("fast_response_mode_activated", session_id=state["session_id"])
            return "synthesize"

        # Direct tool mode: skip plan execution and use direct tool mode in synthesize
        if state.get("direct_tool_mode", False):
            logger.info("direct_tool_mode_activated", session_id=state["session_id"])
            return "synthesize"

        # Normal mode: check if there are tasks to execute
        if not state["execution_plan"].tasks:
            return "synthesize"
        return "execute_agents"

    def _route_after_execution(self, state: AgentState) -> str:
        """Route after agent execution."""
        # Get completed agents
        completed_agents = set(state["sub_agent_results"].keys())

        # Check if more tasks to execute
        remaining_tasks = get_next_tasks_to_execute(
            state["execution_plan"],
            completed_agents,
        )

        if remaining_tasks:
            return "execute_agents"

        # Check if evaluation is needed
        if self.main_agent.router.should_evaluate(state):
            return "evaluate"

        return "synthesize"

    def _route_after_evaluation(self, state: AgentState) -> str:
        """Route after evaluation."""
        evaluation = state["intermediate_evaluation"]

        if evaluation.next_action == "catalog":
            # Add catalog task if not already executed
            if "catalog" not in state["sub_agent_results"]:
                # Add catalog to execution plan
                state["execution_plan"].tasks.append(
                    Task(
                        id="catalog_task",
                        agent_name="catalog",
                        parameters={"query": state["user_input"]},
                    )
                )
                state["execution_plan"].execution_order.append("catalog_task")
                return "execute_agents"

        return "synthesize"

    def compile(self, checkpointer: Any = None):
        """
        Compile the graph for execution.

        Args:
            checkpointer: Optional checkpointer for persistence

        Returns:
            Compiled graph
        """
        return self.graph.compile(checkpointer=checkpointer)

    async def _create_execution_session(
        self,
        state: AgentState,
        tenant_id: str,
    ) -> None:
        """Create execution session in database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ExecutionRepository(session)
            await repo.create_session({
                "session_id": state["session_id"],
                "thread_id": state["thread_id"],
                "tenant_id": tenant_id,
                "user_input": state["user_input"],
                "request_context": {
                    "tenant_id": state["request_context"].tenant_id,
                    "user_id": state["request_context"].user_id,
                    "permissions": state["request_context"].permissions,
                },
            })

    async def _save_execution_plan(self, state: AgentState) -> None:
        """Save execution plan to database."""
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ExecutionRepository(session)
                plan = state["execution_plan"]
                plan_dict = {
                    "tasks": [
                        {
                            "id": t.id,
                            "agent_name": t.agent_name,
                            "is_adhoc": t.is_adhoc,
                            "adhoc_spec": t.adhoc_spec.model_dump() if t.adhoc_spec else None,
                            "priority": t.priority,
                            "depends_on": t.depends_on,
                            "parameters": t.parameters,
                            "status": t.status,
                        }
                        for t in plan.tasks
                    ],
                    "parallel_groups": [g.model_dump() for g in plan.parallel_groups],
                    "execution_order": plan.execution_order,
                    "current_phase": plan.current_phase,
                }
                await repo.update_session_plan(state["session_id"], plan_dict)
        except Exception as e:
            logger.error(
                "failed_to_save_execution_plan",
                session_id=state["session_id"],
                error=str(e),
                exc_info=True,  # Include stack trace
            )
            # Continue execution even if plan save fails

    async def _save_execution_result(
        self,
        state: AgentState,
        agent_name: str,
        result: Any,
        task: Task | None,
    ) -> None:
        """Save execution result to database."""
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ExecutionRepository(session)
                result_data = {
                    "result_type": "agent",
                    "agent_name": agent_name,
                    "is_adhoc": task.is_adhoc if task else False,
                    "adhoc_spec": task.adhoc_spec.model_dump() if task and task.adhoc_spec else None,
                    "task_id": task.id if task else None,
                    "status": result.status,
                    "data": result.data if hasattr(result, "data") else None,
                    "error": result.error if hasattr(result, "error") else None,
                    "retry_count": result.retry_count if hasattr(result, "retry_count") else 0,
                    "search_variations": result.search_variations if hasattr(result, "search_variations") else [],
                    "started_at": result.started_at if hasattr(result, "started_at") else None,
                    "completed_at": result.completed_at if hasattr(result, "completed_at") else None,
                    "duration_ms": result.duration_ms if hasattr(result, "duration_ms") else 0,
                }
                await repo.add_result(state["session_id"], result_data)
        except Exception as e:
            logger.error(
                "failed_to_save_execution_result",
                session_id=state["session_id"],
                agent_name=agent_name,
                error=str(e),
                exc_info=True,  # Include stack trace
            )
            # Results are preserved in state["sub_agent_results"]

    async def _complete_execution_session(self, state: AgentState) -> None:
        """Complete execution session in database."""
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ExecutionRepository(session)
                metrics = state["metrics"]
                await repo.complete_session(
                    state["session_id"],
                    state.get("final_response"),
                    {
                        "total_input_tokens": metrics.total_input_tokens,
                        "total_output_tokens": metrics.total_output_tokens,
                        "total_cost_usd": metrics.total_cost_usd,
                        "llm_call_count": metrics.llm_call_count,
                        "tool_call_count": metrics.tool_call_count,
                        "llm_calls": [
                            {
                                "call_id": c.call_id,
                                "model_id": c.model_id,
                                "agent": c.agent,
                                "phase": c.phase,
                                "input_tokens": c.input_tokens,
                                "output_tokens": c.output_tokens,
                                "cost_usd": c.cost_usd,
                            }
                            for c in metrics.llm_calls
                        ],
                    },
                )
        except Exception as e:
            logger.error(
                "failed_to_complete_execution_session",
                session_id=state["session_id"],
                error=str(e),
            )

    async def run(
        self,
        user_input: str,
        request_context: RequestContext,
        thread_id: str | None = None,
        checkpointer: Any = None,
        fast_response: bool = False,
        direct_tool_mode: bool = False,
        response_format: str = "text",
        response_schema: dict | None = None,
    ) -> AgentState:
        """
        Run the graph synchronously.

        Args:
            user_input: User's input message
            request_context: Request context with auth info
            thread_id: Optional thread ID for continuation
            checkpointer: Optional checkpointer
            fast_response: Enable fast response mode (no sub-agents or tools)
            direct_tool_mode: Enable direct tool mode (MainAgent uses tools directly)
            response_format: Response format ("text" or "json")
            response_schema: JSON schema for structured output

        Returns:
            Final AgentState
        """
        # Ensure initialized
        await self.initialize()

        # Check thread status
        if thread_id:
            can_send, error_msg = await self.thread_manager.check_thread_status(thread_id)
            if not can_send:
                raise AgentError("THREAD_001", detail=error_msg)

        # Create initial state
        state = create_initial_state(
            user_input=user_input,
            request_context=request_context,
            thread_id=thread_id,
            fast_response=fast_response,
            direct_tool_mode=direct_tool_mode,
            response_format=response_format,
            response_schema=response_schema,
        )

        # Create execution session
        await self._create_execution_session(state, request_context.tenant_id)

        # Compile and run graph
        compiled = self.compile(checkpointer)
        config = {"configurable": {"thread_id": state["thread_id"]}}

        final_state = await compiled.ainvoke(state, config)

        # Update thread metrics
        await self.thread_manager.update_thread_metrics(
            final_state["thread_id"],
            final_state["metrics"].total_input_tokens,
            final_state["metrics"].total_output_tokens,
            final_state["metrics"].total_cost_usd,
        )

        return final_state

    async def stream(
        self,
        user_input: str,
        request_context: RequestContext,
        sse_manager: SSEManager,
        thread_id: str | None = None,
        checkpointer: Any = None,
        fast_response: bool = False,
        direct_tool_mode: bool = False,
        response_format: str = "text",
        response_schema: dict | None = None,
    ) -> AsyncIterator[dict]:
        """
        Run the graph with SSE streaming.

        Args:
            user_input: User's input message
            request_context: Request context
            sse_manager: SSE manager for events
            thread_id: Optional thread ID
            checkpointer: Optional checkpointer
            fast_response: Enable fast response mode (no sub-agents or tools)
            direct_tool_mode: Enable direct tool mode (MainAgent uses tools directly)
            response_format: Response format ("text" or "json")
            response_schema: JSON schema for structured output

        Yields:
            Event dicts for sse-starlette
        """
        try:
            # Ensure initialized
            await self.initialize()

            # Check thread status
            if thread_id:
                can_send, error_msg = await self.thread_manager.check_thread_status(thread_id)
                if not can_send:
                    raise AgentError("THREAD_001", detail=error_msg)

            # Create initial state
            state = create_initial_state(
                user_input=user_input,
                request_context=request_context,
                thread_id=thread_id,
                fast_response=fast_response,
                direct_tool_mode=direct_tool_mode,
                response_format=response_format,
                response_schema=response_schema,
            )

            # Create execution session
            await self._create_execution_session(state, request_context.tenant_id)

            # Emit session start
            await sse_manager.emit_session_start()

            # Compile graph
            compiled = self.compile(checkpointer)
            config = {"configurable": {"thread_id": state["thread_id"]}}

            # Stream execution
            async for event in compiled.astream(state, config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    await self._emit_node_events(
                        node_name,
                        node_state,
                        sse_manager,
                    )

            # Get final state
            final_state = await compiled.aget_state(config)

            # Update thread and emit complete
            thread_state = await self.thread_manager.update_thread_metrics(
                final_state.values["thread_id"],
                final_state.values["metrics"].total_input_tokens,
                final_state.values["metrics"].total_output_tokens,
                final_state.values["metrics"].total_cost_usd,
            )

            await self._emit_session_complete(
                final_state.values,
                thread_state,
                sse_manager,
            )

        except AgentError as e:
            await sse_manager.emit_error(
                self.error_handler.create_error_response(e)
            )
        except Exception as e:
            error = self.error_handler.handle_exception(e)
            await sse_manager.emit_error(
                self.error_handler.create_error_response(error)
            )
        finally:
            await sse_manager.close()

        # Yield all events
        async for event in sse_manager.events():
            yield event

    async def _emit_node_events(
        self,
        node_name: str,
        state: AgentState,
        sse_manager: SSEManager,
    ) -> None:
        """Emit SSE events for a node execution."""
        if node_name == "plan":
            plan_summary = self.main_agent.get_plan_summary(state)
            await sse_manager.emit_plan_created(plan_summary)

        elif node_name == "execute_agents":
            for agent_name, result in state.get("sub_agent_results", {}).items():
                # Include ad-hoc agent info if available
                task = self._find_task_by_agent(state, agent_name)
                agent_info = None
                if task and task.is_adhoc:
                    agent_info = {
                        "type": "adhoc",
                        "purpose": task.adhoc_spec.purpose,
                        "tools": task.adhoc_spec.tools,
                    }

                await sse_manager.emit_agent_end(
                    agent_name=agent_name,
                    status=result.status,
                    duration_ms=result.duration_ms,
                    metadata=agent_info,
                )

        elif node_name == "evaluate":
            evaluation = self.main_agent.get_evaluation_summary(state)
            await sse_manager.emit_evaluation(evaluation)

        elif node_name == "synthesize":
            # For direct tool mode or fast response, we need to stream differently
            if state.get("direct_tool_mode", False) and not state.get("fast_response", False):
                # Stream using direct tool executor
                async for token in self.main_agent.direct_tool_executor.execute_stream(state):
                    await sse_manager.emit_token(token)
                await sse_manager.emit_token("", finish_reason="stop")
            elif state.get("final_response"):
                # Stream the final response as tokens
                response = state["final_response"]
                for i in range(0, len(response), 50):
                    chunk = response[i:i+50]
                    await sse_manager.emit_token(chunk)
                await sse_manager.emit_token("", finish_reason="stop")

    def _find_task_by_agent(self, state: AgentState, agent_name: str) -> Task | None:
        """Find a task by agent name."""
        for task in state["execution_plan"].tasks:
            if task.effective_agent_name == agent_name:
                return task
        return None

    async def _emit_session_complete(
        self,
        state: AgentState,
        thread_state: Any,
        sse_manager: SSEManager,
    ) -> None:
        """Emit session complete event."""
        metrics = state["metrics"]

        # Build agents executed summary with ad-hoc info
        agents_executed = []
        for name, result in state.get("sub_agent_results", {}).items():
            task = self._find_task_by_agent(state, name)
            agent_info = {
                "name": name,
                "status": result.status,
                "retries": result.retry_count,
                "search_variations": result.search_variations,
                "duration_ms": result.duration_ms,
            }
            if task and task.is_adhoc:
                agent_info["type"] = "adhoc"
                agent_info["purpose"] = task.adhoc_spec.purpose
                agent_info["tools"] = task.adhoc_spec.tools
            else:
                agent_info["type"] = "template"
            agents_executed.append(agent_info)

        complete_data = {
            "response": {
                "content": state.get("final_response", ""),
                "finish_reason": "stop",
            },
            "execution_summary": {
                "plan": self.main_agent.get_plan_summary(state),
                "agents_executed": agents_executed,
                "tools_executed": [
                    {
                        "tool": tr.tool_name,
                        "agent": tr.agent_name,
                        "success": tr.success,
                    }
                    for tr in state.get("tool_results", [])
                ],
            },
            "metrics": {
                "duration_ms": metrics.duration_ms,
                "llm_calls": [
                    {
                        "call_id": call.call_id,
                        "model_id": call.model_id,
                        "agent": call.agent,
                        "phase": call.phase,
                        "input_tokens": call.input_tokens,
                        "output_tokens": call.output_tokens,
                        "cost_usd": call.cost_usd,
                    }
                    for call in metrics.llm_calls
                ],
                "totals": {
                    "input_tokens": metrics.total_input_tokens,
                    "output_tokens": metrics.total_output_tokens,
                    "total_tokens": metrics.total_input_tokens + metrics.total_output_tokens,
                    "total_cost_usd": metrics.total_cost_usd,
                    "llm_call_count": metrics.llm_call_count,
                    "tool_call_count": metrics.tool_call_count,
                },
            },
            "thread_state": {
                "status": thread_state.status,
                "context_tokens_used": thread_state.context_tokens_used,
                "context_max_tokens": thread_state.context_max_tokens,
                "context_usage_percent": thread_state.context_usage_percent,
                "message_count": thread_state.message_count,
                "thread_total_tokens": thread_state.thread_total_tokens,
                "thread_total_cost_usd": thread_state.thread_total_cost_usd,
            },
        }

        await sse_manager.emit_session_complete(complete_data)


async def create_graph(
    model_id: str | None = None,
    dynamic_mode: bool | None = None,
) -> MultiAgentGraph:
    """
    Create a new multi-agent graph with async initialization.

    Args:
        model_id: Optional model ID for MainAgent
        dynamic_mode: Override dynamic mode setting

    Returns:
        MultiAgentGraph instance
    """
    graph = MultiAgentGraph(model_id, dynamic_mode, _skip_async_init=True)
    await graph.initialize()
    return graph
