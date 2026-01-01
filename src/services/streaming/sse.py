"""SSE (Server-Sent Events) streaming service."""

import asyncio
from typing import AsyncIterator, Any
from datetime import datetime

from sse_starlette.sse import EventSourceResponse
import structlog

from src.api.schemas.events import (
    BaseEvent,
    SessionStartEvent,
    PlanCreatedEvent,
    AgentStartEvent,
    AgentRetryEvent,
    AgentEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    EvaluationEvent,
    TokenEvent,
    LLMMetricsEvent,
    SessionCompleteEvent,
    ErrorEvent,
    ExecutionPlanSummary,
    EvaluationResult,
    LLMCallMetrics,
    SessionCompleteData,
    ErrorEventData,
    ErrorData,
    ResponseData,
    ExecutionSummary,
    AgentExecutionSummary,
    ToolExecutionSummary,
    SessionMetrics,
    TotalMetrics,
    ThreadStateData,
)

logger = structlog.get_logger()


class SSEManager:
    """
    Manager for SSE event streaming.

    Provides methods to emit various event types during agent execution.
    """

    def __init__(self, session_id: str, thread_id: str):
        """
        Initialize the manager.

        Args:
            session_id: Current session ID
            thread_id: Current thread ID
        """
        self.session_id = session_id
        self.thread_id = thread_id
        self._queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        self._closed = False

    async def emit(self, event: BaseEvent) -> None:
        """
        Emit an event to the stream.

        Args:
            event: Event to emit
        """
        if not self._closed:
            await self._queue.put(event)

    async def close(self) -> None:
        """Close the stream."""
        self._closed = True
        await self._queue.put(None)

    async def events(self) -> AsyncIterator[dict]:
        """
        Iterate over events as dicts for sse-starlette.

        Yields:
            Event dicts with event, data, and id keys
        """
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event.to_sse_dict()

    # Event emission helpers

    async def emit_session_start(self) -> None:
        """Emit session start event."""
        await self.emit(SessionStartEvent(
            session_id=self.session_id,
            thread_id=self.thread_id,
        ))

    async def emit_plan_created(self, plan_summary: dict) -> None:
        """Emit plan created event."""
        await self.emit(PlanCreatedEvent(
            session_id=self.session_id,
            plan=ExecutionPlanSummary(
                initial_agents=plan_summary.get("initial_agents", []),
                parallel_groups=plan_summary.get("parallel_groups", []),
                estimated_steps=plan_summary.get("estimated_steps", 0),
            ),
        ))

    async def emit_agent_start(
        self,
        agent_name: str,
        task_description: str | None = None,
    ) -> None:
        """Emit agent start event."""
        await self.emit(AgentStartEvent(
            session_id=self.session_id,
            agent_name=agent_name,
            task_description=task_description,
        ))

    async def emit_agent_retry(
        self,
        agent_name: str,
        attempt: int,
        reason: str,
        modified_query: str | None = None,
    ) -> None:
        """Emit agent retry event."""
        await self.emit(AgentRetryEvent(
            session_id=self.session_id,
            agent_name=agent_name,
            attempt=attempt,
            reason=reason,
            modified_query=modified_query,
        ))

    async def emit_agent_end(
        self,
        agent_name: str,
        status: str,
        duration_ms: int,
        result_summary: str | None = None,
    ) -> None:
        """Emit agent end event."""
        await self.emit(AgentEndEvent(
            session_id=self.session_id,
            agent_name=agent_name,
            status=status,
            duration_ms=duration_ms,
            result_summary=result_summary,
        ))

    async def emit_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        parameters_summary: dict | None = None,
    ) -> None:
        """Emit tool call event."""
        await self.emit(ToolCallEvent(
            session_id=self.session_id,
            tool_name=tool_name,
            agent_name=agent_name,
            parameters_summary=parameters_summary,
        ))

    async def emit_tool_result(
        self,
        tool_name: str,
        agent_name: str,
        success: bool,
        result_summary: str | None = None,
    ) -> None:
        """Emit tool result event."""
        await self.emit(ToolResultEvent(
            session_id=self.session_id,
            tool_name=tool_name,
            agent_name=agent_name,
            success=success,
            result_summary=result_summary,
        ))

    async def emit_evaluation(self, evaluation: dict) -> None:
        """Emit evaluation event."""
        await self.emit(EvaluationEvent(
            session_id=self.session_id,
            evaluation=EvaluationResult(
                has_sufficient_info=evaluation.get("has_sufficient_info", False),
                missing_info=evaluation.get("missing_info", []),
                next_action=evaluation.get("next_action", "respond"),
                reasoning=evaluation.get("reasoning"),
            ),
        ))

    async def emit_token(
        self,
        content: str,
        finish_reason: str | None = None,
    ) -> None:
        """Emit token event."""
        await self.emit(TokenEvent(
            session_id=self.session_id,
            content=content,
            finish_reason=finish_reason,
        ))

    async def emit_llm_metrics(self, metrics: dict) -> None:
        """Emit LLM metrics event."""
        await self.emit(LLMMetricsEvent(
            session_id=self.session_id,
            metrics=LLMCallMetrics(
                call_id=metrics.get("call_id", ""),
                model_id=metrics.get("model_id", ""),
                agent=metrics.get("agent", ""),
                phase=metrics.get("phase"),
                input_tokens=metrics.get("input_tokens", 0),
                output_tokens=metrics.get("output_tokens", 0),
                cost_usd=metrics.get("cost_usd", 0.0),
            ),
        ))

    async def emit_session_complete(self, data: dict) -> None:
        """Emit session complete event."""
        await self.emit(SessionCompleteEvent(
            data=SessionCompleteData(
                session_id=self.session_id,
                thread_id=self.thread_id,
                response=ResponseData(
                    content=data.get("response", {}).get("content", ""),
                    finish_reason=data.get("response", {}).get("finish_reason", "stop"),
                ),
                execution_summary=ExecutionSummary(
                    plan=data.get("execution_summary", {}).get("plan", {}),
                    agents_executed=[
                        AgentExecutionSummary(**agent)
                        for agent in data.get("execution_summary", {}).get("agents_executed", [])
                    ],
                    tools_executed=[
                        ToolExecutionSummary(**tool)
                        for tool in data.get("execution_summary", {}).get("tools_executed", [])
                    ],
                ),
                metrics=SessionMetrics(
                    duration_ms=data.get("metrics", {}).get("duration_ms", 0),
                    llm_calls=[
                        LLMCallMetrics(**call)
                        for call in data.get("metrics", {}).get("llm_calls", [])
                    ],
                    totals=TotalMetrics(**data.get("metrics", {}).get("totals", {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "total_cost_usd": 0.0,
                        "llm_call_count": 0,
                        "tool_call_count": 0,
                    })),
                ),
                thread_state=ThreadStateData(**data.get("thread_state", {
                    "status": "active",
                    "context_tokens_used": 0,
                    "context_max_tokens": 200000,
                    "context_usage_percent": 0.0,
                    "message_count": 0,
                    "thread_total_tokens": 0,
                    "thread_total_cost_usd": 0.0,
                })),
            ),
        ))

    async def emit_error(self, error_data: dict) -> None:
        """Emit error event."""
        await self.emit(ErrorEvent(
            data=ErrorEventData(
                session_id=self.session_id,
                thread_id=self.thread_id,
                error=ErrorData(**error_data.get("error", {})),
                partial_results=error_data.get("partial_results"),
                partial_metrics=error_data.get("partial_metrics"),
                thread_state=error_data.get("thread_state"),
            ),
        ))


def create_sse_response(
    event_generator: AsyncIterator[dict],
) -> EventSourceResponse:
    """
    Create an SSE response from an event generator.

    Args:
        event_generator: Async iterator yielding event dicts

    Returns:
        EventSourceResponse
    """
    async def generate():
        async for event in event_generator:
            yield event

    return EventSourceResponse(generate())
