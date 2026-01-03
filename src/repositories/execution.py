"""Execution repository for database operations."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog

from src.models.execution import ExecutionSessionModel, ExecutionResultModel

logger = structlog.get_logger()


class ExecutionRepository:
    """Repository for execution session CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    # ==========================================================================
    # Execution Session operations
    # ==========================================================================

    async def create_session(
        self,
        session_data: dict[str, Any],
    ) -> ExecutionSessionModel:
        """Create a new execution session."""
        exec_session = ExecutionSessionModel(
            session_id=session_data["session_id"],
            thread_id=session_data["thread_id"],
            tenant_id=session_data["tenant_id"],
            user_input=session_data["user_input"],
            status="pending",
            execution_plan=session_data.get("execution_plan", {}),
            request_context=session_data.get("request_context", {}),
            started_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.session.add(exec_session)
        await self.session.commit()
        await self.session.refresh(exec_session)
        logger.info(
            "execution_session_created",
            session_id=exec_session.session_id,
            thread_id=exec_session.thread_id,
        )
        return exec_session

    async def get_session(
        self,
        session_id: str,
        include_results: bool = True,
    ) -> ExecutionSessionModel | None:
        """Get an execution session by ID."""
        query = select(ExecutionSessionModel).where(
            ExecutionSessionModel.session_id == session_id
        )
        if include_results:
            query = query.options(selectinload(ExecutionSessionModel.results))
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> ExecutionSessionModel | None:
        """Update an execution session."""
        exec_session = await self.get_session(session_id, include_results=False)
        if not exec_session:
            return None

        for key, value in updates.items():
            if hasattr(exec_session, key) and value is not None:
                setattr(exec_session, key, value)
        exec_session.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(exec_session)
        return exec_session

    async def complete_session(
        self,
        session_id: str,
        final_response: str | None,
        metrics: dict[str, Any],
        error: dict[str, Any] | None = None,
    ) -> ExecutionSessionModel | None:
        """Mark a session as completed."""
        exec_session = await self.get_session(session_id, include_results=False)
        if not exec_session:
            return None

        exec_session.status = "failed" if error else "completed"
        exec_session.final_response = final_response
        exec_session.completed_at = datetime.now(timezone.utc)

        # Calculate duration
        if exec_session.started_at:
            delta = exec_session.completed_at - exec_session.started_at
            exec_session.duration_ms = int(delta.total_seconds() * 1000)

        # Update metrics
        exec_session.total_input_tokens = metrics.get("total_input_tokens", 0)
        exec_session.total_output_tokens = metrics.get("total_output_tokens", 0)
        exec_session.total_cost_usd = metrics.get("total_cost_usd", 0.0)
        exec_session.llm_call_count = metrics.get("llm_call_count", 0)
        exec_session.tool_call_count = metrics.get("tool_call_count", 0)
        exec_session.llm_calls = metrics.get("llm_calls", [])

        # Error info
        if error:
            exec_session.error_code = error.get("code")
            exec_session.error_message = error.get("message")

        exec_session.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(exec_session)
        logger.info(
            "execution_session_completed",
            session_id=session_id,
            status=exec_session.status,
            duration_ms=exec_session.duration_ms,
        )
        return exec_session

    async def update_session_plan(
        self,
        session_id: str,
        execution_plan: dict[str, Any],
    ) -> ExecutionSessionModel | None:
        """Update the execution plan for a session."""
        return await self.update_session(
            session_id,
            {"execution_plan": execution_plan, "status": "running"},
        )

    async def list_sessions_by_thread(
        self,
        thread_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExecutionSessionModel]:
        """List execution sessions for a thread."""
        query = (
            select(ExecutionSessionModel)
            .where(ExecutionSessionModel.thread_id == thread_id)
            .order_by(desc(ExecutionSessionModel.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def list_sessions_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExecutionSessionModel]:
        """List execution sessions for a tenant."""
        query = select(ExecutionSessionModel).where(
            ExecutionSessionModel.tenant_id == tenant_id
        )
        if status:
            query = query.where(ExecutionSessionModel.status == status)
        query = query.order_by(desc(ExecutionSessionModel.created_at))
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def delete_session(self, session_id: str) -> bool:
        """Delete an execution session and its results."""
        result = await self.session.execute(
            delete(ExecutionSessionModel).where(
                ExecutionSessionModel.session_id == session_id
            )
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("execution_session_deleted", session_id=session_id)
        return deleted

    # ==========================================================================
    # Execution Result operations
    # ==========================================================================

    async def add_result(
        self,
        session_id: str,
        result_data: dict[str, Any],
    ) -> ExecutionResultModel:
        """Add an execution result to a session."""
        result = ExecutionResultModel(
            session_id=session_id,
            result_type=result_data.get("result_type", "agent"),
            agent_name=result_data["agent_name"],
            tool_name=result_data.get("tool_name"),
            is_adhoc=result_data.get("is_adhoc", False),
            adhoc_spec=result_data.get("adhoc_spec"),
            task_id=result_data.get("task_id"),
            status=result_data["status"],
            data=result_data.get("data"),
            error=result_data.get("error"),
            retry_count=result_data.get("retry_count", 0),
            search_variations=result_data.get("search_variations", []),
            started_at=result_data.get("started_at"),
            completed_at=result_data.get("completed_at"),
            duration_ms=result_data.get("duration_ms", 0),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.session.add(result)
        await self.session.commit()
        await self.session.refresh(result)
        logger.debug(
            "execution_result_added",
            session_id=session_id,
            agent_name=result.agent_name,
            status=result.status,
        )
        return result

    async def add_results_batch(
        self,
        session_id: str,
        results_data: list[dict[str, Any]],
    ) -> list[ExecutionResultModel]:
        """Add multiple execution results to a session."""
        results = []
        for data in results_data:
            result = ExecutionResultModel(
                session_id=session_id,
                result_type=data.get("result_type", "agent"),
                agent_name=data["agent_name"],
                tool_name=data.get("tool_name"),
                is_adhoc=data.get("is_adhoc", False),
                adhoc_spec=data.get("adhoc_spec"),
                task_id=data.get("task_id"),
                status=data["status"],
                data=data.get("data"),
                error=data.get("error"),
                retry_count=data.get("retry_count", 0),
                search_variations=data.get("search_variations", []),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                duration_ms=data.get("duration_ms", 0),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            results.append(result)
            self.session.add(result)

        await self.session.commit()
        for result in results:
            await self.session.refresh(result)
        return results

    async def get_results_for_session(
        self,
        session_id: str,
    ) -> list[ExecutionResultModel]:
        """Get all execution results for a session."""
        query = (
            select(ExecutionResultModel)
            .where(ExecutionResultModel.session_id == session_id)
            .order_by(ExecutionResultModel.id)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_results_by_agent(
        self,
        session_id: str,
        agent_name: str,
    ) -> list[ExecutionResultModel]:
        """Get execution results for a specific agent."""
        query = (
            select(ExecutionResultModel)
            .where(
                ExecutionResultModel.session_id == session_id,
                ExecutionResultModel.agent_name == agent_name,
            )
            .order_by(ExecutionResultModel.id)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
