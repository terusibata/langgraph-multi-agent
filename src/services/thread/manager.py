"""Thread management service with database persistence."""

from datetime import datetime, timezone
from typing import Literal, Any
from uuid import uuid4

import structlog

from src.agents.state import ThreadState
from src.config import get_settings
from src.models.base import get_session_factory
from src.repositories.thread import ThreadRepository

logger = structlog.get_logger()


class ThreadInfo:
    """Information about a conversation thread."""

    def __init__(
        self,
        thread_id: str,
        tenant_id: str,
        status: Literal["active", "warning", "locked"] = "active",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        total_tokens_used: int = 0,
        total_cost_usd: float = 0.0,
        message_count: int = 0,
        context_tokens_used: int = 0,
        metadata: dict | None = None,
    ):
        self.thread_id = thread_id
        self.tenant_id = tenant_id
        self.status = status
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.total_tokens_used = total_tokens_used
        self.total_cost_usd = total_cost_usd
        self.message_count = message_count
        self.context_tokens_used = context_tokens_used
        self.metadata = metadata or {}

    @classmethod
    def from_model(cls, model: Any) -> "ThreadInfo":
        """Create ThreadInfo from database model."""
        return cls(
            thread_id=model.thread_id,
            tenant_id=model.tenant_id,
            status=model.status,
            created_at=model.created_at,
            updated_at=model.updated_at,
            total_tokens_used=model.total_tokens_used,
            total_cost_usd=model.total_cost_usd,
            message_count=model.message_count,
            context_tokens_used=model.context_tokens_used,
            metadata=model.metadata_,
        )


class ThreadManager:
    """
    Manager for conversation threads with database persistence.

    Handles thread lifecycle, status tracking, and context management.
    All data is stored in PostgreSQL database.
    """

    def __init__(self):
        """Initialize the manager."""
        self.settings = get_settings()

    async def create_thread(self, tenant_id: str, metadata: dict | None = None) -> str:
        """
        Create a new thread in database.

        Args:
            tenant_id: Tenant identifier
            metadata: Optional metadata

        Returns:
            New thread ID
        """
        thread_id = f"thread_{uuid4().hex[:12]}"

        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ThreadRepository(session)
            await repo.create_thread(thread_id, tenant_id, metadata)

        logger.info(
            "thread_created",
            thread_id=thread_id,
            tenant_id=tenant_id,
        )

        return thread_id

    async def get_thread(self, thread_id: str) -> ThreadInfo | None:
        """
        Get thread information from database.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadInfo or None if not found
        """
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ThreadRepository(session)
            model = await repo.get_thread(thread_id)
            if model:
                return ThreadInfo.from_model(model)
        return None

    async def get_thread_state(self, thread_id: str) -> ThreadState | None:
        """
        Get thread state for agent processing.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadState or None
        """
        thread = await self.get_thread(thread_id)
        if not thread:
            return None

        return ThreadState(
            status=thread.status,
            context_tokens_used=thread.context_tokens_used,
            context_max_tokens=self.settings.context_max_tokens,
            message_count=thread.message_count,
            thread_total_tokens=thread.total_tokens_used,
            thread_total_cost_usd=thread.total_cost_usd,
        )

    async def check_thread_status(
        self,
        thread_id: str,
    ) -> tuple[bool, str | None]:
        """
        Check if thread can accept new messages.

        Args:
            thread_id: Thread identifier

        Returns:
            Tuple of (can_send, error_message)
        """
        thread = await self.get_thread(thread_id)

        if not thread:
            return True, None  # New thread

        if thread.status == "locked":
            return False, "スレッドの上限に達しました。新しいスレッドを作成してください。"

        return True, None

    async def update_thread_metrics(
        self,
        thread_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> ThreadState:
        """
        Update thread metrics after processing.

        Args:
            thread_id: Thread identifier
            input_tokens: Input tokens used
            output_tokens: Output tokens used
            cost_usd: Cost in USD

        Returns:
            Updated ThreadState
        """
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ThreadRepository(session)
            thread = await repo.get_thread(thread_id)

            if not thread:
                # Create new thread record
                thread = await repo.create_thread(thread_id, "unknown")

            # Update metrics
            new_total = thread.total_tokens_used + input_tokens + output_tokens
            new_cost = thread.total_cost_usd + cost_usd
            new_count = thread.message_count + 1
            new_context = thread.context_tokens_used + input_tokens + output_tokens

            # Calculate usage and determine status
            usage_percent = (new_context / self.settings.context_max_tokens) * 100

            if usage_percent >= self.settings.context_lock_threshold:
                new_status = "locked"
            elif usage_percent >= self.settings.context_warning_threshold:
                new_status = "warning"
            else:
                new_status = "active"

            # Update in database
            await repo.update_thread(
                thread_id,
                {
                    "total_tokens_used": new_total,
                    "total_cost_usd": new_cost,
                    "message_count": new_count,
                    "context_tokens_used": new_context,
                    "status": new_status,
                },
            )

        logger.debug(
            "thread_metrics_updated",
            thread_id=thread_id,
            total_tokens=new_total,
            context_usage=f"{usage_percent:.1f}%",
            status=new_status,
        )

        return ThreadState(
            status=new_status,
            context_tokens_used=new_context,
            context_max_tokens=self.settings.context_max_tokens,
            message_count=new_count,
            thread_total_tokens=new_total,
            thread_total_cost_usd=new_cost,
        )

    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread from database.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False if not found
        """
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ThreadRepository(session)
            deleted = await repo.delete_thread(thread_id)

        if deleted:
            logger.info("thread_deleted", thread_id=thread_id)
        return deleted

    async def list_threads(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ThreadInfo]:
        """
        List threads for a tenant from database.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of ThreadInfo
        """
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ThreadRepository(session)
            threads = await repo.list_threads(tenant_id, status, limit)
            return [ThreadInfo.from_model(t) for t in threads]

    async def get_or_create_thread(
        self,
        thread_id: str | None,
        tenant_id: str,
    ) -> tuple[str, bool]:
        """
        Get existing thread or create new one.

        Args:
            thread_id: Optional existing thread ID
            tenant_id: Tenant identifier

        Returns:
            Tuple of (thread_id, is_new)
        """
        if thread_id:
            thread = await self.get_thread(thread_id)
            if thread:
                return thread_id, False

        # Create new thread
        new_thread_id = await self.create_thread(tenant_id)
        return new_thread_id, True


# Global thread manager
_thread_manager: ThreadManager | None = None


def get_thread_manager() -> ThreadManager:
    """Get the global thread manager."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


def reset_thread_manager() -> None:
    """Reset thread manager (useful for testing)."""
    global _thread_manager
    _thread_manager = None
