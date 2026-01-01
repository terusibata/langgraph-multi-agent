"""Thread management service."""

from datetime import datetime
from typing import Literal
from uuid import uuid4

import structlog

from src.agents.state import ThreadState
from src.config import get_settings

logger = structlog.get_logger()


class ThreadInfo:
    """Information about a conversation thread."""

    def __init__(
        self,
        thread_id: str,
        tenant_id: str,
        status: Literal["active", "warning", "locked"] = "active",
    ):
        self.thread_id = thread_id
        self.tenant_id = tenant_id
        self.status = status
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.message_count = 0
        self.context_tokens_used = 0
        self.metadata: dict = {}


class ThreadManager:
    """
    Manager for conversation threads.

    Handles thread lifecycle, status tracking, and context management.
    """

    def __init__(self):
        """Initialize the manager."""
        self.settings = get_settings()
        # In-memory storage for development
        # In production, this would use database
        self._threads: dict[str, ThreadInfo] = {}

    async def create_thread(self, tenant_id: str) -> str:
        """
        Create a new thread.

        Args:
            tenant_id: Tenant identifier

        Returns:
            New thread ID
        """
        thread_id = f"thread_{uuid4().hex[:12]}"
        self._threads[thread_id] = ThreadInfo(
            thread_id=thread_id,
            tenant_id=tenant_id,
        )

        logger.info(
            "thread_created",
            thread_id=thread_id,
            tenant_id=tenant_id,
        )

        return thread_id

    async def get_thread(self, thread_id: str) -> ThreadInfo | None:
        """
        Get thread information.

        Args:
            thread_id: Thread identifier

        Returns:
            ThreadInfo or None if not found
        """
        return self._threads.get(thread_id)

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
        thread = await self.get_thread(thread_id)

        if not thread:
            # Create new thread record
            thread = ThreadInfo(
                thread_id=thread_id,
                tenant_id="unknown",
            )
            self._threads[thread_id] = thread

        # Update metrics
        thread.total_tokens_used += input_tokens + output_tokens
        thread.total_cost_usd += cost_usd
        thread.message_count += 1
        thread.context_tokens_used += input_tokens + output_tokens
        thread.updated_at = datetime.utcnow()

        # Update status based on context usage
        usage_percent = (
            thread.context_tokens_used / self.settings.context_max_tokens
        ) * 100

        if usage_percent >= self.settings.context_lock_threshold:
            thread.status = "locked"
        elif usage_percent >= self.settings.context_warning_threshold:
            thread.status = "warning"
        else:
            thread.status = "active"

        logger.debug(
            "thread_metrics_updated",
            thread_id=thread_id,
            total_tokens=thread.total_tokens_used,
            context_usage=f"{usage_percent:.1f}%",
            status=thread.status,
        )

        return ThreadState(
            status=thread.status,
            context_tokens_used=thread.context_tokens_used,
            context_max_tokens=self.settings.context_max_tokens,
            message_count=thread.message_count,
            thread_total_tokens=thread.total_tokens_used,
            thread_total_cost_usd=thread.total_cost_usd,
        )

    async def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False if not found
        """
        if thread_id in self._threads:
            del self._threads[thread_id]
            logger.info("thread_deleted", thread_id=thread_id)
            return True
        return False

    async def list_threads(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ThreadInfo]:
        """
        List threads for a tenant.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of ThreadInfo
        """
        threads = [
            t for t in self._threads.values()
            if t.tenant_id == tenant_id
            and (status is None or t.status == status)
        ]

        # Sort by updated_at descending
        threads.sort(key=lambda t: t.updated_at, reverse=True)

        return threads[:limit]


# Global thread manager
_thread_manager: ThreadManager | None = None


def get_thread_manager() -> ThreadManager:
    """Get the global thread manager."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager
