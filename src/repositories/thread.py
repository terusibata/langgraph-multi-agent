"""Thread repository for database operations."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, delete, desc
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.models.thread import ThreadModel

logger = structlog.get_logger()


class ThreadRepository:
    """Repository for thread CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def create_thread(
        self,
        thread_id: str,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ThreadModel:
        """Create a new thread."""
        thread = ThreadModel(
            thread_id=thread_id,
            tenant_id=tenant_id,
            status="active",
            total_tokens_used=0,
            total_cost_usd=0.0,
            message_count=0,
            context_tokens_used=0,
            metadata_=metadata or {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self.session.add(thread)
        await self.session.commit()
        await self.session.refresh(thread)
        logger.info("thread_created", thread_id=thread_id, tenant_id=tenant_id)
        return thread

    async def get_thread(self, thread_id: str) -> ThreadModel | None:
        """Get a thread by ID."""
        result = await self.session.execute(
            select(ThreadModel).where(ThreadModel.thread_id == thread_id)
        )
        return result.scalar_one_or_none()

    async def update_thread(
        self,
        thread_id: str,
        updates: dict[str, Any],
    ) -> ThreadModel | None:
        """Update a thread."""
        thread = await self.get_thread(thread_id)
        if not thread:
            return None

        for key, value in updates.items():
            if hasattr(thread, key) and value is not None:
                if key == "metadata":
                    setattr(thread, "metadata_", value)
                else:
                    setattr(thread, key, value)
        thread.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def update_thread_metrics(
        self,
        thread_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> ThreadModel | None:
        """Update thread metrics after processing."""
        thread = await self.get_thread(thread_id)
        if not thread:
            return None

        thread.total_tokens_used += input_tokens + output_tokens
        thread.total_cost_usd += cost_usd
        thread.message_count += 1
        thread.context_tokens_used += input_tokens + output_tokens
        thread.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def update_thread_status(
        self,
        thread_id: str,
        status: str,
    ) -> ThreadModel | None:
        """Update thread status."""
        thread = await self.get_thread(thread_id)
        if not thread:
            return None

        thread.status = status
        thread.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(thread)
        return thread

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread."""
        result = await self.session.execute(
            delete(ThreadModel).where(ThreadModel.thread_id == thread_id)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("thread_deleted", thread_id=thread_id)
        return deleted

    async def list_threads(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ThreadModel]:
        """List threads for a tenant."""
        query = select(ThreadModel).where(ThreadModel.tenant_id == tenant_id)
        if status:
            query = query.where(ThreadModel.status == status)
        query = query.order_by(desc(ThreadModel.updated_at))
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists."""
        result = await self.session.execute(
            select(ThreadModel.thread_id).where(ThreadModel.thread_id == thread_id)
        )
        return result.scalar_one_or_none() is not None

    async def get_or_create_thread(
        self,
        thread_id: str,
        tenant_id: str,
    ) -> tuple[ThreadModel, bool]:
        """Get an existing thread or create a new one.

        Returns:
            Tuple of (thread, created) where created is True if new.
        """
        thread = await self.get_thread(thread_id)
        if thread:
            return thread, False
        thread = await self.create_thread(thread_id, tenant_id)
        return thread, True

    async def count_threads_by_tenant(
        self,
        tenant_id: str,
        status: str | None = None,
    ) -> int:
        """Count threads for a tenant."""
        from sqlalchemy import func
        query = select(func.count()).select_from(ThreadModel).where(
            ThreadModel.tenant_id == tenant_id
        )
        if status:
            query = query.where(ThreadModel.status == status)
        result = await self.session.execute(query)
        return result.scalar() or 0
