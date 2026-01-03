"""Repository for managing dynamic agent definitions in the database."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from src.agents.registry import AgentDefinition
from src.db.connection import get_db_pool


class AgentRepository:
    """Repository for managing agent definitions in PostgreSQL."""

    @staticmethod
    async def create(definition: AgentDefinition) -> AgentDefinition:
        """Create a new agent definition in the database."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO dynamic_agents (
                    id, name, description, capabilities, tools,
                    executor, retry_strategy, priority, enabled,
                    metadata, created_at, updated_at, created_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                uuid.UUID(definition.id),
                definition.name,
                definition.description,
                json.dumps(definition.capabilities),
                json.dumps(definition.tools),
                json.dumps(definition.executor),
                json.dumps(definition.retry_strategy),
                definition.priority,
                definition.enabled,
                json.dumps(definition.metadata),
                definition.created_at,
                definition.updated_at,
                definition.created_by,
            )
        return definition

    @staticmethod
    async def get_by_name(name: str) -> AgentDefinition | None:
        """Get an agent definition by name."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, description, capabilities, tools,
                       executor, retry_strategy, priority, enabled,
                       metadata, created_at, updated_at, created_by
                FROM dynamic_agents
                WHERE name = $1
                """,
                name,
            )
            if row:
                return AgentRepository._row_to_definition(row)
            return None

    @staticmethod
    async def get_all(enabled_only: bool = False) -> list[AgentDefinition]:
        """Get all agent definitions."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT id, name, description, capabilities, tools,
                       executor, retry_strategy, priority, enabled,
                       metadata, created_at, updated_at, created_by
                FROM dynamic_agents
            """
            if enabled_only:
                query += " WHERE enabled = TRUE"
            query += " ORDER BY priority DESC, name"

            rows = await conn.fetch(query)
            return [AgentRepository._row_to_definition(row) for row in rows]

    @staticmethod
    async def update(definition: AgentDefinition) -> AgentDefinition:
        """Update an agent definition."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            definition.updated_at = datetime.now(timezone.utc)
            await conn.execute(
                """
                UPDATE dynamic_agents
                SET description = $2, capabilities = $3, tools = $4,
                    executor = $5, retry_strategy = $6, priority = $7,
                    enabled = $8, metadata = $9, updated_at = $10
                WHERE name = $1
                """,
                definition.name,
                definition.description,
                json.dumps(definition.capabilities),
                json.dumps(definition.tools),
                json.dumps(definition.executor),
                json.dumps(definition.retry_strategy),
                definition.priority,
                definition.enabled,
                json.dumps(definition.metadata),
                definition.updated_at,
            )
        return definition

    @staticmethod
    async def delete(name: str) -> bool:
        """Delete an agent definition."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM dynamic_agents WHERE name = $1",
                name,
            )
            # Check if any rows were deleted
            return result.split()[-1] != "0"

    @staticmethod
    async def exists(name: str) -> bool:
        """Check if an agent definition exists."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM dynamic_agents WHERE name = $1",
                name,
            )
            return row is not None

    @staticmethod
    def _row_to_definition(row: asyncpg.Record) -> AgentDefinition:
        """Convert a database row to an AgentDefinition object."""
        return AgentDefinition(
            id=str(row["id"]),
            name=row["name"],
            description=row["description"],
            capabilities=json.loads(row["capabilities"]) if isinstance(row["capabilities"], str) else row["capabilities"],
            tools=json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"],
            executor=json.loads(row["executor"]) if isinstance(row["executor"], str) else row["executor"],
            retry_strategy=json.loads(row["retry_strategy"]) if isinstance(row["retry_strategy"], str) else row["retry_strategy"],
            priority=row["priority"],
            enabled=row["enabled"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )
