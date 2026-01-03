"""Initial schema for database storage.

Revision ID: 001
Revises:
Create Date: 2025-01-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create agents table
    op.create_table(
        "agents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False, default=list),
        sa.Column("tools", sa.JSON(), nullable=False, default=list),
        sa.Column("executor", sa.JSON(), nullable=False),
        sa.Column("retry_strategy", sa.JSON(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, default=0),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True, index=True),
        sa.Column("metadata", sa.JSON(), nullable=False, default=dict),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create template_agents table
    op.create_table(
        "template_agents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("purpose", sa.Text(), nullable=False, default=""),
        sa.Column("capabilities", sa.JSON(), nullable=False, default=list),
        sa.Column("tools", sa.JSON(), nullable=False, default=list),
        sa.Column("parallel_execution", sa.Boolean(), nullable=False, default=False),
        sa.Column("expected_output", sa.Text(), nullable=False, default=""),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True, index=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create tools table
    op.create_table(
        "tools",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("category", sa.String(64), nullable=False, default="general", index=True),
        sa.Column("parameters", sa.JSON(), nullable=False, default=list),
        sa.Column("executor", sa.JSON(), nullable=False, default=dict),
        sa.Column("required_service_token", sa.String(64), nullable=True),
        sa.Column("timeout_seconds", sa.Integer(), nullable=False, default=30),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True, index=True),
        sa.Column("metadata", sa.JSON(), nullable=False, default=dict),
        sa.Column("created_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create threads table
    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("status", sa.String(16), nullable=False, default="active", index=True),
        sa.Column("total_tokens_used", sa.Integer(), nullable=False, default=0),
        sa.Column("total_cost_usd", sa.Float(), nullable=False, default=0.0),
        sa.Column("message_count", sa.Integer(), nullable=False, default=0),
        sa.Column("context_tokens_used", sa.Integer(), nullable=False, default=0),
        sa.Column("metadata", sa.JSON(), nullable=False, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create execution_sessions table
    op.create_table(
        "execution_sessions",
        sa.Column("session_id", sa.String(64), primary_key=True),
        sa.Column(
            "thread_id",
            sa.String(64),
            sa.ForeignKey("threads.thread_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("tenant_id", sa.String(255), nullable=False, index=True),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("final_response", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, default="pending", index=True),
        sa.Column("execution_plan", sa.JSON(), nullable=False, default=dict),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, default=0),
        sa.Column("total_input_tokens", sa.Integer(), nullable=False, default=0),
        sa.Column("total_output_tokens", sa.Integer(), nullable=False, default=0),
        sa.Column("total_cost_usd", sa.Float(), nullable=False, default=0.0),
        sa.Column("llm_call_count", sa.Integer(), nullable=False, default=0),
        sa.Column("tool_call_count", sa.Integer(), nullable=False, default=0),
        sa.Column("llm_calls", sa.JSON(), nullable=False, default=list),
        sa.Column("error_code", sa.String(32), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("request_context", sa.JSON(), nullable=False, default=dict),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create execution_results table
    op.create_table(
        "execution_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            sa.String(64),
            sa.ForeignKey("execution_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("result_type", sa.String(16), nullable=False, index=True),
        sa.Column("agent_name", sa.String(255), nullable=False, index=True),
        sa.Column("tool_name", sa.String(255), nullable=True),
        sa.Column("is_adhoc", sa.Boolean(), nullable=False, default=False),
        sa.Column("adhoc_spec", sa.JSON(), nullable=True),
        sa.Column("task_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, default=0),
        sa.Column("search_variations", sa.JSON(), nullable=False, default=list),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Create system_config table
    op.create_table(
        "system_config",
        sa.Column("key", sa.String(255), primary_key=True),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # Insert default configuration
    op.execute(
        """
        INSERT INTO system_config (key, value, description, created_at, updated_at)
        VALUES
        ('planning', '{"dynamic_mode": true, "prefer_templates": true}', 'Planning mode configuration', NOW(), NOW()),
        ('execution', '{"parallel_timeout_seconds": 30, "sub_agent_timeout_seconds": 60, "tool_timeout_seconds": 30, "max_adhoc_agents": 5, "max_tools_per_adhoc_agent": 4}', 'Execution settings', NOW(), NOW())
        """
    )


def downgrade() -> None:
    op.drop_table("execution_results")
    op.drop_table("execution_sessions")
    op.drop_table("system_config")
    op.drop_table("threads")
    op.drop_table("tools")
    op.drop_table("template_agents")
    op.drop_table("agents")
