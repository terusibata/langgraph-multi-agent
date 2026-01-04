-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for LangGraph checkpoints (if needed)
-- Note: langgraph-checkpoint-postgres will create its own tables automatically

-- All application tables are managed by Alembic migrations
-- See alembic/versions/ for schema definitions
