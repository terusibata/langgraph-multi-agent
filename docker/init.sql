-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for LangGraph checkpoints (if needed)
-- Note: langgraph-checkpoint-postgres will create its own tables automatically

-- Create custom tables for thread management
CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_tokens_used BIGINT DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    context_tokens_used BIGINT DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index for tenant lookup
CREATE INDEX IF NOT EXISTS idx_threads_tenant_id ON threads(tenant_id);
CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
CREATE INDEX IF NOT EXISTS idx_threads_created_at ON threads(created_at DESC);

-- Create session metrics table
CREATE TABLE IF NOT EXISTS session_metrics (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id UUID REFERENCES threads(thread_id) ON DELETE CASCADE,
    tenant_id VARCHAR(255) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost_usd DECIMAL(10, 6) DEFAULT 0,
    llm_call_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    agents_executed JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(50) DEFAULT 'in_progress',
    error_code VARCHAR(50),
    error_message TEXT
);

-- Create indexes for session metrics
CREATE INDEX IF NOT EXISTS idx_session_metrics_thread_id ON session_metrics(thread_id);
CREATE INDEX IF NOT EXISTS idx_session_metrics_tenant_id ON session_metrics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_session_metrics_started_at ON session_metrics(started_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for threads table
DROP TRIGGER IF EXISTS update_threads_updated_at ON threads;
CREATE TRIGGER update_threads_updated_at
    BEFORE UPDATE ON threads
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create dynamic tools table
CREATE TABLE IF NOT EXISTS dynamic_tools (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'general',
    parameters JSONB DEFAULT '[]'::jsonb,
    executor JSONB DEFAULT '{}'::jsonb,
    required_service_token VARCHAR(255),
    timeout_seconds INTEGER DEFAULT 30,
    enabled BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- Create indexes for dynamic_tools
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_name ON dynamic_tools(name);
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_category ON dynamic_tools(category);
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_enabled ON dynamic_tools(enabled);
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_created_at ON dynamic_tools(created_at DESC);

-- Trigger for dynamic_tools table
DROP TRIGGER IF EXISTS update_dynamic_tools_updated_at ON dynamic_tools;
CREATE TRIGGER update_dynamic_tools_updated_at
    BEFORE UPDATE ON dynamic_tools
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create dynamic agents table
CREATE TABLE IF NOT EXISTS dynamic_agents (
    id UUID PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    capabilities JSONB DEFAULT '[]'::jsonb,
    tools JSONB DEFAULT '[]'::jsonb,
    executor JSONB DEFAULT '{"type": "llm"}'::jsonb,
    retry_strategy JSONB DEFAULT '{"max_attempts": 3, "retry_conditions": ["no_results"], "query_modification": "synonym", "backoff_seconds": 0.5}'::jsonb,
    priority INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

-- Create indexes for dynamic_agents
CREATE INDEX IF NOT EXISTS idx_dynamic_agents_name ON dynamic_agents(name);
CREATE INDEX IF NOT EXISTS idx_dynamic_agents_enabled ON dynamic_agents(enabled);
CREATE INDEX IF NOT EXISTS idx_dynamic_agents_priority ON dynamic_agents(priority DESC);
CREATE INDEX IF NOT EXISTS idx_dynamic_agents_created_at ON dynamic_agents(created_at DESC);

-- Trigger for dynamic_agents table
DROP TRIGGER IF EXISTS update_dynamic_agents_updated_at ON dynamic_agents;
CREATE TRIGGER update_dynamic_agents_updated_at
    BEFORE UPDATE ON dynamic_agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
