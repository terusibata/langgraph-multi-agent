"""Base class for SubAgents."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field
import structlog

from src.agents.state import AgentState, SubAgentResult
from src.agents.tools.base import ToolBase
from src.config import get_settings

logger = structlog.get_logger()


class RetryStrategy(BaseModel):
    """Configuration for retry behavior."""

    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    retry_conditions: list[str] = Field(
        default_factory=lambda: ["no_results"],
        description="Conditions that trigger retry",
    )
    query_modification: Literal["synonym", "broader", "narrower", "llm_rewrite", "none"] = Field(
        default="synonym",
        description="Query modification strategy",
    )
    backoff_seconds: float = Field(default=0.5, description="Backoff between retries")


class SubAgentBase(ABC):
    """
    Base class for all SubAgents.

    SubAgents are specialized agents that handle specific tasks within
    the multi-agent system. They can use tools, implement retry logic,
    and report their execution status.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: list[str],
        tools: list[ToolBase] | None = None,
        model_id: str | None = None,
        retry_strategy: RetryStrategy | None = None,
    ):
        """
        Initialize the SubAgent.

        Args:
            name: Unique identifier for the agent
            description: Description of what the agent does
            capabilities: List of capability tags
            tools: List of tools available to this agent
            model_id: Model ID to use (defaults to settings)
            retry_strategy: Retry configuration
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.tools = tools or []
        self.model_id = model_id or get_settings().sub_agent_model_id
        self.retry_strategy = retry_strategy or RetryStrategy()
        self._tool_map = {tool.name: tool for tool in self.tools}

    def get_tool(self, name: str) -> ToolBase | None:
        """Get a tool by name."""
        return self._tool_map.get(name)

    @abstractmethod
    async def execute(self, state: AgentState, task_params: dict | None = None) -> SubAgentResult:
        """
        Execute the agent's task.

        Args:
            state: Current agent state
            task_params: Optional parameters for the task

        Returns:
            SubAgentResult with execution status and data
        """
        pass

    async def execute_with_retry(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute the agent with retry logic.

        Args:
            state: Current agent state
            task_params: Optional parameters for the task

        Returns:
            SubAgentResult with execution status and data
        """
        started_at = datetime.now(timezone.utc)
        attempt = 0
        last_result: SubAgentResult | None = None
        search_variations: list[str] = []

        while attempt < self.retry_strategy.max_attempts:
            attempt += 1

            # Modify query if retrying
            modified_params = task_params.copy() if task_params else {}
            if attempt > 1 and "query" in modified_params:
                original_query = modified_params["query"]
                modified_query = await self.modify_query(original_query, attempt)
                modified_params["query"] = modified_query
                search_variations.append(modified_query)

                logger.info(
                    "agent_retry",
                    agent=self.name,
                    attempt=attempt,
                    original_query=original_query,
                    modified_query=modified_query,
                )

            try:
                result = await self.execute(state, modified_params)
                result.retry_count = attempt - 1
                result.search_variations = search_variations
                result.started_at = started_at
                result.completed_at = datetime.now(timezone.utc)
                result.duration_ms = int(
                    (result.completed_at - started_at).total_seconds() * 1000
                )

                # Check if retry is needed
                if not self.should_retry(result):
                    return result

                last_result = result

            except Exception as e:
                logger.error(
                    "agent_execution_error",
                    agent=self.name,
                    attempt=attempt,
                    error=str(e),
                )
                last_result = SubAgentResult(
                    agent_name=self.name,
                    status="failed",
                    error=str(e),
                    retry_count=attempt - 1,
                    search_variations=search_variations,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

                if attempt >= self.retry_strategy.max_attempts:
                    break

            # Wait before retry
            if self.retry_strategy.backoff_seconds > 0:
                import asyncio
                await asyncio.sleep(self.retry_strategy.backoff_seconds)

        # Return last result if all retries exhausted
        if last_result:
            last_result.duration_ms = int(
                (datetime.now(timezone.utc) - started_at).total_seconds() * 1000
            )
            return last_result

        return SubAgentResult(
            agent_name=self.name,
            status="failed",
            error="All retry attempts exhausted",
            retry_count=attempt,
            search_variations=search_variations,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )

    def should_retry(self, result: SubAgentResult) -> bool:
        """
        Determine if a retry is needed based on the result.

        Args:
            result: The execution result to evaluate

        Returns:
            True if retry should be attempted
        """
        if result.status == "success":
            return False

        # Check each retry condition
        conditions = self.retry_strategy.retry_conditions

        if "no_results" in conditions and self._is_no_results(result):
            return True

        if "insufficient_results" in conditions and self._is_insufficient_results(result):
            return True

        if "low_relevance" in conditions and self._is_low_relevance(result):
            return True

        return False

    def _is_no_results(self, result: SubAgentResult) -> bool:
        """Check if result has no data."""
        if result.data is None:
            return True
        if isinstance(result.data, list) and len(result.data) == 0:
            return True
        if isinstance(result.data, dict) and not result.data:
            return True
        return False

    def _is_insufficient_results(self, result: SubAgentResult) -> bool:
        """Check if result has insufficient data."""
        if result.data is None:
            return True
        if isinstance(result.data, list) and len(result.data) < 2:
            return True
        return False

    def _is_low_relevance(self, result: SubAgentResult) -> bool:
        """Check if result has low relevance scores."""
        if not isinstance(result.data, list):
            return False

        # Check for relevance scores in results
        for item in result.data:
            if isinstance(item, dict):
                score = item.get("score", item.get("relevance", 1.0))
                if score >= 0.7:  # At least one high-relevance result
                    return False
        return True

    async def modify_query(self, original_query: str, attempt: int) -> str:
        """
        Modify the query for retry.

        Args:
            original_query: The original search query
            attempt: Current attempt number

        Returns:
            Modified query string
        """
        strategy = self.retry_strategy.query_modification

        if strategy == "none":
            return original_query

        if strategy == "synonym":
            return self._apply_synonym_modification(original_query, attempt)

        if strategy == "broader":
            return self._apply_broader_modification(original_query, attempt)

        if strategy == "narrower":
            return self._apply_narrower_modification(original_query, attempt)

        if strategy == "llm_rewrite":
            return await self._apply_llm_rewrite(original_query, attempt)

        return original_query

    def _apply_synonym_modification(self, query: str, attempt: int) -> str:
        """Apply synonym-based modification."""
        # Simple word variations
        replacements = {
            "エラー": ["障害", "不具合", "問題"],
            "接続": ["つながらない", "通信", "アクセス"],
            "できない": ["失敗", "不可", "NG"],
        }

        result = query
        for original, synonyms in replacements.items():
            if original in query:
                # Use different synonym based on attempt
                idx = (attempt - 2) % len(synonyms)
                result = result.replace(original, synonyms[idx])
                break

        return result if result != query else f"{query} 対処法"

    def _apply_broader_modification(self, query: str, attempt: int) -> str:
        """Apply broader search modification."""
        # Remove specific terms to broaden search
        words = query.split()
        if len(words) > 2:
            # Remove the last word for broader search
            return " ".join(words[:-1])
        return query

    def _apply_narrower_modification(self, query: str, attempt: int) -> str:
        """Apply narrower search modification."""
        # Add more specific terms
        suffixes = ["解決方法", "原因", "手順"]
        idx = (attempt - 2) % len(suffixes)
        return f"{query} {suffixes[idx]}"

    async def _apply_llm_rewrite(self, original_query: str, attempt: int) -> str:
        """
        Use LLM to rewrite the query.

        This is a placeholder - actual implementation would call the LLM.
        """
        # For now, fall back to synonym modification
        return self._apply_synonym_modification(original_query, attempt)

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        )

        return f"""あなたは{self.description}を担当する専門エージェントです。

あなたの能力:
{', '.join(self.capabilities)}

利用可能なツール:
{tool_descriptions}

指示:
1. ユーザーの質問に関連する情報を検索してください
2. 検索結果が不十分な場合は、異なる検索ワードで再試行を提案してください
3. 結果を簡潔にまとめて報告してください
"""

    def format_result_for_main_agent(self, result: SubAgentResult) -> str:
        """
        Format the result for the MainAgent to process.

        Args:
            result: The SubAgent's execution result

        Returns:
            Formatted string representation
        """
        if result.status == "failed":
            return f"[{self.name}] エラー: {result.error}"

        if result.status == "partial":
            return f"[{self.name}] 部分的な結果: {self._format_data(result.data)}"

        return f"[{self.name}] 検索結果:\n{self._format_data(result.data)}"

    def _format_data(self, data: Any) -> str:
        """Format data for display."""
        if data is None:
            return "結果なし"

        if isinstance(data, str):
            return data

        if isinstance(data, list):
            if len(data) == 0:
                return "結果なし"
            items = []
            for i, item in enumerate(data[:5], 1):  # Limit to 5 items
                if isinstance(item, dict):
                    title = item.get("title", item.get("name", f"項目{i}"))
                    items.append(f"{i}. {title}")
                else:
                    items.append(f"{i}. {str(item)[:100]}")
            return "\n".join(items)

        if isinstance(data, dict):
            return str(data)

        return str(data)
