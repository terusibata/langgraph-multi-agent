"""Evaluator component for MainAgent."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import structlog

from src.agents.state import AgentState, Evaluation

logger = structlog.get_logger()


EVALUATOR_SYSTEM_PROMPT = """あなたはマルチエージェントシステムの評価者です。
SubAgentからの検索結果を評価し、次のアクションを決定してください。

評価基準:
1. 検索結果がユーザーの質問に十分な回答を提供できるか
2. 追加の情報（カタログ調査など）が必要か
3. 回答を生成できる状態か

出力形式（JSON）:
{{
    "has_sufficient_info": true/false,
    "missing_info": ["不足している情報のリスト"],
    "next_action": "respond" | "catalog" | "clarify",
    "reasoning": "判断理由"
}}

next_actionの選択基準:
- "respond": 十分な情報があり、回答を生成できる
- "catalog": 解決策がなく、申請先の案内が必要
- "clarify": 情報が曖昧で、ユーザーに確認が必要
"""


class Evaluator:
    """
    Evaluator component that assesses intermediate results and decides next steps.
    """

    def __init__(self, llm: Any):
        """
        Initialize the Evaluator.

        Args:
            llm: Language model instance for evaluation
        """
        self.llm = llm
        self.output_parser = JsonOutputParser()

    async def evaluate(self, state: AgentState) -> Evaluation:
        """
        Evaluate the current state and determine next action.

        Args:
            state: Current agent state

        Returns:
            Evaluation with next action recommendation
        """
        user_input = state["user_input"]
        sub_agent_results = state["sub_agent_results"]
        messages = state["messages"]

        # Format results for evaluation
        results_summary = self._format_results_summary(sub_agent_results)

        # Build conversation history (last 6 messages for brief context)
        conversation_history = self._format_conversation_history(messages[:-1], max_messages=6)

        # Build prompt with caching enabled for system prompt
        messages_list = [
            SystemMessage(
                content=[
                    {"type": "text", "text": EVALUATOR_SYSTEM_PROMPT},
                    {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                ]
            ),
        ]

        # Add conversation history if available
        if conversation_history:
            messages_list.append(
                SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
            )

        # Add current evaluation request
        messages_list.append(
            HumanMessage(content=f"""
ユーザーの質問: {user_input}

検索結果:
{results_summary}

これらの結果を評価し、次のアクションを決定してください。会話履歴がある場合は、その文脈も考慮してください。
""")
        )

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm | self.output_parser
            result = await chain.ainvoke({})

            evaluation = Evaluation(
                has_sufficient_info=result.get("has_sufficient_info", False),
                missing_info=result.get("missing_info", []),
                next_action=result.get("next_action", "respond"),
                reasoning=result.get("reasoning"),
            )

            logger.info(
                "evaluation_complete",
                session_id=state["session_id"],
                has_sufficient_info=evaluation.has_sufficient_info,
                next_action=evaluation.next_action,
            )

            return evaluation

        except Exception as e:
            logger.error(
                "evaluation_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            # Return default evaluation
            return self._create_default_evaluation(sub_agent_results)

    def _format_results_summary(self, sub_agent_results: dict) -> str:
        """Format SubAgent results for evaluation prompt."""
        summaries = []

        for agent_name, result in sub_agent_results.items():
            status_emoji = {
                "success": "✓",
                "partial": "△",
                "failed": "✗",
            }.get(result.status, "?")

            if result.status == "failed":
                summary = f"[{status_emoji}] {agent_name}: エラー - {result.error}"
            elif result.data:
                data_summary = self._summarize_data(result.data)
                retry_info = ""
                if result.retry_count > 0:
                    retry_info = f" (リトライ{result.retry_count}回)"
                summary = f"[{status_emoji}] {agent_name}{retry_info}:\n{data_summary}"
            else:
                summary = f"[{status_emoji}] {agent_name}: 結果なし"

            summaries.append(summary)

        return "\n\n".join(summaries)

    def _summarize_data(self, data: Any) -> str:
        """Summarize result data."""
        if data is None:
            return "データなし"

        if isinstance(data, str):
            return data[:500] + "..." if len(data) > 500 else data

        if isinstance(data, list):
            if len(data) == 0:
                return "0件の結果"

            items = []
            for i, item in enumerate(data[:3], 1):  # First 3 items
                if isinstance(item, dict):
                    title = item.get("title", item.get("name", f"項目{i}"))
                    snippet = item.get("snippet", item.get("description", ""))[:100]
                    items.append(f"  {i}. {title}\n     {snippet}")
                else:
                    items.append(f"  {i}. {str(item)[:100]}")

            remaining = len(data) - 3
            if remaining > 0:
                items.append(f"  ... 他{remaining}件")

            return "\n".join(items)

        if isinstance(data, dict):
            return str(data)[:500]

        return str(data)[:500]

    def _format_conversation_history(self, messages: list, max_messages: int = 10) -> str:
        """
        Format conversation history for inclusion in prompt.

        Args:
            messages: List of BaseMessage objects from conversation history
            max_messages: Maximum number of messages to include

        Returns:
            Formatted conversation history string
        """
        if not messages:
            return ""

        # Take last N messages for context
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages

        formatted = []
        for msg in recent_messages:
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate long messages
            if len(content) > 300:
                content = content[:300] + "..."
            formatted.append(f"**{role}**: {content}")

        return "\n\n".join(formatted)

    def _create_default_evaluation(self, sub_agent_results: dict) -> Evaluation:
        """Create a default evaluation based on results."""
        # Check if any agent was successful
        has_results = any(
            result.status == "success" and result.data
            for result in sub_agent_results.values()
        )

        if has_results:
            return Evaluation(
                has_sufficient_info=True,
                missing_info=[],
                next_action="respond",
                reasoning="デフォルト評価: 検索結果あり",
            )
        else:
            return Evaluation(
                has_sufficient_info=False,
                missing_info=["検索結果"],
                next_action="catalog",
                reasoning="デフォルト評価: 検索結果なし、カタログ調査を推奨",
            )

    def needs_catalog_search(self, evaluation: Evaluation) -> bool:
        """Check if catalog search is needed."""
        return evaluation.next_action == "catalog"

    def can_respond(self, evaluation: Evaluation) -> bool:
        """Check if we can generate a response."""
        return evaluation.next_action == "respond" or evaluation.has_sufficient_info

    def get_evaluation_summary(self, evaluation: Evaluation) -> dict:
        """Get a summary of the evaluation for events."""
        return {
            "has_sufficient_info": evaluation.has_sufficient_info,
            "missing_info": evaluation.missing_info,
            "next_action": evaluation.next_action,
            "reasoning": evaluation.reasoning,
        }
