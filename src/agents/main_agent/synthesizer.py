"""Synthesizer component for MainAgent."""

from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import structlog

from src.agents.state import AgentState
from src.config.models import should_use_prompt_caching

logger = structlog.get_logger()


SYNTHESIZER_SYSTEM_PROMPT = """あなたはユーザーサポートアシスタントです。
検索結果を基に、ユーザーの質問に対して分かりやすく回答してください。

回答のガイドライン:
1. 検索結果の情報を整理して提示
2. 具体的な解決手順がある場合は番号付きリストで表示
3. 参照元（ナレッジ記事番号等）があれば記載
4. 解決策がない場合は、問い合わせ先や申請方法を案内
5. 技術的すぎない、親しみやすい言葉遣いで説明

フォーマット:
- 見出しは適切に使用
- 手順は番号付きリスト
- 重要な情報は強調
- 長すぎない回答（必要十分な情報量）
"""


class Synthesizer:
    """
    Synthesizer component that generates the final response.
    """

    def __init__(self, llm: Any):
        """
        Initialize the Synthesizer.

        Args:
            llm: Language model instance for synthesis
        """
        self.llm = llm

    async def synthesize(self, state: AgentState) -> str:
        """
        Generate the final response based on all gathered information.

        Args:
            state: Current agent state

        Returns:
            Final response string
        """
        user_input = state["user_input"]
        sub_agent_results = state["sub_agent_results"]
        evaluation = state["intermediate_evaluation"]
        messages = state["messages"]

        # Format context for synthesis
        context = self._build_context(sub_agent_results, evaluation)

        # Build conversation history context (last 10 messages, excluding current)
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching based on model support
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with caching enabled for system prompt AND conversation history if supported
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": SYNTHESIZER_SYSTEM_PROMPT},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                )
            )

            # Add conversation history with caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(
                        content=[
                            {"type": "text", "text": f"## 会話履歴\n\n{conversation_history}"},
                            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                        ]
                    )
                )
        else:
            messages_list.append(SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT))

            # Add conversation history without caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        # Add current request
        messages_list.append(
            HumanMessage(content=f"""
ユーザーの質問: {user_input}

収集した情報:
{context}

上記の情報を基に、ユーザーへの回答を生成してください。会話履歴がある場合は、その文脈も考慮してください。
""")
        )

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm
            result = await chain.ainvoke({})

            response = result.content if hasattr(result, 'content') else str(result)

            logger.info(
                "synthesis_complete",
                session_id=state["session_id"],
                response_length=len(response),
            )

            return response

        except Exception as e:
            logger.error(
                "synthesis_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            return self._generate_fallback_response(sub_agent_results)

    async def synthesize_stream(self, state: AgentState) -> AsyncIterator[str]:
        """
        Generate the final response with streaming.

        Args:
            state: Current agent state

        Yields:
            Response tokens
        """
        user_input = state["user_input"]
        sub_agent_results = state["sub_agent_results"]
        evaluation = state["intermediate_evaluation"]
        messages = state["messages"]

        context = self._build_context(sub_agent_results, evaluation)

        # Build conversation history context (last 10 messages, excluding current)
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching based on model support
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with caching enabled for system prompt AND conversation history if supported
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": SYNTHESIZER_SYSTEM_PROMPT},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                )
            )

            # Add conversation history with caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(
                        content=[
                            {"type": "text", "text": f"## 会話履歴\n\n{conversation_history}"},
                            {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                        ]
                    )
                )
        else:
            messages_list.append(SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT))

            # Add conversation history without caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        # Add current request
        messages_list.append(
            HumanMessage(content=f"""
ユーザーの質問: {user_input}

収集した情報:
{context}

上記の情報を基に、ユーザーへの回答を生成してください。会話履歴がある場合は、その文脈も考慮してください。
""")
        )

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm
            async for chunk in chain.astream({}):
                if hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)

        except Exception as e:
            logger.error(
                "synthesis_stream_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            yield self._generate_fallback_response(sub_agent_results)

    def _build_context(self, sub_agent_results: dict, evaluation: Any) -> str:
        """Build context string from results."""
        sections = []

        for agent_name, result in sub_agent_results.items():
            if result.status == "failed":
                continue

            section = self._format_agent_result(agent_name, result)
            if section:
                sections.append(section)

        if evaluation.reasoning:
            sections.append(f"評価コメント: {evaluation.reasoning}")

        return "\n\n---\n\n".join(sections) if sections else "情報が見つかりませんでした。"

    def _format_agent_result(self, agent_name: str, result: Any) -> str:
        """Format a single agent's result."""
        agent_labels = {
            "knowledge_search": "ナレッジベース検索結果",
            "vector_search": "関連ドキュメント検索結果",
            "catalog": "サービスカタログ情報",
        }

        label = agent_labels.get(agent_name, agent_name)

        if not result.data:
            return ""

        if isinstance(result.data, list):
            if len(result.data) == 0:
                return ""

            items = []
            for item in result.data[:5]:
                if isinstance(item, dict):
                    title = item.get("title", item.get("name", ""))
                    content = item.get("content", item.get("description", item.get("snippet", "")))
                    source = item.get("source", item.get("sys_id", ""))

                    item_text = f"**{title}**"
                    if content:
                        item_text += f"\n{content[:300]}"
                    if source:
                        item_text += f"\n(参照: {source})"
                    items.append(item_text)
                else:
                    items.append(str(item)[:200])

            return f"### {label}\n\n" + "\n\n".join(items)

        if isinstance(result.data, dict):
            return f"### {label}\n\n{self._format_dict(result.data)}"

        return f"### {label}\n\n{str(result.data)[:500]}"

    def _format_dict(self, data: dict) -> str:
        """Format a dictionary for display."""
        lines = []
        for key, value in data.items():
            if isinstance(value, str):
                lines.append(f"- **{key}**: {value[:200]}")
            elif isinstance(value, list):
                lines.append(f"- **{key}**: {len(value)}件")
            else:
                lines.append(f"- **{key}**: {str(value)[:100]}")
        return "\n".join(lines)

    def _format_conversation_history(self, messages: list) -> str:
        """
        Format conversation history for inclusion in prompt.

        Args:
            messages: List of BaseMessage objects from conversation history

        Returns:
            Formatted conversation history string (last 10 messages)
        """
        if not messages:
            return ""

        # Take last 10 messages for context (5 turns)
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        formatted = []
        for msg in recent_messages:
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            formatted.append(f"**{role}**: {content}")

        return "\n\n".join(formatted)

    def _generate_fallback_response(self, sub_agent_results: dict) -> str:
        """Generate a fallback response when synthesis fails."""
        # Check if we have any successful results
        has_results = any(
            result.status == "success" and result.data
            for result in sub_agent_results.values()
        )

        if has_results:
            return """申し訳ございません。検索結果の処理中にエラーが発生しました。

検索自体は完了しておりますので、以下をお試しください：
1. 少し時間をおいてから再度お問い合わせください
2. より具体的なキーワードで検索してみてください

問題が続く場合は、サポート窓口までご連絡ください。"""

        return """申し訳ございません。お探しの情報が見つかりませんでした。

以下をお試しください：
1. 異なるキーワードで検索してみてください
2. より具体的な内容でお問い合わせください

それでも解決しない場合は、サービスカタログから適切な申請フォームをご利用ください。"""
