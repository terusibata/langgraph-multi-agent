"""Synthesizer component for MainAgent."""

import json
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

FAST_RESPONSE_SYSTEM_PROMPT = """あなたはユーザーサポートアシスタントです。
ユーザーの質問に対して、あなたの知識のみを使って迅速に回答してください。

回答のガイドライン:
1. 一般的な知識や経験に基づいて回答
2. 具体的な解決手順がある場合は番号付きリストで表示
3. 確信が持てない情報は、その旨を明記
4. 簡潔で分かりやすい回答を心がける
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
            Final response string (may be JSON if response_format='json')
        """
        # Check if JSON format is requested
        if state.get("response_format") == "json":
            return await self._synthesize_json_response(state)

        # Check if fast response mode is enabled
        if state.get("fast_response", False):
            return await self._synthesize_fast_response(state)

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
        # Check if fast response mode is enabled
        if state.get("fast_response", False):
            async for token in self._synthesize_fast_response_stream(state):
                yield token
            return

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

    def _format_resources_for_json(self, sub_agent_results: dict) -> list[dict]:
        """
        Format agent results as standardized resource objects for JSON response.

        Returns:
            List of standardized resource objects
        """
        resources = []

        for agent_name, result in sub_agent_results.items():
            if result.status == "failed" or not result.data:
                continue

            # Determine tool_name mapping
            tool_name_map = {
                "knowledge_search": "frontend_servicenow_search",
                "vector_search": "frontend_vector_search",
                "catalog": "frontend_catalog_search",
            }
            tool_name = tool_name_map.get(agent_name, f"frontend_{agent_name}")

            # Process result data
            if isinstance(result.data, list):
                for item in result.data:
                    if isinstance(item, dict):
                        resource = {
                            "id": item.get("sys_id", item.get("id", item.get("kb_number", ""))),
                            "type": "knowledge_base" if "knowledge" in agent_name else "document",
                            "title": item.get("title", item.get("name", "")),
                            "content": item.get("content", item.get("description", item.get("snippet", ""))),
                            "score": item.get("score", item.get("relevance", None)),
                            "tool_name": tool_name,
                            "metadata": {
                                k: v for k, v in item.items()
                                if k not in ["id", "sys_id", "kb_number", "title", "name", "content", "description", "snippet", "score", "relevance"]
                            }
                        }
                        resources.append(resource)
            elif isinstance(result.data, dict):
                resource = {
                    "id": result.data.get("sys_id", result.data.get("id", result.data.get("kb_number", ""))),
                    "type": "knowledge_base" if "knowledge" in agent_name else "document",
                    "title": result.data.get("title", result.data.get("name", "")),
                    "content": result.data.get("content", result.data.get("description", result.data.get("snippet", ""))),
                    "score": result.data.get("score", result.data.get("relevance", None)),
                    "tool_name": tool_name,
                    "metadata": {
                        k: v for k, v in result.data.items()
                        if k not in ["id", "sys_id", "kb_number", "title", "name", "content", "description", "snippet", "score", "relevance"]
                    }
                }
                resources.append(resource)

        return resources

    async def generate_thread_title(self, user_input: str) -> str:
        """
        Generate a short Japanese title for the thread based on the user's first message.

        Args:
            user_input: User's first message

        Returns:
            Short Japanese title (max 50 characters)
        """
        prompt_content = f"""以下のユーザーの最初のメッセージから、会話のタイトルを生成してください。

ユーザーのメッセージ:
{user_input}

要件:
- 日本語で簡潔に（最大30文字）
- メッセージの主要なトピックを反映
- 「〜について」のような不自然な表現は避ける
- 質問形式ではなく、トピックを表す名詞句で

例:
- ユーザー: "プリンターに接続できない不具合が発生していますが、どうすればいいですか？"
  タイトル: "プリンター接続トラブル"

- ユーザー: "VPNの設定方法を教えてください"
  タイトル: "VPN設定方法"

タイトルのみを返してください（説明や記号は不要）。"""

        messages_list = [
            SystemMessage(content="あなたは会話のタイトルを生成する専門アシスタントです。簡潔で分かりやすい日本語のタイトルを生成してください。"),
            HumanMessage(content=prompt_content)
        ]

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm
            result = await chain.ainvoke({})

            title = result.content if hasattr(result, 'content') else str(result)

            # Clean and trim the title
            title = title.strip().strip('"').strip("'")

            # Limit to 50 characters
            if len(title) > 50:
                title = title[:50]

            logger.info("thread_title_generated", title=title)
            return title

        except Exception as e:
            logger.error("thread_title_generation_failed", error=str(e))
            # Fallback: use first 30 characters of user input
            fallback_title = user_input[:30].strip()
            if len(user_input) > 30:
                fallback_title += "..."
            return fallback_title

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

    async def _synthesize_fast_response(self, state: AgentState) -> str:
        """
        Generate fast response using only LLM knowledge (no sub-agents or tools).

        Args:
            state: Current agent state

        Returns:
            Fast response string
        """
        user_input = state["user_input"]
        messages = state["messages"]

        # Build conversation history context (last 10 messages, excluding current)
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching based on model support
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with caching enabled for system prompt if supported
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": FAST_RESPONSE_SYSTEM_PROMPT},
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
            messages_list.append(SystemMessage(content=FAST_RESPONSE_SYSTEM_PROMPT))

            # Add conversation history without caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        # Add current request
        messages_list.append(HumanMessage(content=user_input))

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm
            result = await chain.ainvoke({})

            response = result.content if hasattr(result, 'content') else str(result)

            logger.info(
                "fast_response_complete",
                session_id=state["session_id"],
                response_length=len(response),
            )

            return response

        except Exception as e:
            logger.error(
                "fast_response_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            return "申し訳ございません。現在、回答の生成中にエラーが発生しました。しばらくしてから再度お試しください。"

    async def _synthesize_json_response(self, state: AgentState) -> str:
        """
        Generate JSON-formatted response based on response_schema.

        Args:
            state: Current agent state

        Returns:
            JSON string response
        """
        user_input = state["user_input"]
        sub_agent_results = state["sub_agent_results"]
        response_schema = state.get("response_schema", {})

        # Format resources in standardized format
        resources = self._format_resources_for_json(sub_agent_results)

        # Build the prompt to generate JSON response
        schema_str = json.dumps(response_schema, ensure_ascii=False, indent=2) if response_schema else "{}"

        prompt_content = f"""ユーザーの質問: {user_input}

検索されたリソース:
{json.dumps(resources, ensure_ascii=False, indent=2)}

以下のJSONスキーマに従って、ユーザーの質問に対する回答をJSON形式で生成してください:
{schema_str}

要件:
- 必ず有効なJSON形式で応答してください
- スキーマで定義されたフィールドのみを含めてください
- リソースがある場合は、それらの情報も含めてください
- 日本語で回答してください"""

        messages_list = [
            SystemMessage(content="あなたは構造化されたJSON応答を生成する専門アシスタントです。必ず有効なJSON形式で回答してください。"),
            HumanMessage(content=prompt_content)
        ]

        prompt = ChatPromptTemplate.from_messages(messages_list)

        try:
            chain = prompt | self.llm
            result = await chain.ainvoke({})

            response = result.content if hasattr(result, 'content') else str(result)

            # Try to parse and validate JSON
            try:
                json_response = json.loads(response)
                # Add resources to the response if not already included and if schema allows
                if "resources" in response_schema.get("properties", {}) and "resources" not in json_response:
                    json_response["resources"] = resources
                return json.dumps(json_response, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                # If LLM didn't return valid JSON, wrap it in a basic structure
                logger.warning("llm_returned_invalid_json", session_id=state["session_id"])
                fallback_response = {
                    "answer": response,
                    "resources": resources
                }
                return json.dumps(fallback_response, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(
                "json_synthesis_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            # Return error in JSON format
            error_response = {
                "error": "応答の生成中にエラーが発生しました",
                "resources": resources
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)

    async def _synthesize_fast_response_stream(self, state: AgentState) -> AsyncIterator[str]:
        """
        Generate fast response with streaming (no sub-agents or tools).

        Args:
            state: Current agent state

        Yields:
            Response tokens
        """
        user_input = state["user_input"]
        messages = state["messages"]

        # Build conversation history context (last 10 messages, excluding current)
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching based on model support
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with caching enabled for system prompt if supported
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": FAST_RESPONSE_SYSTEM_PROMPT},
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
            messages_list.append(SystemMessage(content=FAST_RESPONSE_SYSTEM_PROMPT))

            # Add conversation history without caching if available
            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        # Add current request
        messages_list.append(HumanMessage(content=user_input))

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
                "fast_response_stream_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            yield "申し訳ございません。現在、回答の生成中にエラーが発生しました。しばらくしてから再度お試しください。"
