"""Direct Tool Executor for MainAgent - allows MainAgent to use tools directly without sub-agents."""

from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
import structlog

from src.agents.state import AgentState, ToolResult
from src.agents.tools.base import ToolContext
from src.agents.registry import get_tool_registry
from src.config.models import should_use_prompt_caching

logger = structlog.get_logger()


DIRECT_TOOL_SYSTEM_PROMPT = """あなたはユーザーサポートアシスタントです。
利用可能なツールを使って、ユーザーの質問に回答してください。

使用方法:
1. ユーザーの質問を理解する
2. 必要に応じてツールを呼び出して情報を収集
3. 収集した情報を基に回答を生成
4. ツールが不要な場合は、直接回答する

回答のガイドライン:
- 具体的で分かりやすい回答を心がける
- 手順は番号付きリストで表示
- 重要な情報は強調
- 技術的すぎない、親しみやすい言葉遣いで説明
"""


class DirectToolExecutor:
    """
    Executor that allows MainAgent to use tools directly.

    This bypasses the SubAgent layer and allows MainAgent to call tools
    using LangChain's tool calling functionality.
    """

    def __init__(self, llm: Any):
        """
        Initialize the DirectToolExecutor.

        Args:
            llm: Language model instance with tool calling support
        """
        self.llm = llm

    async def execute(self, state: AgentState) -> str:
        """
        Execute with direct tool calling.

        Args:
            state: Current agent state

        Returns:
            Final response string
        """
        user_input = state["user_input"]
        messages = state["messages"]
        request_context = state["request_context"]

        # Get available tools
        tools = await self._get_available_tools()

        if not tools:
            # No tools available, respond directly
            return await self._respond_without_tools(state)

        # Create tool context
        tool_context = ToolContext(
            service_tokens=request_context.service_tokens,
            tenant_id=request_context.tenant_id,
            user_id=request_context.user_id,
            request_id=request_context.request_id,
        )

        # Build conversation history
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build prompt with tools
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": DIRECT_TOOL_SYSTEM_PROMPT},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                )
            )

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
            messages_list.append(SystemMessage(content=DIRECT_TOOL_SYSTEM_PROMPT))

            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        messages_list.append(HumanMessage(content=user_input))

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)

        try:
            # Execute with tool calling
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Get LLM response
                response = await llm_with_tools.ainvoke(messages_list)

                # Check if tools were called
                if not response.tool_calls:
                    # No tools called, return the response
                    final_response = response.content if hasattr(response, 'content') else str(response)
                    logger.info(
                        "direct_tool_execution_complete",
                        session_id=state["session_id"],
                        iterations=iteration,
                        tools_used=0,
                    )
                    return final_response

                # Add AI message to conversation
                messages_list.append(response)

                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    logger.info(
                        "direct_tool_call",
                        session_id=state["session_id"],
                        tool_name=tool_name,
                        iteration=iteration,
                    )

                    # Find and execute the tool
                    tool_result = await self._execute_tool(
                        tool_name,
                        tool_args,
                        tool_context,
                    )

                    # Store tool result in state
                    state["tool_results"].append(ToolResult(
                        tool_name=tool_name,
                        agent_name="main_agent_direct",
                        success=tool_result.success,
                        data=tool_result.data,
                        error=tool_result.error,
                        duration_ms=tool_result.duration_ms,
                    ))

                    # Create tool message
                    tool_message_content = str(tool_result.data) if tool_result.success else f"Error: {tool_result.error}"
                    tool_messages.append(
                        ToolMessage(
                            content=tool_message_content,
                            tool_call_id=tool_call["id"],
                        )
                    )

                # Add tool messages to conversation
                messages_list.extend(tool_messages)

            # Max iterations reached
            logger.warning(
                "direct_tool_max_iterations",
                session_id=state["session_id"],
                max_iterations=max_iterations,
            )
            return "申し訳ございません。情報の収集に時間がかかっています。より具体的な質問でお試しください。"

        except Exception as e:
            logger.error(
                "direct_tool_execution_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            return "申し訳ございません。回答の生成中にエラーが発生しました。しばらくしてから再度お試しください。"

    async def execute_stream(self, state: AgentState) -> AsyncIterator[str]:
        """
        Execute with direct tool calling and streaming.

        Args:
            state: Current agent state

        Yields:
            Response tokens
        """
        # For streaming, we need to handle tools synchronously first
        # then stream the final response
        user_input = state["user_input"]
        messages = state["messages"]
        request_context = state["request_context"]

        # Get available tools
        tools = await self._get_available_tools()

        if not tools:
            # No tools available, respond directly with streaming
            async for token in self._respond_without_tools_stream(state):
                yield token
            return

        # Create tool context
        tool_context = ToolContext(
            service_tokens=request_context.service_tokens,
            tenant_id=request_context.tenant_id,
            user_id=request_context.user_id,
            request_id=request_context.request_id,
        )

        # Build conversation history
        conversation_history = self._format_conversation_history(messages[:-1])

        # Check if we should use prompt caching
        model_id = getattr(self.llm, 'model_id', None)
        use_caching = should_use_prompt_caching(model_id) if model_id else False

        # Build initial messages
        messages_list = []

        if use_caching:
            messages_list.append(
                SystemMessage(
                    content=[
                        {"type": "text", "text": DIRECT_TOOL_SYSTEM_PROMPT},
                        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
                    ]
                )
            )

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
            messages_list.append(SystemMessage(content=DIRECT_TOOL_SYSTEM_PROMPT))

            if conversation_history:
                messages_list.append(
                    SystemMessage(content=f"## 会話履歴\n\n{conversation_history}")
                )

        messages_list.append(HumanMessage(content=user_input))

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)

        try:
            # Execute tool calls first (non-streaming)
            max_iterations = 5
            iteration = 0

            while iteration < max_iterations:
                iteration += 1

                # Get LLM response
                response = await llm_with_tools.ainvoke(messages_list)

                # Check if tools were called
                if not response.tool_calls:
                    # No more tools, stream the final response
                    final_response = response.content if hasattr(response, 'content') else str(response)

                    # Stream the response character by character
                    for char in final_response:
                        yield char

                    logger.info(
                        "direct_tool_stream_complete",
                        session_id=state["session_id"],
                        iterations=iteration,
                    )
                    return

                # Add AI message
                messages_list.append(response)

                # Execute tool calls
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    logger.info(
                        "direct_tool_call_stream",
                        session_id=state["session_id"],
                        tool_name=tool_name,
                        iteration=iteration,
                    )

                    # Execute the tool
                    tool_result = await self._execute_tool(
                        tool_name,
                        tool_args,
                        tool_context,
                    )

                    # Store tool result in state
                    state["tool_results"].append(ToolResult(
                        tool_name=tool_name,
                        agent_name="main_agent_direct",
                        success=tool_result.success,
                        data=tool_result.data,
                        error=tool_result.error,
                        duration_ms=tool_result.duration_ms,
                    ))

                    # Create tool message
                    tool_message_content = str(tool_result.data) if tool_result.success else f"Error: {tool_result.error}"
                    tool_messages.append(
                        ToolMessage(
                            content=tool_message_content,
                            tool_call_id=tool_call["id"],
                        )
                    )

                # Add tool messages
                messages_list.extend(tool_messages)

            # Max iterations reached
            yield "申し訳ございません。情報の収集に時間がかかっています。より具体的な質問でお試しください。"

        except Exception as e:
            logger.error(
                "direct_tool_stream_failed",
                session_id=state["session_id"],
                error=str(e),
            )
            yield "申し訳ございません。回答の生成中にエラーが発生しました。しばらくしてから再度お試しください。"

    async def _get_available_tools(self) -> list:
        """Get list of available LangChain tools."""
        tool_registry = get_tool_registry()
        langchain_tools = []

        # Get static tools
        for tool in tool_registry.list_enabled():
            try:
                lc_tool = tool.to_langchain_tool()
                langchain_tools.append(lc_tool)
            except Exception as e:
                logger.warning(
                    "failed_to_convert_tool",
                    tool_name=tool.name,
                    error=str(e),
                )

        # Note: Dynamic tools from DB would need additional conversion logic
        # For now, we only support static tools in direct mode

        return langchain_tools

    async def _execute_tool(self, tool_name: str, tool_args: dict, context: ToolContext):
        """Execute a tool by name."""
        tool_registry = get_tool_registry()

        # Find the tool
        tool = tool_registry.get(tool_name)
        if not tool:
            logger.error("tool_not_found", tool_name=tool_name)
            from src.agents.tools.base import ToolResult
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found",
            )

        # Execute the tool
        result = await tool.execute_with_validation(tool_args, context)
        return result

    async def _respond_without_tools(self, state: AgentState) -> str:
        """Respond without using tools."""
        from src.agents.main_agent.synthesizer import Synthesizer
        synthesizer = Synthesizer(self.llm)
        return await synthesizer._synthesize_fast_response(state)

    async def _respond_without_tools_stream(self, state: AgentState) -> AsyncIterator[str]:
        """Respond without using tools with streaming."""
        from src.agents.main_agent.synthesizer import Synthesizer
        synthesizer = Synthesizer(self.llm)
        async for token in synthesizer._synthesize_fast_response_stream(state):
            yield token

    def _format_conversation_history(self, messages: list) -> str:
        """Format conversation history for inclusion in prompt."""
        if not messages:
            return ""

        # Take last 10 messages for context
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
