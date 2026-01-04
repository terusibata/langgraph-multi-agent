"""Source reference normalization for multi-agent system."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """
    Unified source reference format for all tool results.

    This normalizes different tool outputs into a consistent structure
    that can be easily processed and displayed.
    """

    id: str = Field(..., description="Unique identifier for this source")
    type: Literal[
        "knowledge_base",
        "vector_search",
        "case",
        "catalog",
        "document",
        "web",
        "custom"
    ] = Field(..., description="Type of source")
    title: str = Field(..., description="Title or name of the source")
    content: str = Field(..., description="Main content or snippet")
    url: str | None = Field(default=None, description="URL or reference link")
    score: float | None = Field(
        default=None,
        description="Relevance score (0-1) if available"
    )
    agent_name: str | None = Field(
        default=None,
        description="Name of agent that retrieved this source"
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of tool that retrieved this source"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific metadata"
    )
    retrieved_at: str | None = Field(
        default=None,
        description="ISO timestamp when retrieved"
    )


class SourceCollection(BaseModel):
    """Collection of normalized source references."""

    sources: list[SourceReference] = Field(default_factory=list)
    total_count: int = Field(default=0)

    def add_source(self, source: SourceReference) -> None:
        """Add a source to the collection."""
        self.sources.append(source)
        self.total_count = len(self.sources)

    def get_by_type(self, source_type: str) -> list[SourceReference]:
        """Get all sources of a specific type."""
        return [s for s in self.sources if s.type == source_type]

    def get_by_agent(self, agent_name: str) -> list[SourceReference]:
        """Get all sources from a specific agent."""
        return [s for s in self.sources if s.agent_name == agent_name]

    def sort_by_score(self, descending: bool = True) -> list[SourceReference]:
        """Sort sources by relevance score."""
        scored = [s for s in self.sources if s.score is not None]
        unscored = [s for s in self.sources if s.score is None]
        scored.sort(key=lambda x: x.score, reverse=descending)
        return scored + unscored


def normalize_tool_result(
    tool_name: str,
    agent_name: str,
    result_data: Any,
) -> list[SourceReference]:
    """
    Normalize tool result into SourceReference format.

    Args:
        tool_name: Name of the tool that produced the result
        agent_name: Name of the agent that used the tool
        result_data: Raw result data from the tool

    Returns:
        List of normalized SourceReference objects
    """
    sources = []

    # Handle different result formats
    if isinstance(result_data, list):
        for i, item in enumerate(result_data):
            source = _normalize_single_item(tool_name, agent_name, item, i)
            if source:
                sources.append(source)
    elif isinstance(result_data, dict):
        # Check if it's a single result or a results container
        if "results" in result_data:
            for i, item in enumerate(result_data["results"]):
                source = _normalize_single_item(tool_name, agent_name, item, i)
                if source:
                    sources.append(source)
        else:
            source = _normalize_single_item(tool_name, agent_name, result_data, 0)
            if source:
                sources.append(source)

    return sources


def _normalize_single_item(
    tool_name: str,
    agent_name: str,
    item: dict,
    index: int,
) -> SourceReference | None:
    """Normalize a single item into SourceReference."""
    if not isinstance(item, dict):
        return None

    # Determine source type based on tool name and data
    source_type = _infer_source_type(tool_name, item)

    # Extract common fields with fallbacks
    source_id = (
        item.get("id") or
        item.get("sys_id") or
        item.get("kb_id") or
        item.get("number") or
        f"{agent_name}_{index}"
    )

    title = (
        item.get("title") or
        item.get("name") or
        item.get("short_description") or
        f"Result {index + 1}"
    )

    content = (
        item.get("content") or
        item.get("description") or
        item.get("snippet") or
        item.get("text") or
        ""
    )

    url = (
        item.get("url") or
        item.get("link") or
        item.get("sys_id")
    )

    # Extract score if available
    score = (
        item.get("score") or
        item.get("relevance") or
        item.get("similarity") or
        item.get("confidence")
    )

    # Extract metadata (everything else)
    metadata = {
        k: v for k, v in item.items()
        if k not in ["id", "sys_id", "kb_id", "number", "title", "name",
                     "short_description", "content", "description", "snippet",
                     "text", "url", "link", "score", "relevance", "similarity",
                     "confidence"]
    }

    return SourceReference(
        id=str(source_id),
        type=source_type,
        title=title,
        content=content[:500] if content else "",  # Limit content length
        url=url,
        score=float(score) if score is not None else None,
        agent_name=agent_name,
        tool_name=tool_name,
        metadata=metadata,
    )


def _infer_source_type(tool_name: str, item: dict) -> str:
    """Infer source type from tool name and item data."""
    tool_lower = tool_name.lower()

    if "knowledge" in tool_lower or "kb" in tool_lower:
        return "knowledge_base"
    elif "vector" in tool_lower or "embedding" in tool_lower:
        return "vector_search"
    elif "case" in tool_lower or "incident" in tool_lower:
        return "case"
    elif "catalog" in tool_lower or "service" in tool_lower:
        return "catalog"
    elif "document" in tool_lower or "doc" in tool_lower:
        return "document"
    elif "web" in tool_lower or "search" in tool_lower:
        return "web"

    # Check item metadata
    if "kb_id" in item or "article_id" in item:
        return "knowledge_base"
    elif "case_number" in item or "incident_number" in item:
        return "case"
    elif "similarity" in item or "embedding" in item:
        return "vector_search"

    return "custom"
