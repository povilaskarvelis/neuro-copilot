"""
Dynamic MCP tool registry and retrieval helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


_CAPABILITY_STOPWORDS: set[str] = {
    "about",
    "across",
    "after",
    "also",
    "analysis",
    "answer",
    "based",
    "before",
    "between",
    "build",
    "collect",
    "compare",
    "current",
    "data",
    "details",
    "evidence",
    "final",
    "find",
    "from",
    "high",
    "include",
    "into",
    "latest",
    "list",
    "many",
    "method",
    "more",
    "most",
    "need",
    "only",
    "output",
    "query",
    "report",
    "results",
    "review",
    "score",
    "search",
    "show",
    "step",
    "steps",
    "summary",
    "synthesize",
    "this",
    "tool",
    "tools",
    "using",
    "with",
}


@dataclass
class ToolDescriptor:
    name: str
    description: str = ""
    input_schema: dict | None = None
    capabilities: set[str] = field(default_factory=set)
    source: str = "mcp"

    def searchable_text(self) -> str:
        schema_text = ""
        if isinstance(self.input_schema, dict):
            schema_text = str(self.input_schema)
        return f"{self.name} {self.description} {schema_text}".strip().lower()


def infer_capabilities_from_text(text: str) -> set[str]:
    value = re.sub(r"[-/]+", " ", str(text or "").lower())
    tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", value)
    found: set[str] = set()
    for token in tokens:
        if token in _CAPABILITY_STOPWORDS:
            continue
        if len(token) < 4:
            continue
        normalized = token.rstrip("s") if len(token) > 5 else token
        if normalized in _CAPABILITY_STOPWORDS:
            continue
        found.add(normalized)
        if len(found) >= 24:
            break
    return found


class ToolRegistry:
    def __init__(self, tools: list[ToolDescriptor] | None = None) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        for tool in tools or []:
            self._tools[tool.name] = tool

    async def refresh_from_mcp_toolset(self, mcp_tools, *, merge: bool = True) -> int:
        if mcp_tools is None:
            return 0
        try:
            raw_tools = await mcp_tools.get_tools()
        except Exception:
            return 0
        parsed: list[ToolDescriptor] = []
        for tool in raw_tools or []:
            name = str(getattr(tool, "name", "") or "").strip()
            if not name:
                continue
            description = str(getattr(tool, "description", "") or "").strip()
            input_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            caps = infer_capabilities_from_text(f"{name} {description} {input_schema}")
            parsed.append(
                ToolDescriptor(
                    name=name,
                    description=description,
                    input_schema=input_schema if isinstance(input_schema, dict) else None,
                    capabilities=caps,
                    source="mcp",
                )
            )
        if not merge:
            self._tools = {}
        for item in parsed:
            existing = self._tools.get(item.name)
            if existing and existing.capabilities and not item.capabilities:
                item.capabilities = set(existing.capabilities)
            self._tools[item.name] = item
        return len(parsed)

    def get(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(str(name or "").strip())

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def summary(self, *, max_tools: int = 40) -> list[dict]:
        output: list[dict] = []
        for name in self.names()[:max_tools]:
            descriptor = self._tools[name]
            output.append(
                {
                    "name": descriptor.name,
                    "capabilities": sorted(descriptor.capabilities),
                    "source": descriptor.source,
                }
            )
        return output

    def rank_tools(
        self,
        *,
        query: str,
        capability_hints: set[str] | None = None,
        candidates: list[str] | None = None,
        k: int = 8,
    ) -> list[str]:
        hints = set(capability_hints or set())
        query_caps = infer_capabilities_from_text(query)
        hints.update(query_caps)
        pool = [name for name in (candidates or self.names()) if name in self._tools]
        scored: list[tuple[int, str]] = []
        query_lower = str(query or "").lower()
        for name in pool:
            descriptor = self._tools[name]
            score = 0
            overlap = hints.intersection(descriptor.capabilities)
            score += len(overlap) * 4
            text = descriptor.searchable_text()
            if query_lower and query_lower in text:
                score += 2
            for token in re.findall(r"[a-z0-9_]{3,}", query_lower):
                if token in text:
                    score += 1
            # Keep deterministic ordering when scores tie.
            scored.append((score, name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranked = [name for _, name in scored]
        if hints:
            ranked = [name for name in ranked if self._tools[name].capabilities.intersection(hints)] + [
                name for name in ranked if not self._tools[name].capabilities.intersection(hints)
            ]
        return ranked[: max(1, k)]

    def compact_descriptions(self, tool_names: list[str], *, max_chars: int = 180) -> list[str]:
        lines: list[str] = []
        for name in tool_names:
            descriptor = self._tools.get(name)
            if not descriptor:
                continue
            description = re.sub(r"\s+", " ", descriptor.description or "").strip()
            if not description:
                description = "No description available."
            if len(description) > max_chars:
                description = f"{description[: max_chars - 3].rstrip()}..."
            caps = ", ".join(sorted(descriptor.capabilities)[:4]) or "uncategorized"
            lines.append(f"- {name}: {description} (capabilities: {caps})")
        return lines
