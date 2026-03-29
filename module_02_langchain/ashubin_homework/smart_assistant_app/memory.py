from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .models import MemoryStrategy


NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\-']+)", re.IGNORECASE),
    re.compile(r"\bменя зовут\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-']+)", re.IGNORECASE),
]

LANGUAGE_PATTERNS = [
    re.compile(r"\bmy favorite language is\s+([A-Za-z0-9+#.\-]+)", re.IGNORECASE),
    re.compile(r"\bмой любимый язык(?: программирования)?(?:\s*(?:-|:|это)\s*|\s+)([A-Za-zА-Яа-яЁё0-9+#.\-]+)", re.IGNORECASE),
]

CITY_PATTERNS = [
    re.compile(r"\bi live in\s+([A-Za-z][A-Za-z .'\-]+)", re.IGNORECASE),
    re.compile(r"\bя живу в\s+([A-Za-zА-Яа-яЁё .'\-]+)", re.IGNORECASE),
]


def _clean_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def render_messages(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        label = "System"
        if isinstance(message, HumanMessage):
            label = "Human"
        elif isinstance(message, AIMessage):
            label = "Assistant"
        lines.append(f"{label}: {_clean_content(message)}")
    return "\n".join(lines)


@dataclass
class MemoryManager:
    strategy: MemoryStrategy = MemoryStrategy.BUFFER
    max_messages: int = 20
    summary_tail_messages: int = 6
    history: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    entities: dict[str, str] = field(default_factory=dict)

    def set_strategy(self, strategy: MemoryStrategy) -> None:
        self.strategy = strategy
        if self.strategy is MemoryStrategy.BUFFER and len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages :]

    def clear(self, include_entities: bool = False) -> None:
        self.history.clear()
        self.summary = ""
        if include_entities:
            self.entities.clear()

    def add_turn(
        self,
        user_text: str,
        assistant_text: str,
        summarizer: Callable[[str, str], str] | None = None,
    ) -> None:
        self._extract_entities(user_text)
        self.history.append(HumanMessage(content=user_text))
        self.history.append(AIMessage(content=assistant_text))

        if self.strategy is MemoryStrategy.BUFFER:
            self.history = self.history[-self.max_messages :]
            return

        if len(self.history) > self.max_messages:
            self._compress_history(summarizer)

    def context_messages(self) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        if self.summary:
            messages.append(SystemMessage(content=f"Conversation summary:\n{self.summary}"))
        if self.entities:
            messages.append(SystemMessage(content=self.entity_context()))
        messages.extend(self.history[-self.max_messages :])
        return messages

    def entity_context(self) -> str:
        if not self.entities:
            return "No durable user facts have been stored yet."
        lines = ["Durable user facts:"]
        for key, value in sorted(self.entities.items()):
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def status(self) -> dict[str, int | str]:
        return {
            "strategy": self.strategy.value,
            "messages": len(self.history),
            "entities": len(self.entities),
            "has_summary": int(bool(self.summary)),
        }

    def _compress_history(self, summarizer: Callable[[str, str], str] | None = None) -> None:
        cutoff = max(0, len(self.history) - self.summary_tail_messages)
        if cutoff <= 0:
            return

        old_messages = self.history[:cutoff]
        transcript = render_messages(old_messages)
        if not transcript.strip():
            return

        if summarizer is None:
            raise RuntimeError("Summary memory requires a summarizer callback.")

        self.summary = summarizer(self.summary, transcript).strip()
        self.history = self.history[cutoff:]

    def _extract_entities(self, user_text: str) -> None:
        self._extract_first_match(NAME_PATTERNS, user_text, "name")
        self._extract_first_match(LANGUAGE_PATTERNS, user_text, "favorite_language")
        self._extract_first_match(CITY_PATTERNS, user_text, "city")

    def _extract_first_match(self, patterns: list[re.Pattern[str]], text: str, key: str) -> None:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip(" .,!?:;")
                if value:
                    self.entities[key] = value
                return
