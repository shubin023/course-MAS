from __future__ import annotations

import json
import re
from collections.abc import Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr

from .heuristics import generate_handler_reply, heuristic_classify, simple_summary
from .models import RequestType


def _messages_to_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


def _extract_latest_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        content = str(message.content).strip()
        if content.startswith("User query:"):
            return content.removeprefix("User query:").strip()
        if content:
            return content
    return ""


def _extract_entities(messages: list[BaseMessage]) -> dict[str, str]:
    text = _messages_to_text(messages)
    entities: dict[str, str] = {}
    for key in ("name", "favorite_language", "city"):
        match = re.search(rf"- {key}: (.+)", text)
        if match:
            entities[key] = match.group(1).strip()
    return entities


class RuleBasedChatModel(BaseChatModel):
    character: str = "friendly"
    _model_name: str = PrivateAttr(default="rule-based-chat-model")

    @property
    def _llm_type(self) -> str:
        return self._model_name

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        content = self._respond(messages)
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        content = self._respond(messages)
        for token in content.split():
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"{token} "))

    def _respond(self, messages: list[BaseMessage]) -> str:
        text = _messages_to_text(messages)

        if "[TASK:CLASSIFY_REQUEST]" in text:
            query = _extract_latest_query(messages)
            classification = heuristic_classify(query)
            return classification.model_dump_json()

        if "[TASK:SUMMARIZE_HISTORY]" in text:
            existing = re.search(r"Existing summary:\n(.*?)\n\nConversation to compress:", text, re.DOTALL)
            transcript = re.search(r"Conversation to compress:\n(.*)", text, re.DOTALL)
            existing_summary = existing.group(1).strip() if existing else ""
            transcript_text = transcript.group(1).strip() if transcript else text
            return simple_summary(existing_summary, transcript_text)

        handler_match = re.search(r"\[HANDLER:([a-z_]+)\]", text)
        handler_name = handler_match.group(1) if handler_match else RequestType.UNKNOWN.value
        request_type = RequestType(handler_name)
        query = _extract_latest_query(messages)
        entities = _extract_entities(messages)
        return generate_handler_reply(request_type, query, entities, self.character)
