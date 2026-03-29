from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from .memory import MemoryManager
from .model_factory import ProviderConfigError, build_resilient_model, configure_cache, load_project_env
from .models import AssistantResponse, Classification, MemoryStrategy, RequestType
from .personas import CHARACTER_PROMPTS, HANDLER_INSTRUCTIONS


CLASSIFIER_TYPE_DESCRIPTIONS = """
Request type definitions:
- question: the user asks for information, explanation, clarification, or a factual answer.
- task: the user asks the assistant to do something, create something, transform text, or perform an action.
- small_talk: greetings, introductions, casual conversation, polite chatter, or social interaction.
- complaint: frustration, criticism, negative feedback, or reporting that something works badly.
- unknown: gibberish, meaningless text, or input whose intent cannot be determined reliably.
""".strip()


CLASSIFIER_FEW_SHOTS = """
Few-shot examples:
Input: "Привет!"
Output class: small_talk

Input: "Меня зовут Анна"
Output class: small_talk

Input: "Что такое Python?"
Output class: question

Input: "How does LCEL work?"
Output class: question

Input: "Напиши короткое стихотворение про осень"
Output class: task

Input: "Summarize this text in three bullet points"
Output class: task

Input: "Это ужасно работает, почему так медленно?"
Output class: complaint

Input: "Your answer was useless and frustrating"
Output class: complaint

Input: "asdfghjkl"
Output class: unknown

Input: "тррр зщщщ ккк"
Output class: unknown
""".strip()


@dataclass
class AssistantConfig:
    provider: str = "openrouter"
    model: str = "gpt-5-mini"
    fallback_provider: str | None = None
    fallback_model: str | None = None
    character: str = "friendly"
    memory_strategy: MemoryStrategy = MemoryStrategy.BUFFER
    max_messages: int = 20
    summary_tail_messages: int = 6
    temperature: float = 0.2
    stream: bool = False
    cache_backend: str = "none"
    cache_path: Path | None = None
    env_path: Path | None = None
    base_url: str | None = None
    api_key: str | None = None


class SmartAssistant:
    def __init__(
        self,
        *,
        config: AssistantConfig | None = None,
        model=None,
        classifier_model=None,
        summary_model=None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.config = config or AssistantConfig()
        self._validate_character(self.config.character)

        if self.config.cache_backend != "none":
            configure_cache(self.config.cache_backend, self.config.cache_path)

        if model is None:
            load_project_env(self.config.env_path)
            try:
                model = build_resilient_model(
                    self.config.provider,
                    self.config.model,
                    fallback_model=self.config.fallback_model,
                    fallback_provider=self.config.fallback_provider,
                    temperature=self.config.temperature,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    streaming=self.config.stream,
                    character=self.config.character,
                )
            except ProviderConfigError as error:
                raise RuntimeError(str(error)) from error

        self.response_model = model
        self.classifier_model = classifier_model or model
        self.summary_model = summary_model or model
        self.memory = memory_manager or MemoryManager(
            strategy=self.config.memory_strategy,
            max_messages=self.config.max_messages,
            summary_tail_messages=self.config.summary_tail_messages,
        )
        self.classification_parser = PydanticOutputParser(pydantic_object=Classification)

        self._build_chains()

    def set_character(self, character: str) -> None:
        self._validate_character(character)
        self.config.character = character
        self._sync_model_character(character)
        self._build_chains()

    def set_memory_strategy(self, strategy: MemoryStrategy | str) -> None:
        value = MemoryStrategy(strategy)
        self.config.memory_strategy = value
        self.memory.set_strategy(value)

    def clear(self, include_entities: bool = False) -> None:
        self.memory.clear(include_entities=include_entities)

    def status(self) -> str:
        memory_status = self.memory.status()
        return (
            f"character={self.config.character} | memory={self.config.memory_strategy.value} | "
            f"provider={self.config.provider} | model={self.config.model} | "
            f"messages={memory_status['messages']} | entities={memory_status['entities']} | "
            f"summary={bool(memory_status['has_summary'])}"
        )

    def process(self, query: str, *, stream: bool | None = None, printer: Callable[[str], None] | None = None) -> AssistantResponse:
        classification = self.classify(query)
        payload = {
            "query": query,
            "history": self.memory.context_messages(),
        }

        handler = self.handlers[classification.request_type]
        use_stream = self.config.stream if stream is None else stream

        if use_stream and printer is not None:
            chunks: list[str] = []
            for chunk in handler.stream(payload):
                if chunk:
                    printer(chunk)
                    chunks.append(chunk)
            content = "".join(chunks).strip()
        else:
            content = str(handler.invoke(payload)).strip()

        self.memory.add_turn(query, content, summarizer=self._summarize_history)
        return AssistantResponse(
            content=content,
            request_type=classification.request_type,
            confidence=classification.confidence,
            tokens_used=self._estimate_tokens(query, content),
        )

    def classify(self, query: str) -> Classification:
        try:
            result = self.classifier_chain.invoke(query)
            if isinstance(result, Classification):
                return result
        except Exception:
            return Classification(
                request_type=RequestType.UNKNOWN,
                confidence=0.5,
                reasoning="Ошибка парсинга ответа модели",
            )
        return Classification(
            request_type=RequestType.UNKNOWN,
            confidence=0.5,
            reasoning="Ошибка парсинга ответа модели",
        )

    def _build_chains(self) -> None:
        classifier_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "[TASK:CLASSIFY_REQUEST]\n"
                    "Classify the user request into question, task, small_talk, complaint, or unknown. "
                    "Use the type definitions and examples below. Return only valid JSON "
                    "that follows the provided schema instructions.\n\n"
                    f"{CLASSIFIER_TYPE_DESCRIPTIONS}\n\n"
                    f"{CLASSIFIER_FEW_SHOTS}\n\n"
                    "{format_instructions}",
                ),
                ("human", "User query: {query}"),
            ]
        )

        classifier_inputs = RunnableParallel(
            query=RunnablePassthrough(),
            format_instructions=RunnableLambda(lambda _: self.classification_parser.get_format_instructions()),
        )
        self.classifier_chain = (
            classifier_inputs
            | classifier_prompt
            | self.classifier_model
            | self.classification_parser
        )

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "[TASK:SUMMARIZE_HISTORY]\n"
                    "Summarize the earlier conversation for future turns. Preserve durable facts "
                    "such as the user's name, favorite language, preferences, and location. "
                    "Keep the summary short.",
                ),
                (
                    "human",
                    "Existing summary:\n{existing_summary}\n\nConversation to compress:\n{conversation}",
                ),
            ]
        )
        self.summary_chain = summary_prompt | self.summary_model | StrOutputParser()

        self.handlers = {
            request_type: self._build_handler_chain(request_type)
            for request_type in RequestType
        }

    def _build_handler_chain(self, request_type: RequestType):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"[TASK:RESPOND][HANDLER:{request_type.value}]\n"
                    f"{CHARACTER_PROMPTS[self.config.character]}\n\n"
                    f"{HANDLER_INSTRUCTIONS[request_type]}\n"
                    "Use any durable user facts or conversation summary when they help answer accurately.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "User query: {query}"),
            ]
        )
        return prompt | self.response_model | StrOutputParser()

    def _summarize_history(self, existing_summary: str, transcript: str) -> str:
        return str(
            self.summary_chain.invoke(
                {
                    "existing_summary": existing_summary,
                    "conversation": transcript,
                }
            )
        ).strip()

    def _estimate_tokens(self, query: str, content: str) -> int:
        words = len((query + " " + content).split())
        return max(1, int(words * 1.35))

    def _sync_model_character(self, character: str) -> None:
        seen_ids: set[int] = set()
        for model in (self.response_model, self.classifier_model, self.summary_model):
            model_id = id(model)
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            if hasattr(model, "character"):
                setattr(model, "character", character)

    @staticmethod
    def _validate_character(character: str) -> None:
        if character not in CHARACTER_PROMPTS:
            available = ", ".join(sorted(CHARACTER_PROMPTS))
            raise ValueError(f"Unknown character '{character}'. Available characters: {available}")
