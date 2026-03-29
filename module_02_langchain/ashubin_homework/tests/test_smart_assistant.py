from __future__ import annotations

from smart_assistant_app.app import AssistantConfig, SmartAssistant
from smart_assistant_app.fake_model import RuleBasedChatModel
from smart_assistant_app.models import MemoryStrategy, RequestType


def build_assistant(memory_strategy: MemoryStrategy = MemoryStrategy.BUFFER) -> SmartAssistant:
    fake = RuleBasedChatModel(character="friendly")
    config = AssistantConfig(
        provider="fake",
        model="fake-model",
        character="friendly",
        memory_strategy=memory_strategy,
    )
    return SmartAssistant(
        config=config,
        model=fake,
        classifier_model=fake,
        summary_model=fake,
    )


def test_question_request_is_classified_and_answered() -> None:
    assistant = build_assistant()

    response = assistant.process("What is LCEL?")

    assert response.request_type is RequestType.QUESTION
    assert "LCEL" in response.content
    assert response.confidence > 0


def test_classifier_returns_literal_fallback_on_parse_error() -> None:
    class ExplodingChain:
        def invoke(self, query: str):
            raise ValueError("invalid json")

    assistant = build_assistant()
    assistant.classifier_chain = ExplodingChain()

    classification = assistant.classify("What is LCEL?")

    assert classification.request_type is RequestType.UNKNOWN
    assert classification.confidence == 0.5
    assert classification.reasoning == "Ошибка парсинга ответа модели"


def test_character_change_affects_response_style() -> None:
    assistant = build_assistant()
    assistant.set_character("pirate")

    response = assistant.process("What is LCEL?")

    assert response.request_type is RequestType.QUESTION
    assert response.content.startswith("Arrr, ")


def test_summary_memory_retains_user_facts_after_history_compression() -> None:
    assistant = build_assistant(memory_strategy=MemoryStrategy.SUMMARY)
    assistant.process("My name is Alex")
    assistant.process("My favorite language is Rust")

    for index in range(12):
        assistant.process(f"Write a short reply number {index}")

    response = assistant.process("What is my favorite language?")

    assert assistant.memory.summary
    assert "Rust" in response.content


def test_memory_remembers_name_and_favorite_language_before_summary() -> None:
    assistant = build_assistant()
    assistant.process("Hello, my name is Dasha")
    assistant.process("My favorite language is Python")

    response = assistant.process("What is my name and my favorite language?")

    assert response.request_type is RequestType.QUESTION
    assert "Dasha" in response.content
    assert "Python" in response.content
