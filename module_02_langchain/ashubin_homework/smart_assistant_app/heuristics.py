from __future__ import annotations

import re
from collections.abc import Mapping

from .models import Classification, RequestType


QUESTION_WORDS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "can you",
    "could you",
    "would you",
    "что",
    "как",
    "почему",
    "зачем",
    "когда",
    "где",
    "кто",
    "какой",
    "какая",
)

GREETING_MARKERS = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "привет",
    "здравствуй",
    "здравствуйте",
    "доброе утро",
    "добрый вечер",
)

COMPLAINT_MARKERS = (
    "terrible",
    "awful",
    "broken",
    "this sucks",
    "hate",
    "ужас",
    "плохо",
    "не работает",
    "сломано",
    "бесит",
)

TASK_MARKERS = (
    "write",
    "tell me",
    "create",
    "generate",
    "draft",
    "summarize",
    "напиши",
    "создай",
    "сгенерируй",
    "расскажи",
    "сделай",
)


def heuristic_classify(query: str) -> Classification:
    text = query.strip()
    lower = text.lower()

    if _looks_like_gibberish(lower):
        return Classification(
            request_type=RequestType.UNKNOWN,
            confidence=0.9,
            reasoning="The message looks like gibberish or lacks a clear intent.",
        )

    if any(marker in lower for marker in COMPLAINT_MARKERS):
        return Classification(
            request_type=RequestType.COMPLAINT,
            confidence=0.92,
            reasoning="Complaint language is present.",
        )

    if lower.startswith(("/help", "/clear", "/status", "/quit", "/character", "/memory")):
        return Classification(
            request_type=RequestType.TASK,
            confidence=0.8,
            reasoning="This looks like an imperative command-like input.",
        )

    if any(marker in lower for marker in GREETING_MARKERS) or "меня зовут" in lower or "my name is" in lower:
        return Classification(
            request_type=RequestType.SMALL_TALK,
            confidence=0.9,
            reasoning="Greeting or introduction detected.",
        )

    if any(lower.startswith(marker) for marker in TASK_MARKERS):
        return Classification(
            request_type=RequestType.TASK,
            confidence=0.86,
            reasoning="Imperative phrasing suggests a task request.",
        )

    if text.endswith("?") or any(lower.startswith(word) for word in QUESTION_WORDS):
        return Classification(
            request_type=RequestType.QUESTION,
            confidence=0.88,
            reasoning="Question wording or punctuation detected.",
        )

    return Classification(
        request_type=RequestType.TASK,
        confidence=0.62,
        reasoning="Defaulted to task because the request is intelligible but not clearly a question.",
    )


def _looks_like_gibberish(lower: str) -> bool:
    normalized = re.sub(r"[^a-zа-яё]", "", lower)
    if len(normalized) < 4:
        return False
    if " " in lower:
        return False
    vowels = set("aeiouyауоыиэяюёе")
    vowel_count = sum(char in vowels for char in normalized)
    return vowel_count <= 1


def simple_summary(existing_summary: str, transcript: str) -> str:
    lines = []
    if existing_summary.strip():
        lines.append(existing_summary.strip())

    extracted = []
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Human:"):
            candidate = line.removeprefix("Human:").strip()
            if any(
                marker in candidate.lower()
                for marker in ("меня зовут", "my name is", "любимый язык", "favorite language", "живу в", "live in")
            ):
                extracted.append(candidate)

    if extracted:
        lines.append("Remembered user facts: " + "; ".join(extracted[-4:]))
    else:
        compact = " ".join(part.strip() for part in transcript.splitlines() if part.strip())
        if compact:
            lines.append(compact[:320])

    return "\n".join(lines).strip()


def answer_from_entities(query: str, entities: Mapping[str, str]) -> str | None:
    lower = query.lower()
    wants_name = "name" in lower or "зовут" in lower or "имя" in lower
    wants_language = "favorite language" in lower or "любимый язык" in lower
    wants_city = "live" in lower or "живу" in lower or "city" in lower or "город" in lower

    if wants_name and wants_language and entities.get("name") and entities.get("favorite_language"):
        return f"Your name is {entities['name']}, and your favorite language is {entities['favorite_language']}."
    if wants_name and entities.get("name"):
        return f"Your name is {entities['name']}."
    if wants_language and entities.get("favorite_language"):
        return f"Your favorite language is {entities['favorite_language']}."
    if wants_city and entities.get("city"):
        return f"You live in {entities['city']}."
    return None


def generate_handler_reply(
    request_type: RequestType,
    query: str,
    entities: Mapping[str, str],
    character: str,
) -> str:
    memory_answer = answer_from_entities(query, entities)
    prefix = {
        "friendly": "",
        "professional": "",
        "sarcastic": "Obviously, ",
        "pirate": "Arrr, ",
    }.get(character, "")

    if memory_answer:
        return f"{prefix}{memory_answer}"

    lower = query.lower()

    if request_type is RequestType.SMALL_TALK:
        if entities.get("name") and ("hello" in lower or "привет" in lower or "меня зовут" in lower):
            return f"{prefix}Nice to meet you, {entities['name']}. How can I help?"
        if entities.get("favorite_language") and ("любимый язык" in lower or "favorite language" in lower):
            return f"{prefix}{entities['favorite_language']} is a great choice."
        return f"{prefix}Happy to chat. What would you like to talk about?"

    if request_type is RequestType.COMPLAINT:
        return (
            f"{prefix}I can see why that is frustrating. Tell me what went wrong and "
            "I will help you troubleshoot it."
        )

    if request_type is RequestType.TASK:
        if "poem" in lower or "стих" in lower:
            return (
                f"{prefix}Here is a short poem:\n"
                "Code in the terminal glow,\n"
                "Small chains guide the thoughts we sow,\n"
                "Step by step, the answers grow."
            )
        if "joke" in lower or "анекдот" in lower:
            return f"{prefix}Why do prompts love structure? Because chaos does not parse."
        return f"{prefix}I can help with that. Give me the exact format or outcome you want."

    if request_type is RequestType.QUESTION:
        if "lcel" in lower:
            return (
                f"{prefix}LCEL is the LangChain Expression Language. It lets you compose "
                "prompt, model, parser, and routing steps with runnable pipelines."
            )
        if "langchain" in lower:
            return (
                f"{prefix}LangChain is a framework for orchestrating LLM applications "
                "with prompts, models, memory, tools, and execution chains."
            )
        return f"{prefix}Here is the best concise answer I can give based on the available context."

    return f"{prefix}I did not fully understand the request. Could you rephrase it?"
