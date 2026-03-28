"""
Конфигурация LLM для примеров модуля 6.

Использует OpenRouter API. Настройте .env файл:
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Модель по умолчанию для примеров
DEFAULT_MODEL = "gpt-5.4-mini"


def get_llm(model: str = DEFAULT_MODEL, **kwargs) -> ChatOpenAI:
    """Создаёт ChatOpenAI с настройками из .env."""
    return ChatOpenAI(
        model=model,
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.environ["OPENROUTER_BASE_URL"],
        **kwargs,
    )


def check_api_key() -> bool:
    """Проверяет наличие API-ключа."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ВНИМАНИЕ: Установите OPENROUTER_API_KEY в .env файле")
        print("  OPENROUTER_API_KEY=sk-or-...")
        print("  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1")
        return False
    return True


if __name__ == "__main__":
    print(check_api_key())
