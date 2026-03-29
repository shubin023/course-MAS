from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI

from .fake_model import RuleBasedChatModel


PROVIDER_ENV = {
    "openrouter": ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY"),
    "lmstudio": ("LMSTUDIO_BASE_URL", "LMSTUDIO_API_KEY"),
    "ollama": ("OLLAMA_HOST", None),
    "polzaai": ("POLZAAI_BASE_URL", "POLZAAI_API_KEY"),
    "vsellm": ("VSELLM_BASE_URL", "VSELLM_API_KEY"),
    "cloudru": ("CLOUDRU_BASE_URL", "CLOUDRU_API_KEY"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "fake": (None, None),
}


class ProviderConfigError(RuntimeError):
    pass


def load_project_env(env_path: Path | None = None) -> None:
    if env_path and env_path.exists():
        load_dotenv(env_path)
        return

    candidate = Path(".env")
    if candidate.exists():
        load_dotenv(candidate)


def configure_cache(cache_backend: str, cache_path: Path | None = None) -> None:
    if cache_backend == "memory":
        set_llm_cache(InMemoryCache())
    elif cache_backend == "sqlite":
        path = cache_path or Path("assistant_cache.sqlite3")
        set_llm_cache(SQLiteCache(database_path=str(path)))


def build_chat_model(
    provider: str,
    model: str,
    *,
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    streaming: bool = False,
    character: str = "friendly",
):
    if provider == "fake":
        return RuleBasedChatModel(character=character)

    resolved_base_url, resolved_api_key = _resolve_provider(provider, base_url, api_key)
    kwargs = {
        "model": model,
        "temperature": temperature,
        "streaming": streaming,
    }
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key

    return ChatOpenAI(**kwargs)


def build_resilient_model(
    provider: str,
    model: str,
    *,
    fallback_model: str | None = None,
    fallback_provider: str | None = None,
    temperature: float = 0.2,
    base_url: str | None = None,
    api_key: str | None = None,
    streaming: bool = False,
    character: str = "friendly",
):
    primary = build_chat_model(
        provider,
        model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        character=character,
    )
    if not fallback_model:
        return primary

    fallback = build_chat_model(
        fallback_provider or provider,
        fallback_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        streaming=streaming,
        character=character,
    )
    return primary.with_fallbacks([fallback])


def _resolve_provider(
    provider: str,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str | None, str | None]:
    if provider not in PROVIDER_ENV:
        raise ProviderConfigError(f"Unsupported provider: {provider}")

    env_base_key, env_api_key = PROVIDER_ENV[provider]
    resolved_base_url = base_url or (os.getenv(env_base_key) if env_base_key else None)
    resolved_api_key = api_key or (os.getenv(env_api_key) if env_api_key else None)

    if provider == "ollama":
        if not resolved_base_url:
            raise ProviderConfigError("OLLAMA_HOST is required for the ollama provider.")
        if not resolved_base_url.rstrip("/").endswith("/v1"):
            resolved_base_url = resolved_base_url.rstrip("/") + "/v1"
        resolved_api_key = resolved_api_key or "ollama"

    if provider != "lmstudio" and provider not in {"ollama"} and provider != "fake":
        if not resolved_api_key:
            raise ProviderConfigError(
                f"An API key is required for provider '{provider}'. "
                "Set it via the environment or pass --api-key."
            )

    return resolved_base_url, resolved_api_key
