from __future__ import annotations

import argparse
from pathlib import Path

from .app import AssistantConfig, SmartAssistant
from .models import MemoryStrategy
from .personas import CHARACTER_PROMPTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI smart assistant built with LangChain.")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fallback-provider", default=None)
    parser.add_argument("--fallback-model", default=None)
    parser.add_argument("--character", default="friendly", choices=sorted(CHARACTER_PROMPTS))
    parser.add_argument("--memory", default=MemoryStrategy.BUFFER.value, choices=[m.value for m in MemoryStrategy])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--cache", default="none", choices=["none", "memory", "sqlite"])
    parser.add_argument("--cache-path", default="assistant_cache.sqlite3")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = AssistantConfig(
        provider=args.provider,
        model=args.model,
        fallback_provider=args.fallback_provider,
        fallback_model=args.fallback_model,
        character=args.character,
        memory_strategy=MemoryStrategy(args.memory),
        stream=args.stream,
        cache_backend=args.cache,
        cache_path=Path(args.cache_path),
        env_path=Path(args.env_file),
        base_url=args.base_url,
        api_key=args.api_key,
    )
    assistant = SmartAssistant(config=config)

    print("Smart Assistant")
    print(f"character={assistant.config.character} | memory={assistant.config.memory_strategy.value}")
    print("-" * 48)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not raw:
            continue

        if raw.startswith("/"):
            should_exit = _handle_command(assistant, raw)
            if should_exit:
                return 0
            continue

        if assistant.config.stream:
            streamed_chunks: list[str] = []

            def _printer(chunk: str) -> None:
                print(chunk, end="", flush=True)
                streamed_chunks.append(chunk)

            response = assistant.process(raw, stream=True, printer=_printer)
            print()
            print(f"[{response.request_type.value}]")
        else:
            response = assistant.process(raw)
            print(f"[{response.request_type.value}] {response.content}")

        print(f"confidence: {response.confidence:.2f} | tokens: ~{response.tokens_used}")


def _handle_command(assistant: SmartAssistant, raw: str) -> bool:
    parts = raw.split()
    command = parts[0]

    if command == "/quit":
        return True

    if command == "/help":
        print("/clear")
        print("/clear all")
        print("/character <friendly|professional|sarcastic|pirate>")
        print("/memory <buffer|summary>")
        print("/status")
        print("/help")
        print("/quit")
        return False

    if command == "/clear":
        include_entities = len(parts) > 1 and parts[1] == "all"
        assistant.clear(include_entities=include_entities)
        if include_entities:
            print("History and durable facts cleared.")
        else:
            print("History cleared. Durable facts were kept.")
        return False

    if command == "/character":
        if len(parts) != 2:
            print("Usage: /character <friendly|professional|sarcastic|pirate>")
            return False
        assistant.set_character(parts[1])
        print(f"Character changed to: {parts[1]}")
        return False

    if command == "/memory":
        if len(parts) != 2:
            print("Usage: /memory <buffer|summary>")
            return False
        assistant.set_memory_strategy(parts[1])
        print(f"Memory strategy changed to: {parts[1]}")
        return False

    if command == "/status":
        print(assistant.status())
        return False

    print("Unknown command. Use /help to see the available commands.")
    return False
