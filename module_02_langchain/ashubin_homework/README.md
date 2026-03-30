# Умный ассистент с характером

## Описание проекта

В данной директории находится код для использования CLI-ассистента, который:

- классифицирует запрос пользователя;
- направляет запрос в нужный обработчик;
- поддерживает несколько "характеров" общения;
- хранит историю диалога;
- умеет работать с буферной и summary-памятью;
- поддерживает entity memory для долговременных фактов;
- может работать как с локальными, так и с API-моделями;
- поддерживает streaming, кэширование и fallback на запасную модель.

Основная точка входа: `smart_assistant.py`.

## Что реализовано

### Структурированные модели данных

Реализованы модели:

- `RequestType`
- `Classification`
- `AssistantResponse`

Файл: `smart_assistant_app/models.py`

Что хранится:

- тип запроса;
- уверенность классификатора;
- обоснование классификации;
- текст ответа;
- примерная оценка числа токенов.

### Классификатор запросов

Реализована LCEL-цепочка вида:

`вход -> prompt -> model -> PydanticOutputParser -> Classification`

Особенности:

- есть описание типов запросов;
- добавлены few-shot примеры;
- используется `PydanticOutputParser`;
- при ошибке парсинга возвращается fallback-классификация `unknown`.

Файл: `smart_assistant_app/app.py`

### Обработчики и роутинг

Для каждого типа запроса есть свой обработчик:

- `question`
- `task`
- `small_talk`
- `complaint`
- `unknown`

Роутинг выполнен через словарь обработчиков, ключом служит `classification.request_type`.

Файлы:

- `smart_assistant_app/app.py`
- `smart_assistant_app/personas.py`

### Характеры ассистента

Поддерживаются характеры:

- `friendly`
- `professional`
- `sarcastic`
- `pirate`

При смене характера обработчики пересоздаются с новыми системными промптами.

Файлы:

- `smart_assistant_app/personas.py`
- `smart_assistant_app/app.py`
- `smart_assistant_app/cli.py`

### Память диалога

Реализован `MemoryManager` с двумя стратегиями:

- `buffer` — хранит последние сообщения;
- `summary` — сжимает старую часть диалога в summary.

Дополнительно реализована entity memory:

- имя пользователя;
- любимый язык;
- город.

Файл: `smart_assistant_app/memory.py`

### CLI-интерфейс

Реализован интерактивный цикл с командами:

- `/clear`
- `/clear all`
- `/character <name>`
- `/memory <buffer|summary>`
- `/status`
- `/help`
- `/quit`

Файл: `smart_assistant_app/cli.py`

## Дополнительно реализовано

- потоковый вывод `streaming`;
- кэширование ответов;
- fallback на запасную модель;
- entity memory.

## Структура директории

```text
ashubin_homework/
├── README.md
├── pyproject.toml
├── .env.template
├── smart_assistant.py
├── smart_assistant_app/
│   ├── __init__.py
│   ├── app.py
│   ├── cli.py
│   ├── fake_model.py
│   ├── heuristics.py
│   ├── memory.py
│   ├── model_factory.py
│   ├── models.py
│   └── personas.py
└── tests/
    └── test_smart_assistant.py
```

## Как устроено решение

Высокоуровневая схема работы:

1. Пользователь вводит сообщение в CLI.
2. `SmartAssistant.process()` сначала вызывает классификатор.
3. Классификатор возвращает объект `Classification`.
4. По `request_type` выбирается нужный обработчик.
5. В обработчик передаются текущий запрос и история диалога.
6. Ответ сохраняется в память.
7. CLI печатает результат и метаданные.

### Основные компоненты

- `smart_assistant_app/app.py`
  Главная логика ассистента: конфигурация, классификация, роутинг, обработка, summary.

- `smart_assistant_app/cli.py`
  CLI-интерфейс, команды и аргументы командной строки.

- `smart_assistant_app/memory.py`
  История сообщений, summary-память и долговременные факты пользователя.

- `smart_assistant_app/model_factory.py`
  Создание моделей, настройка провайдеров, кэша и fallback-модели.

- `smart_assistant_app/fake_model.py`
  Локальная rule-based модель для офлайн-тестов.

## Поддерживаемые провайдеры моделей

Проект умеет работать не только с тестовой моделью, но и с реальными провайдерами.

Поддерживаются:

- `openrouter`
- `openai`
- `lmstudio`
- `ollama`
- `polzaai`
- `vsellm`
- `cloudru`
- `fake`

Важно: проект использует `ChatOpenAI`, поэтому для реальных провайдеров ожидается OpenAI-совместимый API.

## Настройка окружения

### Установить зависимости вручную

```bash
pip install -e .
pip install pytest
```

## Настройка `.env`

Шаблон находится в `.env.template`.

Пример для OpenRouter:

```env
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_api_key
```

Пример для Ollama:

```env
OLLAMA_HOST=http://localhost:11434
```

Пример для LM Studio:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=no_needed
```

## Запуск

### Базовый запуск

```bash
python3 smart_assistant.py
```

### Офлайн-режим с fake-моделью

```bash
python3 smart_assistant.py --provider fake --model fake-model
```

### Запуск через OpenRouter

```bash
python3 smart_assistant.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free
```

### Запуск локальной модели через Ollama

```bash
python3 smart_assistant.py --provider ollama --model qwen3:32b
```

### Запуск локальной модели через LM Studio

```bash
python3 smart_assistant.py --provider lmstudio --model qwen/qwen3-32b
```

### Запуск со стримингом

```bash
python3 smart_assistant.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free --stream
```

### Запуск с summary-памятью

```bash
python3 smart_assistant.py --memory summary
```

### Запуск с fallback-моделью

```bash
python3 smart_assistant.py \
  --provider openrouter \
  --model nvidia/nemotron-3-nano-30b-a3b:free \
  --fallback-provider openrouter \
  --fallback-model minimax/minimax-m2.5:free
```

## Аргументы CLI

Поддерживаются аргументы:

- `--provider`
- `--model`
- `--fallback-provider`
- `--fallback-model`
- `--character`
- `--memory`
- `--stream`
- `--cache`
- `--cache-path`
- `--env-file`
- `--base-url`
- `--api-key`

Пример:

```bash
python3 smart_assistant.py \
  --provider openrouter \
  --model nvidia/nemotron-3-nano-30b-a3b:free \
  --character professional \
  --memory summary \
  --stream \
  --cache sqlite \
  --cache-path .assistant_cache.sqlite3
```

## Команды внутри CLI

### `/help`

Показывает список доступных команд.

### `/status`

Показывает:

- текущий характер;
- стратегию памяти;
- провайдера;
- модель;
- количество сообщений в памяти;
- количество сохраненных entity-фактов;
- создано ли summary.

Пример:

```text
character=professional | memory=summary | provider=openrouter | model=nvidia/nemotron-3-nano-30b-a3b:free | messages=16 | entities=1 | summary=False
```

Что означает `summary=False`:

- режим `summary` включен;
- но summary еще не было создано;
- история пока не дошла до момента сжатия.

### `/clear`

Очищает историю сообщений, но сохраняет entity memory.

### `/clear all`

Очищает и историю, и долговременные факты пользователя.

### `/character <name>`

Меняет характер ассистента.

### `/memory <buffer|summary>`

Меняет стратегию памяти.

### `/quit`

Выход из программы.

## Память

### Buffer memory

В режиме `buffer` хранится ограниченное число последних сообщений.

Плюсы:

- простая реализация;
- дешевле по токенам;
- удобно для коротких диалогов.

Минусы:

- ранний контекст теряется.

### Summary memory

В режиме `summary` старая часть истории сжимается в краткое summary через LLM.

Плюсы:

- лучше подходит для длинных диалогов;
- позволяет сохранять ранний контекст.

Минусы:

- требует дополнительного вызова модели;
- summary создается не сразу, а только после достижения порога.

### Entity memory

Даже после `/clear` ассистент может помнить некоторые факты о пользователе:

- имя;
- любимый язык;
- город.

Эти факты удаляются только через `/clear all`.

## Streaming

При запуске с `--stream` ответ выводится по мере генерации.

Важно:

- классификатор перед ответом все равно отрабатывает полностью, поэтому небольшая пауза перед началом стриминга нормальна;
- в текущей реализации в stream-режиме текст ответа печатается один раз, после чего отдельно выводятся тип запроса и строка метрик.

## Кэширование ответов

Поддерживаются два режима кэша:

- `--cache memory`
- `--cache sqlite`

Примеры:

```bash
python3 smart_assistant.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free --cache memory
```

```bash
python3 smart_assistant.py --provider openrouter --model nvidia/nemotron-3-nano-30b-a3b:free --cache sqlite --cache-path .assistant_cache.sqlite3
```

Что важно понимать:

- кэш срабатывает только для одинаковых вызовов модели;
- ответы обработчиков в диалоге кэшируются реже, потому что меняется история сообщений;
- `sqlite` удобен, если нужен кэш между запусками.

## Fallback на запасную модель

Если основная модель недоступна или выбрасывает ошибку, можно использовать fallback:

```bash
python3 smart_assistant.py \
  --provider openrouter \
  --model primary-model \
  --fallback-provider openrouter \
  --fallback-model backup-model
```

Fallback строится через `with_fallbacks()` в `smart_assistant_app/model_factory.py`.

## Локальные модели

Проект умеет работать с локальными моделями, если они доступны через локальный сервер:

- `ollama`
- `lmstudio`

## Тесты

Тесты находятся в `tests/test_smart_assistant.py`.

В текущей версии оставлены 5 ключевых сценариев:

- базовый вопрос и корректный ответ;
- fallback классификатора при ошибке парсинга;
- смена характера ассистента;
- запоминание имени пользователя и любимого языка в обычном диалоге;
- сохранение фактов пользователя при `summary`-памяти;

Запуск:

```bash
pytest
```

## Известные ограничения

- часть системных промптов и служебных описаний сформулирована на английском;
- для OpenRouter возможны ошибки, связанные не с кодом, а с настройками приватности и guardrails в аккаунте;
- `GIGACHAT_CREDENTIALS` есть в `.env.template`, но отдельная поддержка GigaChat в коде не реализована.
