# Раздел 6: Память и Цепочки — от компонентов к приложениям

## Введение: Собираем пазл воедино

В прошлых пяти разделах были изучены компоненты LangChain по отдельности: интерфейс Runnable и LCEL, промпт-шаблоны, модели, парсеры вывода. Каждый компонент — как отдельная деталь конструктора. Добавив память, пора собрать из этих деталей работающие приложения.

LLM-приложение — это не единичный вызов модели, а многошаговый процесс с состоянием. Управление этим состоянием и есть суть production-ready систем.

---

## Часть 1: Проблема памяти в LLM

### Stateless-природа языковых моделей

Языковые модели по своей природе не имеют состояния. Каждый запрос к API — изолированная транзакция. Модель получает промпт, генерирует ответ, забывает всё. Следующий запрос — чистый лист.

Это контрастирует с человеческим восприятием диалога. Когда мы говорим "Расскажи подробнее", мы подразумеваем контекст предыдущего ответа. Для модели этого контекста не существует — если мы явно его не передали.

Решение кажется очевидным: передавать историю диалога с каждым запросом. И это действительно работает:

```python
messages = [
    SystemMessage(content="Ты — полезный ассистент."),
    HumanMessage(content="Меня зовут Алексей."),
    AIMessage(content="Приятно познакомиться, Алексей!"),
    HumanMessage(content="Как меня зовут?")
]

response = model.invoke(messages)
# "Вас зовут Алексей."
```

Модель "помнит" имя, потому что оно есть в переданной истории. Но это иллюзия памяти — мы просто включили всю историю в промпт.

### Ограничение контекстного окна

Проблема простого подхода "передавать всю историю" упирается в контекстное окно. 128K, 200K - Звучит много, но длинные диалоги быстро накапливаются.

Предположим, средний обмен (вопрос + ответ) занимает 500 токенов. После 200 обменов история займёт 100K токенов. А ещё нужно место для системного промпта, контекста из документов, нового вопроса, ответа.

Более того, вы платите за входные токены. Если каждый запрос включает всю историю, стоимость растёт линейно с длиной диалога. Сотый запрос стоит в сто раз дороже первого.

### Феномен "Lost in the Middle"

Даже если история помещается в окно, модели плохо работают с длинным контекстом. Исследования показали, что информация в начале и конце промпта используется лучше, чем в середине. Этот феномен назвали "Lost in the Middle".

Если важная деталь упомянута в середине длинной истории, модель может "забыть" её. Не потому что не видит — видит. Но attention-механизм распределяет "внимание" неравномерно.

Это означает, что даже технически возможная передача всей истории — не лучшая стратегия. Нужны умные подходы к управлению памятью.

---

## Часть 2: Стратегии управления памятью

### Buffer Memory: простое окно

Самая простая стратегия — хранить последние N сообщений. Как окно, скользящее по истории:

```python
from langchain_core.messages import trim_messages

history = get_full_history()  # Все сообщения

# Оставляем последние 10 сообщений
trimmed = trim_messages(
    history,
    max_tokens=2000,
    strategy="last",
    token_counter=model,
    include_system=True  # Системное сообщение сохраняем всегда
)
```

Параметр `include_system=True` важен: системное сообщение задаёт поведение бота и должно присутствовать всегда, независимо от обрезки.

Достоинства подхода: простота, предсказуемость, гарантированное укладывание в бюджет токенов.

Недостатки: ранний контекст теряется. Если пользователь представился в начале диалога, через 20 сообщений модель "забудет" его имя.

### Summary Memory: сжатие истории

Альтернатива — периодически суммаризировать историю:

```python
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Сделай краткое саммари диалога, сохрани ключевые факты."),
    ("human", "{history}")
])

summary_chain = summarize_prompt | model | StrOutputParser()

# Периодически сжимаем историю
if len(history) > 20:
    summary = summary_chain.invoke({"history": format_messages(history[:-5])})
    history = [SystemMessage(content=f"Саммари диалога: {summary}")] + history[-5:]
```

Идея: вместо хранения 100 сообщений храним саммари первых 95 + последние 5. Саммари занимает меньше токенов, но сохраняет ключевую информацию.

Достоинства: лучше сохраняет долгосрочный контекст, экономит токены.

Недостатки: саммари — lossy compression, детали теряются. Дополнительные вызовы API для суммаризации.

### Entity Memory: выделение сущностей

Более продвинутый подход — извлекать и хранить сущности из диалога:

```python
class EntityMemory(BaseModel):
    entities: dict[str, str] = {}  # название -> описание

# Извлекаем сущности из нового сообщения
extract_prompt = """Извлеки сущности (имена, места, даты, факты) из сообщения.
Текущие известные сущности: {current_entities}
Сообщение: {message}
Обнови сущности (добавь новые, обнови существующие):"""

# Храним компактный словарь сущностей вместо полной истории
entities = {
    "user_name": "Алексей",
    "user_city": "Москва",
    "discussed_topics": ["Python", "машинное обучение"]
}
```

При генерации ответа включаем сущности в промпт:

```python
prompt = f"""Известные факты о пользователе: {entities}
Последние сообщения: {recent_messages}
Вопрос: {question}"""
```

Достоинства: компактное хранение ключевой информации, не зависит от длины диалога.

Недостатки: сложность реализации, риск неправильного извлечения сущностей.

### Hybrid Approaches: комбинирование стратегий

На практике часто комбинируют подходы:

```python
memory_prompt = """Контекст диалога:
=== Саммари предыдущей беседы ===
{summary}

=== Известные факты ===
{entities}

=== Последние сообщения ===
{recent_messages}

Вопрос пользователя: {question}"""
```

Саммари сохраняет общую картину, сущности — ключевые факты, последние сообщения — непосредственный контекст.

---

## Часть 3: Реализация памяти в LCEL

### Почему память не встроена в Runnable

В ранних версиях LangChain память была частью цепочек. Вы создавали `ConversationChain`, передавали ему объект памяти, и цепочка автоматически загружала/сохраняла историю.

В современном LCEL память — явная часть вашего кода. Почему?

Во-первых, гибкость. Разные приложения требуют разных стратегий памяти. Чат-бот для поддержки нуждается в истории тикета. Персональный ассистент — в долгосрочной памяти о предпочтениях. Одноразовый Q&A-бот вообще не нуждается в памяти.

Во-вторых, явное лучше неявного. Когда память явная, вы контролируете, когда она загружается, как обрезается, где хранится. Неявная память — магия, которая может вести себя неожиданно.

В-третьих, интеграция с внешними системами. Продакшн-приложения часто хранят историю в базе данных. Явное управление памятью упрощает эту интеграцию.

### Паттерн: история как входной параметр

Стандартный паттерн LCEL — передавать историю как часть входа:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — полезный ассистент."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

# Использование
history = load_history(session_id)
response = chain.invoke({"history": history, "input": user_message})

# Сохраняем новые сообщения
history.extend([
    HumanMessage(content=user_message),
    AIMessage(content=response)
])
save_history(session_id, history)
```

Логика загрузки и сохранения — снаружи цепочки. Цепочка просто использует то, что ей передали.

### RunnableWithMessageHistory: обёртка для удобства

LangChain предоставляет обёртку, которая автоматизирует работу с историей:

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Хранилище историй по session_id
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain = prompt | model | StrOutputParser()

with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Использование
response = with_history.invoke(
    {"input": "Привет!"},
    config={"configurable": {"session_id": "user_123"}}
)
```

RunnableWithMessageHistory:
1. Извлекает session_id из конфигурации
2. Загружает историю через get_session_history
3. Передаёт историю в цепочку
4. После выполнения сохраняет новые сообщения

Это удобно для типовых случаев, но менее гибко, чем явное управление.

### Персистентные хранилища

В продакшне история хранится не в памяти Python, а в базе данных:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")
```

Или в PostgreSQL:

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string="postgresql://user:pass@localhost/db",
        session_id=session_id
    )
```

Эти хранилища сохраняют историю между перезапусками приложения и позволяют масштабировать на несколько серверов.

---

## Часть 4: Классические цепочки (Legacy Chains)

### Эволюция от Chains к LCEL

В ранних версиях LangChain основной абстракцией были Chains — классы для типовых задач. ConversationChain для диалогов, RetrievalQA для RAG, SQLDatabaseChain для работы с базами данных.

Эти цепочки удобны: создаёшь объект, вызываешь метод, получаешь результат. Но они были негибкими. Хотите изменить промпт? Наследуйтесь и переопределяйте. Хотите добавить шаг обработки? Создавайте новую цепочку.

LCEL решил проблему гибкости: compose что угодно из простых компонентов. Но потерял удобство готовых решений.

Современный LangChain использует гибридный подход. Legacy Chains всё ещё существуют для совместимости, но рекомендуется строить приложения через LCEL с использованием готовых паттернов.

### LLMChain: базовая цепочка

LLMChain — простейшая цепочка: промпт + модель + (опционально) парсер.

```python
from langchain.chains import LLMChain

chain = LLMChain(
    llm=model,
    prompt=prompt,
    output_parser=parser
)

result = chain.invoke({"topic": "Python"})
```

LCEL-эквивалент:

```python
chain = prompt | model | parser
result = chain.invoke({"topic": "Python"})
```

LCEL-версия короче, гибче и современнее. LLMChain остаётся для совместимости с legacy-кодом.

### ConversationChain: диалог с памятью

ConversationChain — цепочка для чат-ботов со встроенной памятью:

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

chain = ConversationChain(
    llm=model,
    memory=memory
)

print(chain.run("Привет! Меня зовут Алексей."))
print(chain.run("Как меня зовут?"))
```

Память автоматически накапливается и передаётся модели. Удобно для прототипов.

LCEL-эквивалент требует явного управления историей (как мы разбирали выше), но даёт полный контроль.

### SequentialChain: последовательность шагов

Для многошаговых процессов:

```python
from langchain.chains import SequentialChain, LLMChain

# Первый шаг: перевод
translate_chain = LLMChain(
    llm=model,
    prompt=translate_prompt,
    output_key="translation"
)

# Второй шаг: анализ перевода
analyze_chain = LLMChain(
    llm=model,
    prompt=analyze_prompt,
    output_key="analysis"
)

overall_chain = SequentialChain(
    chains=[translate_chain, analyze_chain],
    input_variables=["text"],
    output_variables=["translation", "analysis"]
)
```

LCEL-эквивалент:

```python
chain = (
    RunnablePassthrough.assign(translation=translate_chain)
    | RunnablePassthrough.assign(analysis=analyze_chain)
)
```

Снова LCEL гибче и выразительнее.

---

## Часть 5: Router Chains — условная маршрутизация

### Проблема разных типов запросов

Реальные приложения получают разные типы запросов. Чат-бот поддержки должен по-разному обрабатывать технические вопросы, жалобы и запросы о статусе заказа. Универсальный промпт плохо справляется со всем.

Router Chains решают эту проблему: сначала классифицируем запрос, затем направляем к специализированному обработчику.

### MultiPromptChain

```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

# Определяем специализированные промпты
prompt_infos = [
    {
        "name": "physics",
        "description": "Вопросы о физике",
        "prompt_template": physics_prompt
    },
    {
        "name": "math",
        "description": "Математические задачи",
        "prompt_template": math_prompt
    },
    {
        "name": "general",
        "description": "Общие вопросы",
        "prompt_template": general_prompt
    }
]

chain = MultiPromptChain.from_prompts(model, prompt_infos)

# Цепочка сама определяет, какой промпт использовать
result = chain.invoke({"input": "Чему равен интеграл от x^2?"})
```

Под капотом MultiPromptChain:
1. Вызывает router-модель для классификации
2. Выбирает соответствующий промпт
3. Генерирует ответ

### LCEL-подход к маршрутизации

В LCEL маршрутизация реализуется через RunnableBranch или семантическую классификацию:

```python
from langchain_core.runnables import RunnableBranch

# Классификатор
classifier = classification_prompt | model.with_structured_output(Classification)

# Специализированные цепочки
physics_chain = physics_prompt | model | parser
math_chain = math_prompt | model | parser
general_chain = general_prompt | model | parser

# Маршрутизатор
router = RunnableBranch(
    (lambda x: x["category"] == "physics", physics_chain),
    (lambda x: x["category"] == "math", math_chain),
    general_chain
)

# Полная цепочка
full_chain = (
    RunnablePassthrough.assign(category=classifier)
    | router
)
```

Более явно, более гибко, полный контроль над логикой.

---

## Часть 6: Практические паттерны

### Паттерн: Чат-бот с RAG и памятью

Собираем всё вместе — бот, который помнит диалог и использует базу знаний:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages

# Промпт с историей и контекстом
prompt = ChatPromptTemplate.from_messages([
    ("system", """Ты — ассистент компании TechCorp.
Используй контекст для ответа. Если не знаешь — скажи честно.
Контекст: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Функция обрезки истории
def trim_history(data):
    data["history"] = trim_messages(
        data["history"],
        max_tokens=1000,
        strategy="last",
        token_counter=model
    )
    return data

# Цепочка
chain = (
    RunnableLambda(trim_history)
    | RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["input"])
    )
    | prompt
    | model
    | StrOutputParser()
)

# Использование
def chat(session_id: str, message: str):
    history = load_history(session_id)

    response = chain.invoke({
        "input": message,
        "history": history
    })

    history.extend([
        HumanMessage(content=message),
        AIMessage(content=response)
    ])
    save_history(session_id, history)

    return response
```

### Паттерн: Мультиагентная обработка

Разные агенты обрабатывают разные аспекты запроса:

```python
# Агент анализа тональности
sentiment_chain = sentiment_prompt | model.with_structured_output(SentimentResult)

# Агент извлечения сущностей
entities_chain = entities_prompt | model.with_structured_output(EntitiesResult)

# Агент генерации ответа
response_chain = response_prompt | model | StrOutputParser()

# Параллельный анализ + последовательная генерация
full_chain = (
    RunnableParallel({
        "sentiment": sentiment_chain,
        "entities": entities_chain,
        "input": RunnablePassthrough()
    })
    | response_chain
)
```

Анализ тональности и извлечение сущностей происходят параллельно, затем результаты передаются агенту генерации.

### Паттерн: Самокорректирующаяся генерация

Модель генерирует, затем проверяет себя:

```python
# Генерация
generate_chain = generate_prompt | model | StrOutputParser()

# Проверка
check_chain = check_prompt | model.with_structured_output(CheckResult)

# Исправление при необходимости
fix_chain = fix_prompt | model | StrOutputParser()

def self_correcting_generate(input_data, max_iterations=3):
    result = generate_chain.invoke(input_data)

    for i in range(max_iterations):
        check = check_chain.invoke({"content": result, **input_data})

        if check.is_valid:
            return result

        result = fix_chain.invoke({
            "content": result,
            "issues": check.issues,
            **input_data
        })

    return result  # Лучшее, что получилось
```

Этот паттерн повышает качество за счёт дополнительных итераций.

---

## Часть 7: Миграция на LangGraph

### Ограничения LCEL

LCEL прекрасен для DAG (направленных ациклических графов). Но реальные агенты требуют циклов: попытаться действие → проверить результат → если ошибка, попробовать снова.

LCEL поддерживает циклы только через Python-код (while loops внутри RunnableLambda). Это работает, но теряет декларативность.

### LangGraph: следующий уровень

LangGraph — библиотека для построения циклических графов состояний. Тут важно понимать, куда эволюционирует экосистема.

```python
from langgraph.graph import StateGraph

# Определяем состояние
class State(TypedDict):
    messages: list[BaseMessage]
    attempts: int

# Определяем узлы
def should_retry(state):
    return state["attempts"] < 3 and has_error(state)

# Строим граф
graph = StateGraph(State)
graph.add_node("generate", generate_node)
graph.add_node("check", check_node)
graph.add_edge("generate", "check")
graph.add_conditional_edges("check", should_retry, {
    True: "generate",  # Цикл!
    False: END
})
```

LangGraph декларативно описывает циклы, автоматически управляет состоянием, поддерживает persistence и human-in-the-loop.

### Когда использовать что

**LCEL** подходит для:
- Линейных пайплайнов (промпт → модель → парсер)
- Ветвления без циклов
- Простых приложений

**LangGraph** нужен для:
- Агентов с циклами (попытка → проверка → исправление)
- Сложных многоагентных систем
- Приложений с persistence и recovery

Это не взаимоисключающие инструменты. LangGraph использует LCEL-примитивы внутри узлов. Вы можете начать с LCEL и мигрировать на LangGraph, когда понадобятся циклы.

---

## Часть 8: Best Practices для памяти и цепочек

### Управление памятью

1. **Выбирайте стратегию под задачу.** Короткие сессии — buffer memory. Длинные диалоги — summary + entities. Персональные ассистенты — долгосрочное хранение.

2. **Всегда устанавливайте лимиты.** Без ограничений история съест контекстное окно и бюджет.

3. **Сохраняйте системное сообщение.** При обрезке истории системный промпт должен сохраняться.

4. **Используйте персистентное хранилище.** В продакшне — база данных, не память процесса.

5. **Обрабатывайте длинные сообщения.** Одно огромное сообщение может занять большую часть контекста.

### Проектирование цепочек

1. **Начинайте с LCEL.** Легаси-цепочки — для совместимости, не для нового кода.

2. **Декомпозируйте.** Большие цепочки разбивайте на переиспользуемые компоненты.

3. **Явное лучше неявного.** Лучше явно передать параметр, чем полагаться на магию.

4. **Тестируйте компоненты отдельно.** Каждый Runnable можно протестировать изолированно.

5. **Логируйте промежуточные шаги.** Callbacks и трейсинг — ваши друзья при отладке.

### Производительность

1. **Параллелизуйте независимые шаги.** RunnableParallel для операций без зависимостей.

2. **Батчите где возможно.** Один вызов batch() эффективнее N вызовов invoke().

3. **Кэшируйте повторяющиеся запросы.** Одинаковые промпты — одинаковые ответы.

4. **Стримьте для UX.** Пользователь не должен ждать полного ответа.

---

## Заключение: От компонентов к системам

LangChain — это не магия, а набор продуманных абстракций. Runnable унифицирует компоненты. Pipe-оператор соединяет их. LCEL делает код декларативным и читаемым. Но LangChain — это инструмент, не решение. Он не делает вашего бота умнее. Он не выбирает правильный промпт. Он не определяет архитектуру приложения. Всё это — ваша работа как инженера.

---

## Вопросы для самопроверки

1. Почему языковые модели не имеют "настоящей" памяти? Как мы создаём иллюзию памяти?

2. Сравните Buffer Memory и Summary Memory. Когда предпочтительнее каждая стратегия?

3. Объясните, почему LCEL не имеет встроенного управления памятью. Как это соотносится с принципом "явное лучше неявного"?

4. Когда стоит использовать legacy Chains, а когда — чистый LCEL?

5. Спроектируйте систему памяти для персонального ассистента, который должен помнить предпочтения пользователя месяцами.

---

## Ключевые термины

| Термин | Определение |
|--------|-------------|
| **Buffer Memory** | Стратегия хранения последних N сообщений диалога |
| **Summary Memory** | Стратегия сжатия истории в краткое саммари |
| **Entity Memory** | Хранение извлечённых сущностей вместо полной истории |
| **Lost in the Middle** | Феномен плохого внимания к информации в середине контекста |
| **RunnableWithMessageHistory** | Обёртка для автоматического управления историей диалога |
| **ChatMessageHistory** | Интерфейс для хранилищ истории сообщений |
| **LLMChain** | Legacy-цепочка: промпт + модель + парсер |
| **ConversationChain** | Legacy-цепочка для диалогов со встроенной памятью |
| **Router Chain** | Цепочка с условной маршрутизацией запросов |
| **trim_messages** | Функция для обрезки истории до заданного лимита токенов |
