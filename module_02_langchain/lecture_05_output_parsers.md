# Раздел 5: Структурированный вывод и Output Parsers

## Введение: Проблема свободного текста

Языковые модели генерируют текст. Просто текст — последовательность символов, слов, предложений. Для человека это идеально: мы читаем, понимаем, действуем. Но для программы свободный текст — проблема.

Представьте: вы просите модель классифицировать документ по категориям. Модель отвечает: "Основываясь на содержании документа, я бы отнёс его к категории 'Финансы', поскольку в нём обсуждаются вопросы бюджетирования и инвестиций." Прекрасный ответ для человека. Кошмар для программы.

Чтобы использовать результат, нужно извлечь строку "Финансы" из этого предложения. Написать регулярное выражение? Оно сломается, когда модель решит сформулировать ответ иначе. Использовать другую модель для извлечения? Дорого и добавляет новые точки отказа.

Проблема структурированного вывода — одна из центральных в инженерии LLM-приложений. Как заставить вероятностную, творческую систему выдавать предсказуемые, парсируемые результаты?

LangChain предлагает два взаимодополняющих подхода. Первый — output parsers: специализированные компоненты, которые извлекают структуру из свободного текста. Второй — структурированные API: возможности провайдеров генерировать JSON напрямую. Сегодня мы детально разберём оба.

---

## Часть 1: Философия парсинга вывода

### Почему JSON — lingua franca агентов

JSON стал стандартом обмена данными между агентами и остальным кодом. Не XML, не YAML, не собственные форматы — именно JSON. Почему?

Во-первых, JSON универсален. Любой язык программирования имеет встроенную или стандартную библиотеку для работы с JSON. Парсинг занимает микросекунды. Формат читаем и человеком, и машиной.

Во-вторых, JSON типизирован (хотя и слабо). Строки, числа, булевы значения, массивы, объекты — этого достаточно для большинства задач. В отличие от чистого текста, JSON несёт семантическую информацию о структуре данных.

В-третьих, JSON-схема позволяет валидировать структуру. Можно формально описать, какие поля обязательны, какие типы ожидаются, какие значения допустимы. Pydantic в Python превращает JSON-схему в мощный инструмент валидации.

Наконец, современные модели хорошо "знают" JSON. Они обучались на миллионах файлов конфигураций, API-ответов, датасетов. JSON — часть их "родного языка".

### Два пути к структуре

Получить структурированный вывод от модели можно двумя способами.

**Путь первый: инструкции в промпте.** Вы объясняете модели желаемый формат в системном сообщении или примерах. Модель генерирует текст, который (надеемся) соответствует формату. Затем output parser пытается извлечь структуру.

```python
prompt = """Классифицируй текст. Верни ТОЛЬКО JSON в формате:
{"category": "название категории", "confidence": число от 0 до 1}

Текст: {text}"""
```

Плюсы: работает с любой моделью, не требует специальных API. Минусы: модель может не следовать инструкциям, добавить лишний текст, допустить синтаксическую ошибку в JSON.

**Путь второй: структурированные API.** Некоторые провайдеры (OpenAI, Anthropic) предлагают специальные режимы, где модель гарантированно генерирует валидный JSON по заданной схеме. Механизм работы — ограниченная генерация (constrained decoding): модель может выбирать только токены, которые приводят к валидному JSON.

```python
model = ChatOpenAI(model="gpt-4o").with_structured_output(MySchema)
```

Плюсы: гарантированно валидный вывод, не нужен промпт-инжиниринг для формата. Минусы: работает не со всеми моделями, может быть медленнее.

LangChain поддерживает оба пути и позволяет комбинировать их.

---

## Часть 2: Базовые Output Parsers

### StrOutputParser: просто строка

Самый простой парсер — извлечение текстового содержимого из ответа модели:

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Вход: AIMessage(content="Python — язык программирования")
# Выход: "Python — язык программирования"
```

Кажется тривиальным, но этот парсер важен для унификации. Модель возвращает AIMessage, а вашему коду нужна строка. StrOutputParser — стандартный способ этого преобразования в цепочках LCEL.

### JsonOutputParser: извлечение JSON

JsonOutputParser пытается найти и распарсить JSON в ответе модели:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# Вход: AIMessage(content='{"name": "Alice", "age": 30}')
# Выход: {"name": "Alice", "age": 30}
```

Парсер достаточно умён, чтобы извлечь JSON даже из текста с пояснениями:

```python
# Вход: "Вот результат:\n```json\n{\"name\": \"Alice\"}\n```"
# Выход: {"name": "Alice"}
```

Он находит JSON-блок (в markdown code fence или просто фигурные скобки) и парсит его.

Можно указать Pydantic-схему для валидации:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = JsonOutputParser(pydantic_object=Person)

# Вход: '{"name": "Alice", "age": 30}'
# Выход: Person(name="Alice", age=30)
```

Теперь парсер не просто извлекает JSON, но и валидирует его против схемы. Если поле отсутствует или имеет неверный тип — исключение.

### PydanticOutputParser: сила типизации

PydanticOutputParser — развитие JsonOutputParser, полностью интегрированное с Pydantic:

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Название фильма")
    rating: float = Field(description="Оценка от 0 до 10")
    summary: str = Field(description="Краткое содержание")
    recommend: bool = Field(description="Рекомендуете ли вы фильм")

parser = PydanticOutputParser(pydantic_object=MovieReview)

# Важно: парсер генерирует инструкции для промпта!
print(parser.get_format_instructions())
```

Метод `get_format_instructions()` возвращает текст, объясняющий модели ожидаемый формат. Его нужно включить в промпт:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — кинокритик. {format_instructions}"),
    ("human", "Напиши рецензию на фильм: {movie}")
])

chain = prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser
```

Теперь цепочка:
1. Подставляет инструкции о формате в промпт
2. Модель генерирует JSON по этим инструкциям
3. Парсер валидирует JSON и возвращает Pydantic-объект

### CommaSeparatedListOutputParser: простые списки

Для простых случаев, когда нужен список строк:

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# Вход: "яблоко, банан, апельсин"
# Выход: ["яблоко", "банан", "апельсин"]
```

Парсер умеет работать с разными разделителями и пробелами.

---

## Часть 3: Структурированные API — with_structured_output

### Встроенная поддержка схем

Современные провайдеры предлагают API для структурированного вывода. LangChain абстрагирует это через метод `with_structured_output`:

```python
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    entities: list[str]
    sentiment: str
    topics: list[str]

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(ExtractedInfo)

result = structured_model.invoke("Анализируй: Apple представила новый iPhone...")
# result — это ExtractedInfo, не AIMessage
```

Никаких парсеров в цепочке, никаких инструкций в промпте. Модель гарантированно возвращает объект нужной структуры.

### Как это работает под капотом

OpenAI реализует structured output через параметр `response_format`:

```python
# Под капотом LangChain делает что-то вроде:
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ExtractedInfo",
            "schema": ExtractedInfo.model_json_schema()
        }
    }
)
```

Модель использует constrained decoding: на каждом шаге генерации из вокабуляра исключаются токены, которые приведут к невалидному JSON. Это гарантирует синтаксическую корректность.

Anthropic использует аналогичный механизм через Tool Use — модель "вызывает инструмент" с аргументами, соответствующими схеме.

### method="json_mode" vs method="function_calling"

`with_structured_output` поддерживает разные методы:

```python
# JSON mode — модель генерирует JSON напрямую
model.with_structured_output(Schema, method="json_mode")

# Function calling — модель вызывает "функцию" со схемой
model.with_structured_output(Schema, method="function_calling")
```

Function calling обычно надёжнее для сложных схем, JSON mode — быстрее для простых. LangChain выбирает метод автоматически, но можно указать явно.

### Strict mode

OpenAI поддерживает "strict mode", где схема соблюдается абсолютно:

```python
model.with_structured_output(Schema, strict=True)
```

Без strict mode модель может иногда генерировать невалидный JSON (редко, но случается). Со strict=True — гарантия. Но strict mode работает не со всеми функциями Pydantic (например, не поддерживаются кастомные валидаторы).

---

## Часть 4: Обработка ошибок парсинга

### Почему парсинг падает

Даже лучшие модели иногда генерируют невалидный вывод:

- Синтаксическая ошибка в JSON (незакрытая скобка, лишняя запятая)
- Неверный тип данных (строка вместо числа)
- Отсутствующее обязательное поле
- Лишний текст вокруг JSON

Когда парсер не может извлечь данные, он выбрасывает `OutputParserException`.

### OutputFixingParser: автоматическое исправление

OutputFixingParser оборачивает другой парсер и пытается исправить ошибки:

```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=MySchema)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=model)

# Если base_parser падает, fixing_parser:
# 1. Получает ошибку парсинга
# 2. Отправляет модели запрос "Исправь этот JSON: ..."
# 3. Пытается распарсить исправленный ответ
```

Это рекурсивный подход: используем модель для исправления ошибок модели. Дорого (дополнительный вызов API), но часто работает.

### RetryWithErrorOutputParser: повторная попытка с контекстом

Альтернативный подход — попросить модель переделать ответ, показав ошибку:

```python
from langchain.output_parsers import RetryWithErrorOutputParser

retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=model
)

# При ошибке отправляет модели:
# "Твой предыдущий ответ не прошёл валидацию.
# Ошибка: field 'age' must be integer
# Исправь ответ:"
```

Разница с OutputFixingParser: здесь модель получает контекст исходного промпта, а не только сломанный JSON.

### Fallback-стратегии

Иногда лучше вернуть частичный результат, чем упасть:

```python
from langchain_core.runnables import chain

@chain
def safe_parse(response):
    try:
        return parser.invoke(response)
    except OutputParserException as e:
        # Возвращаем дефолтные значения или частичный результат
        return MySchema(
            field1="unknown",
            field2=0,
            error=str(e)
        )
```

Или использовать `with_fallbacks`:

```python
main_chain = prompt | model | strict_parser
fallback_chain = prompt | model | lenient_parser

robust_chain = main_chain.with_fallbacks([fallback_chain])
```

---

## Часть 5: Продвинутые паттерны

### Вложенные схемы

Pydantic поддерживает вложенные модели, и парсеры это понимают:

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    city: str
    street: str
    building: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

parser = PydanticOutputParser(pydantic_object=Person)
```

Модель должна сгенерировать JSON с вложенной структурой. Инструкции формата автоматически включают описание вложенных объектов.

### Optional и Union типы

```python
from typing import Optional, Union

class Response(BaseModel):
    answer: str
    confidence: Optional[float] = None
    source: Union[str, List[str]] = None
```

Парсер понимает Optional (поле может отсутствовать) и Union (поле может быть одним из типов).

### Enum для ограниченных значений

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Analysis(BaseModel):
    text: str
    sentiment: Sentiment
```

Инструкции формата укажут модели допустимые значения. Парсер валидирует, что ответ соответствует Enum.

### Кастомные валидаторы

Pydantic позволяет добавлять кастомную логику валидации:

```python
from pydantic import BaseModel, field_validator

class Review(BaseModel):
    rating: float
    text: str

    @field_validator('rating')
    @classmethod
    def rating_in_range(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Rating must be between 0 and 10')
        return v
```

Если модель вернёт rating=15, валидатор выбросит исключение, и парсер упадёт (или OutputFixingParser попытается исправить).

### Streaming с парсерами

Некоторые парсеры поддерживают стриминг:

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# Парсер накапливает токены и выдаёт частичные результаты
async for partial in (model | parser).astream("..."):
    print(partial)  # Может быть неполный JSON на промежуточных шагах
```

JsonOutputParser умеет выдавать частично распарсенный объект по мере поступления токенов. Это полезно для больших JSON-ответов — пользователь видит данные ещё до завершения генерации.

---

## Часть 6: Практические рецепты

### Рецепт: Извлечение сущностей

```python
from pydantic import BaseModel
from typing import List

class Entity(BaseModel):
    text: str
    type: str  # PERSON, ORGANIZATION, LOCATION, etc.
    start: int
    end: int

class ExtractionResult(BaseModel):
    entities: List[Entity]

prompt = ChatPromptTemplate.from_messages([
    ("system", """Извлеки именованные сущности из текста.
Типы: PERSON, ORGANIZATION, LOCATION, DATE, MONEY.
Укажи позиции в тексте (индексы символов)."""),
    ("human", "{text}")
])

model = ChatOpenAI(model="gpt-4o")
structured_model = model.with_structured_output(ExtractionResult)

chain = prompt | structured_model
```

### Рецепт: Классификация с объяснением

```python
class ClassificationResult(BaseModel):
    category: str
    confidence: float
    reasoning: str  # Объяснение решения

prompt = ChatPromptTemplate.from_messages([
    ("system", """Классифицируй обращение клиента.
Категории: billing, technical, sales, complaint.
Объясни своё решение."""),
    ("human", "{message}")
])

chain = prompt | model.with_structured_output(ClassificationResult)

result = chain.invoke({"message": "Не могу войти в аккаунт"})
print(f"Категория: {result.category}")
print(f"Уверенность: {result.confidence}")
print(f"Почему: {result.reasoning}")
```

### Рецепт: Генерация с ограничениями

```python
from typing import Literal

class MarketingCopy(BaseModel):
    headline: str  # Field(max_length=60)
    body: str  # Field(max_length=300)
    cta: Literal["Купить", "Узнать больше", "Попробовать бесплатно"]
    tone: Literal["formal", "casual", "urgent"]

prompt = ChatPromptTemplate.from_messages([
    ("system", "Создай маркетинговый текст для продукта."),
    ("human", "Продукт: {product}\nЦелевая аудитория: {audience}")
])

chain = prompt | model.with_structured_output(MarketingCopy)
```

### Рецепт: Пошаговый анализ

```python
class Step(BaseModel):
    step_number: int
    action: str
    result: str

class Analysis(BaseModel):
    steps: List[Step]
    final_answer: str
    confidence: float

prompt = ChatPromptTemplate.from_messages([
    ("system", """Реши задачу пошагово.
Запиши каждый шаг рассуждения."""),
    ("human", "{problem}")
])

chain = prompt | model.with_structured_output(Analysis)
```

Это заставляет модель явно фиксировать цепочку рассуждений (Chain of Thought) в структурированном виде.

---

## Часть 7: Сравнение подходов

### Когда использовать парсеры

**PydanticOutputParser** подходит, когда:
- Работаете с моделями без поддержки structured output
- Нужны сложные кастомные валидаторы
- Хотите контролировать инструкции формата

### Когда использовать with_structured_output

**with_structured_output** подходит, когда:
- Работаете с OpenAI/Anthropic или другими поддерживающими провайдерами
- Критична надёжность парсинга (strict mode)
- Хотите минимизировать prompt engineering
- Нужна максимальная скорость

### Гибридный подход

Можно комбинировать:

```python
# Основная модель со structured output
main_chain = model.with_structured_output(Schema)

# Fallback для моделей без поддержки
fallback_chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | fallback_model | parser

robust_chain = main_chain.with_fallbacks([fallback_chain])
```

---

## Часть 8: Интеграция с LCEL

### Парсеры как Runnable

Все парсеры реализуют Runnable:

```python
chain = prompt | model | parser

# Эквивалентно:
result = parser.invoke(model.invoke(prompt.invoke(input)))
```

### Conditional parsing

Иногда разные ответы требуют разных парсеров:

```python
from langchain_core.runnables import RunnableBranch

def is_list_response(x):
    return x.content.startswith("[")

parser = RunnableBranch(
    (is_list_response, list_parser),
    single_item_parser  # default
)

chain = prompt | model | parser
```

### Валидация в середине цепочки

Парсер может стоять не только в конце:

```python
chain = (
    prompt
    | model
    | parser  # Извлекаем структуру
    | RunnableLambda(enrich_with_metadata)  # Обогащаем данные
    | next_model  # Передаём дальше
)
```

---

## Заключение: Структура — основа автоматизации

Структурированный вывод превращает языковую модель из инструмента для людей в компонент программной системы. JSON-ответы можно валидировать, передавать в базы данных, использовать для управления логикой приложения.

LangChain предлагает богатый арсенал инструментов: от простых парсеров для базовых задач до интеграции со structured output API для максимальной надёжности. Выбор инструмента зависит от требований к надёжности, поддержки провайдера, сложности схемы данных.

Ключевые принципы:
- Используйте Pydantic для описания схем — это даёт валидацию и документацию
- Предпочитайте with_structured_output когда возможно — это надёжнее
- Всегда имейте стратегию обработки ошибок — модели ненадёжны
- Тестируйте парсинг на edge cases — модели креативны в неожиданные моменты

---

## Вопросы для самопроверки

1. Почему JSON стал стандартом для структурированного вывода LLM-приложений?

2. Объясните разницу между JsonOutputParser и with_structured_output. Когда предпочтительнее каждый?

3. Как OutputFixingParser пытается исправить ошибки парсинга? Какие у этого подхода недостатки?

4. Почему strict mode в with_structured_output не поддерживает все возможности Pydantic?

5. Спроектируйте схему для извлечения информации о товаре из описания: название, цена, характеристики, отзывы.

---

## Ключевые термины

| Термин | Определение |
|--------|-------------|
| **Output Parser** | Компонент для извлечения структурированных данных из текстового ответа модели |
| **PydanticOutputParser** | Парсер, использующий Pydantic-схемы для валидации JSON |
| **with_structured_output** | Метод модели для гарантированной генерации JSON по схеме |
| **Constrained Decoding** | Техника ограничения генерации токенами, ведущими к валидному выводу |
| **OutputFixingParser** | Парсер-обёртка, использующий модель для исправления ошибок |
| **get_format_instructions** | Метод парсера, возвращающий инструкции формата для промпта |
| **Strict Mode** | Режим with_structured_output с гарантией соответствия схеме |
| **JSON Mode** | Режим генерации, где модель выдаёт только валидный JSON |
| **Function Calling** | Механизм структурированного вывода через "вызов функции" |
