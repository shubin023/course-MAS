# Раздел 5: Продвинутый поиск — Beyond Naive Retrieval

## Введение: Когда простого поиска недостаточно

Представьте детектива, расследующего сложное дело. Ему недостаточно просто открыть картотеку и найти папки с похожими названиями. Он формулирует гипотезы, переформулирует вопросы, ищет по разным базам данных, связывает факты из разных источников, отсеивает ложные следы. Каждая улика может привести к новым вопросам, а ответ часто скрывается не в одном документе, а в сопоставлении нескольких.

Именно так должен работать продвинутый retrieval в RAG-системах. Наивный подход — "преобразуй вопрос в вектор, найди похожие" — работает для простых случаев. Но реальные вопросы пользователей часто многослойны, неоднозначны, требуют информации из нескольких источников.

---

## Часть 1: Гибридный поиск — союз семантики и лексики

### Ахиллесова пята векторного поиска

Векторный поиск прекрасно справляется с перефразированиями. "Как оформить отпуск?" найдёт документ "Процедура предоставления ежегодного оплачиваемого отдыха" — разные слова, но близкие векторы.

Однако у семантического поиска есть слепое пятно: точные термины, имена, коды.

Пример: пользователь спрашивает "Требования ISO 27001 к управлению инцидентами". Векторный поиск может найти документы про "управление инцидентами безопасности" в целом, но не обязательно те, где упомянут именно ISO 27001. Потому что "27001" — это не семантическое понятие, это идентификатор, и его близость к другим числам (27002, 27017) в векторном пространстве случайна.

Другие примеры проблемных запросов:
- "Ошибка NullPointerException в модуле AuthService" — точные технические термины
- "Контракт с ООО Ромашка от 15.03.2024" — имена собственные и даты
- "Статья 128 ТК РФ" — юридические ссылки

### BM25: классика не стареет

BM25 (Best Matching 25) — алгоритм ранжирования, основанный на частоте слов. Он существует с 1990-х годов, но до сих пор удивительно эффективен для точного поиска.

Идея BM25: документ релевантен запросу, если:
1. Слова из запроса часто встречаются в документе (TF — term frequency)
2. Эти слова редкие в коллекции в целом (IDF — inverse document frequency)
3. Документ не слишком длинный (нормализация по длине)

Формула BM25:

$$\text{score}(D, Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$$

где:
- $f(q_i, D)$ — частота слова $q_i$ в документе $D$
- $|D|$ — длина документа
- $avgdl$ — средняя длина документов
- $k_1$ и $b$ — параметры (типично 1.2 и 0.75)

Не пугайтесь формулы — понимать её досконально не обязательно. Важно помнить: BM25 идеально находит документы, содержащие точные слова из запроса.

### Объединение: Hybrid Search

Гибридный поиск комбинирует векторное и лексическое ранжирование. Каждый подход возвращает свой список кандидатов, затем списки объединяются.

Методы объединения:

**Linear Combination:**
$$\text{score}_{hybrid} = \alpha \cdot \text{score}_{vector} + (1-\alpha) \cdot \text{score}_{bm25}$$

Параметр α (0-1) контролирует баланс. α=0.7 означает "больше доверяем семантике".

**Reciprocal Rank Fusion (RRF):** — не зависит от масштаба скоров, только от позиций.

**Convex Combination с нормализацией:** Сначала нормализуем скоры (min-max или z-score), затем комбинируем.

### Реализация гибридного поиска

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

# Подготовка данных
documents = [...]  # Список Document объектов

# Создаём BM25 retriever
bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=10  # Топ-10 по BM25
)

# Создаём vector retriever
vectorstore = Chroma.from_documents(documents, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Комбинируем
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% BM25, 70% vector
)

# Использование
results = ensemble_retriever.invoke("Требования ISO 27001")
```

### Выбор весов: эмпирика важнее теории

Оптимальные веса зависят от данных и типа запросов. Общие рекомендации:

- **Технические документы с много кодов и терминов:** больше вес BM25 (0.5/0.5 или даже 0.6/0.4)
- **Разговорные запросы к описательным текстам:** больше вес вектору (0.3/0.7)
- **Смешанные запросы:** начните с 0.4/0.6, экспериментируйте

**Важно:** Проводите A/B тестирование на реальных запросах. Теоретические предположения часто не совпадают с практикой.

---

## Часть 2: Multi-Query Retrieval — один вопрос, много поисков

### Проблема: пользователь формулирует неоптимально

Пользователь спрашивает: "Как мне поступить, если заболею во время командировки?"

Этот вопрос затрагивает несколько тем:
- Больничные листы и их оформление
- Правила командировок
- Страховка при выездах
- Возврат билетов и отмена бронирований

Один векторный запрос может найти документы про одну из тем, но вряд ли про все.

### Решение: генерация множества запросов

Multi-Query Retriever использует LLM для генерации нескольких вариантов запроса:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

# Базовый retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Multi-query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    # Опционально: кастомный промпт для генерации запросов
)

# При вызове LLM генерирует варианты, поиск выполняется для каждого
results = multi_query_retriever.invoke(
    "Как мне поступить, если заболею во время командировки?"
)
```

LLM может сгенерировать:
- "Оформление больничного листа в командировке"
- "Политика компании при болезни сотрудника в поездке"
- "Страхование здоровья в рабочих поездках"
- "Отмена командировки по болезни"

Поиск выполняется для каждого варианта, результаты объединяются и дедуплицируются.

### Кастомизация генерации запросов

```python
from langchain_core.prompts import PromptTemplate

# Кастомный промпт
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Ты — помощник по генерации поисковых запросов.
Твоя задача — сгенерировать 4 разные версии данного вопроса
для поиска релевантных документов в корпоративной базе знаний.

Оригинальный вопрос: {question}

Сгенерируй 4 альтернативных запроса (по одному на строку):"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=QUERY_PROMPT
)
```

### Трейдофф: качество vs стоимость

Multi-query улучшает recall, но:
- Требует вызова LLM для генерации запросов (задержка + стоимость)
- Увеличивает количество поисковых запросов в N раз
- Может найти больше нерелевантного (снижение precision)

Рекомендация: используйте для сложных вопросов. Для простых фактоидных запросов ("Сколько дней отпуска?") — избыточно.

---

## Часть 3: Self-Query — извлечение структуры из естественного языка

### Когда фильтрация в голове пользователя

Пользователь: "Покажи HR-политики, обновлённые в 2025 году"

Этот запрос содержит:
- Семантическую часть: "HR-политики"
- Структурированные фильтры: department=HR, year=2025

Наивный vector search проигнорирует фильтры — модель эмбеддингов не "понимает" что 2024 — это дата, а HR — категория. Она просто найдёт документы, семантически похожие на всю строку.

### Self-Query Retriever: пусть LLM парсит запрос

Self-Query использует LLM для извлечения:
1. Семантического запроса (для vector search)
2. Фильтров по метаданным (для structured filtering)

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# Описание метаданных документов
metadata_field_info = [
    AttributeInfo(
        name="department",
        description="Отдел, к которому относится документ (HR, Finance, IT, Legal)",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="Год публикации или последнего обновления документа",
        type="integer",
    ),
    AttributeInfo(
        name="document_type",
        description="Тип документа (policy, procedure, guideline, form)",
        type="string",
    ),
]

document_content_description = "Корпоративные документы: политики, процедуры, инструкции"

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

# Запрос с неявными фильтрами
results = retriever.invoke("HR-политики, обновлённые в 2024 году")
```

LLM преобразует запрос в:
- Query: "HR политики"
- Filter: {"department": "HR", "year": {"$gte": 2024}}

### Примеры преобразований Self-Query

| Пользовательский запрос | Извлечённый query | Извлечённые фильтры |
|-------------------------|-------------------|---------------------|
| "Финансовые отчёты за Q3" | "финансовые отчёты" | quarter=3 |
| "Процедуры IT не старше 2023" | "IT процедуры" | department=IT, year>=2023 |
| "Политика отпусков кроме Legal" | "политика отпусков" | department != Legal |
| "Документы про безопасность от Иванова" | "безопасность" | author=Иванов |

### Ограничения Self-Query

- Зависит от качества описания метаданных (AttributeInfo)
- Может неправильно интерпретировать сложные запросы
- Требует вызова LLM (задержка)
- Не все vectorstore поддерживают сложную фильтрацию

---

## Часть 4: HyDE — Hypothetical Document Embeddings

### Революционная идея: искать по ответу, а не по вопросу

HyDE (Hypothetical Document Embeddings) — контринтуитивная, но мощная техника.

Проблема: вопрос и ответ могут иметь разную семантику. "Сколько дней отпуска положено сотруднику?" (вопрос) vs "Ежегодный оплачиваемый отпуск составляет 28 календарных дней" (ответ). Вектор вопроса может быть далёк от вектора ответа!

Решение HyDE:
1. LLM генерирует гипотетический ответ на вопрос (без retrieval, "из головы")
2. Этот гипотетический ответ превращается в вектор
3. Поиск идёт по вектору ответа, а не вопроса

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-5")
base_embeddings = OpenAIEmbeddings()

# Создаём HyDE embeddings
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"  # или кастомный промпт
)

# Использование: вместо обычных embeddings
vectorstore = Chroma.from_documents(documents, hyde_embeddings)
```

### Почему это работает?

Гипотетический ответ, даже если он фактически неверен, будет использовать похожую терминологию и структуру, как реальный ответ в базе знаний.

Вопрос: "Каковы требования к паролям в нашей компании?"

Гипотетический ответ LLM:
> "Согласно политике информационной безопасности, пароли должны содержать минимум 12 символов, включать буквы разного регистра, цифры и специальные символы. Смена пароля требуется каждые 90 дней."

Этот текст ближе по стилю к реальному документу политики, чем сам вопрос.

### Ограничения HyDE

- Требует вызова LLM для каждого запроса (задержка + стоимость)
- Галлюцинации LLM могут направить поиск в неверную сторону
- Работает лучше для фактоидных вопросов, хуже для аналитических

### Кастомный промпт для HyDE

```python
from langchain_core.prompts import PromptTemplate

hyde_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Ты — эксперт по корпоративным политикам.
Напиши короткий параграф (2-3 предложения), который мог бы быть фрагментом
документа, отвечающего на данный вопрос.
Используй формальный стиль корпоративной документации.

Вопрос: {question}

Гипотетический фрагмент документа:"""
)
```

---

## Часть 5: Композиция Retriever'ов

### Паттерн: последовательная обработка

Retriever'ы можно соединять в цепочки. Результат одного становится входом другого.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Базовый retriever возвращает "сырые" документы
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Компрессор извлекает только релевантные части
compressor = LLMChainExtractor.from_llm(llm)

# Композитный retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Результат: меньше текста, только релевантное
results = compression_retriever.invoke("Какие документы нужны для отпуска?")
```

### Паттерн: параллельное объединение

Несколько retriever'ов работают параллельно, результаты объединяются.

```python
from langchain.retrievers.merger_retriever import MergerRetriever

# Разные retriever'ы для разных источников
hr_retriever = hr_vectorstore.as_retriever()
it_retriever = it_vectorstore.as_retriever()
finance_retriever = finance_vectorstore.as_retriever()

# Объединяем
merged_retriever = MergerRetriever(
    retrievers=[hr_retriever, it_retriever, finance_retriever]
)
```

### Паттерн: условная маршрутизация

Выбор retriever'а в зависимости от типа запроса.

```python
from langchain_core.runnables import RunnableBranch

def classify_query(query):
    """Классифицирует запрос по теме"""
    # В реальности — вызов LLM или простой классификатор
    if "HR" in query or "отпуск" in query or "зарплата" in query:
        return "hr"
    elif "IT" in query or "компьютер" in query or "пароль" in query:
        return "it"
    else:
        return "general"

routing_retriever = RunnableBranch(
    (lambda x: classify_query(x) == "hr", hr_retriever),
    (lambda x: classify_query(x) == "it", it_retriever),
    general_retriever  # default
)
```

### Построение сложного retrieval pipeline

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Сложный pipeline:
# 1. Параллельно: BM25 + Vector search
# 2. Объединение результатов
# 3. Reranking
# 4. Контекстное сжатие

# Шаг 1-2: Гибридный поиск
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# Шаг 3-4: Постобработка
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)
```

---

## Часть 6: Стратегии для сложных вопросов

### Декомпозиция вопроса

Сложные вопросы можно разбить на подвопросы:

"Сравни политики удалённой работы в нашей компании и у конкурента X, с фокусом на гибкость графика"

Декомпозиция:
1. "Политика удалённой работы в нашей компании"
2. "Политика удалённой работы у компании X"
3. "Гибкость рабочего графика при удалённой работе"

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

decomposition_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Разбей сложный вопрос на 2-4 простых подвопроса,
которые вместе помогут ответить на исходный вопрос.

Сложный вопрос: {question}

Простые подвопросы (по одному на строку):"""
)

# Используем Multi-Query с кастомным промптом для декомпозиции
decomposition_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=decomposition_prompt
)
```

### Step-Back Prompting: абстракция вопроса

Иногда полезно сначала задать более общий вопрос.

Оригинал: "Какие налоговые вычеты может получить сотрудник за обучение ребёнка в 2025 году?"

Step-back: "Какие налоговые вычеты существуют для физических лиц?"

Сначала ищем общий контекст, затем конкретный.

```python
stepback_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Сформулируй более общий, абстрактный вопрос,
который поможет понять контекст исходного вопроса.

Исходный вопрос: {question}
Более общий вопрос:"""
)

# Двухэтапный retrieval
def stepback_retrieval(question):
    # Генерируем step-back вопрос
    stepback_q = llm.invoke(stepback_prompt.format(question=question)).content

    # Ищем по обоим
    general_docs = retriever.invoke(stepback_q)
    specific_docs = retriever.invoke(question)

    # Комбинируем
    return general_docs + specific_docs
```

### Temporal Awareness: учёт времени

Для вопросов про "текущую" или "новую" информацию важно учитывать даты.

```python
from datetime import datetime

def add_temporal_filter(query, retriever):
    """Добавляет временной фильтр для вопросов про актуальность"""
    temporal_keywords = ["текущий", "актуальный", "новый", "последний", "2025", "2026"]

    if any(kw in query.lower() for kw in temporal_keywords):
        current_year = datetime.now().year
        return retriever.with_config({
            "filter": {"year": {"$gte": current_year - 1}}
        })
    return retriever
```

---

## Часть 7: Практические рекомендации

### Выбор стратегии retrieval

| Тип запросов | Рекомендуемая стратегия |
|--------------|------------------------|
| Простые фактоидные | Базовый vector search |
| Смесь терминов и семантики | Hybrid (BM25 + Vector) |
| Сложные, многоаспектные | Multi-Query |
| С явными фильтрами | Self-Query |
| Запросы-вопросы | HyDE |
| Разнородные источники | Routing + Merger |

### Чек-лист оптимизации retrieval

1. **Baseline:** Начните с простого vector search
2. **Измерьте:** recall@k на тестовом наборе вопросов
3. **Гибридный поиск:** Если много терминов/кодов — добавьте BM25
4. **Фильтрация:** Если есть структурированные метаданные — Self-Query
5. **Multi-Query:** Для сложных вопросов
6. **Reranking:** Если precision важнее recall

### Мониторинг и отладка

```python
import logging

# Включаем логирование для отладки retrieval
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain.retrievers")

# Или используем callbacks
from langchain_core.callbacks import StdOutCallbackHandler

results = retriever.invoke(
    "Вопрос",
    config={"callbacks": [StdOutCallbackHandler()]}
)
```

Логируйте:
- Какие запросы генерирует Multi-Query
- Какие фильтры извлекает Self-Query
- Какой гипотетический документ создаёт HyDE
- Скоры найденных документов

---

## Заключение: Retrieval как искусство

Продвинутый retrieval — это не один алгоритм, а набор техник, которые комбинируются под конкретную задачу. Нет универсального "лучшего" подхода — есть подход, оптимальный для ваших данных и запросов.

Ключевые идеи:

**Гибридный поиск** объединяет сильные стороны семантики (понимание смысла) и лексики (точные термины).

**Multi-Query** расширяет охват, генерируя варианты формулировок вопроса.

**Self-Query** извлекает структурированные фильтры из естественного языка.

**HyDE** сдвигает поиск от "похоже на вопрос" к "похоже на ответ".

**Композиция** позволяет строить сложные pipeline из простых компонентов.

---

## Вопросы для самопроверки

1. В чём слабость чистого векторного поиска? Приведите примеры запросов, где он может провалиться.

2. Объясните, как BM25 и vector search дополняют друг друга в гибридном поиске.

3. Какую проблему решает Multi-Query Retriever? Когда его стоит использовать?

4. Как работает Self-Query? Какие компоненты нужны для его настройки?

5. Объясните идею HyDE. Почему поиск по гипотетическому ответу может работать лучше, чем поиск по вопросу?

6. Спроектируйте retrieval pipeline для корпоративной базы знаний с документами разных отделов и разных лет.

---

## Ключевые термины

| Термин | Определение |
|--------|-------------|
| **Hybrid Search** | Комбинация векторного и лексического (BM25) поиска |
| **BM25** | Алгоритм ранжирования на основе частоты терминов |
| **Multi-Query Retriever** | Retriever, генерирующий несколько вариантов запроса |
| **Self-Query** | Техника извлечения структурированных фильтров из естественного языка |
| **HyDE** | Hypothetical Document Embeddings — поиск по гипотетическому ответу |
| **RRF** | Reciprocal Rank Fusion — метод объединения ранжированных списков |
| **Ensemble Retriever** | Композиция нескольких retriever'ов с весами |
| **Query Decomposition** | Разбиение сложного вопроса на подвопросы |
