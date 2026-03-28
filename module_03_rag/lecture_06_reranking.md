# Раздел 6: Переранжирование — Второй шанс для точности

## Введение: Почему первый результат не всегда лучший

Представьте себе процесс найма на важную позицию. Сначала HR-менеджер просматривает сотни резюме и отбирает 20 кандидатов, которые "на первый взгляд" подходят. Это быстрый, поверхностный отбор — как наш retrieval. Затем руководитель внимательно изучает эту двадцатку, проводит интервью и выбирает тройку финалистов. Этот второй этап — медленный, тщательный, глубокий анализ — и есть reranking.

Retrieval-системы сталкиваются с фундаментальным компромиссом: быстро или точно. Bi-encoder эмбеддинги позволяют искать среди миллионов документов за миллисекунды, но их "понимание" релевантности поверхностно. Они сравнивают готовые векторы, не видя прямого взаимодействия между словами вопроса и словами документа.

Reranking — это второй этап pipeline, на котором мы берём "грубый" топ-k из retrieval и тщательно пересортировываем его более мощной, но медленной моделью. Результат: точность как у глубокого анализа, скорость как у быстрого поиска.

---

## Часть 1: Bi-Encoder vs Cross-Encoder — ключевое различие

### Ограничения Bi-Encoder

Вспомним, как работает Bi-Encoder (стандартный embedding-подход):

```
Вопрос: "Какие документы нужны для отпуска?"
    ↓ Encoder
    [0.23, -0.45, 0.12, ...]  ← Вектор вопроса

Документ: "Для оформления отпуска предоставьте заявление и согласование"
    ↓ Encoder (тот же)
    [0.21, -0.43, 0.15, ...]  ← Вектор документа

    ↓
    Косинусное сходство: 0.94
```

Ключевой момент: вопрос и документ обрабатываются независимо. Encoder не знает, с чем он сравнивает текущий текст. Сходство определяется только по финальным векторам.

Это ограничивает "глубину понимания". Вектор документа должен универсально представлять его смысл для любого возможного вопроса. Нюансы взаимосвязи конкретного вопроса и конкретного документа теряются.

### Cross-Encoder: совместная обработка

Cross-Encoder работает иначе:

```
Вход: "[CLS] Какие документы нужны для отпуска? [SEP] Для оформления отпуска предоставьте заявление и согласование [SEP]"
    ↓ Transformer (все слои внимания видят и вопрос, и документ)
    ↓ Classification Head
    Score: 0.91
```

Вопрос и документ подаются вместе как единый вход. Механизм внимания (attention) напрямую связывает слова вопроса со словами документа. Модель видит, что "документы" в вопросе соответствуют "заявление и согласование" в ответе, что "отпуск" встречается в обоих.

Это гораздо более глубокий анализ релевантности. Исследования показывают, что Cross-Encoder превосходит Bi-Encoder на 10-20% по метрикам точности.

### Почему не использовать только Cross-Encoder?

Проблема в вычислительной сложности.

**Bi-Encoder:** индексируем N документов (N вызовов encoder'а один раз). При запросе: 1 вызов для вопроса + поиск по индексу. Время: O(1) + O(log N) или O(√N) с ANN.

**Cross-Encoder:** для каждого запроса нужно прогнать N пар (вопрос, документ). Время: O(N) вызовов encoder'а на каждый запрос.

При N = 1 000 000 документов Cross-Encoder потребует миллион прогонов нейросети на один вопрос. Это минуты или часы, даже на мощном GPU.

Решение: двухэтапный pipeline.
1. **Retrieval (Bi-Encoder):** быстро отбираем топ-100 кандидатов
2. **Reranking (Cross-Encoder):** тщательно ранжируем 100 → выбираем топ-5

100 вызовов Cross-Encoder — это секунды, вполне приемлемо для interactive latency.

---

## Часть 2: Модели для Reranking

### Cohere Rerank

Cohere предлагает API для reranking как сервис. Преимущества: простота использования, высокое качество, не нужно управлять инфраструктурой.

```python
import cohere

co = cohere.Client('your-api-key')

query = "Какие документы нужны для оформления отпуска?"
documents = [
    "Для отпуска заполните форму Т-6 и получите подпись руководителя.",
    "График отпусков утверждается в декабре предшествующего года.",
    "Командировочные расходы компенсируются по авансовому отчёту.",
    "Заявление на отпуск подаётся за 2 недели до начала."
]

results = co.rerank(
    model="rerank-v4.0-pro",  # Новая версия с улучшенным качеством
    query=query,
    documents=documents,
    top_n=2,  # Вернуть топ-2
    return_documents=True
)

for result in results.results:
    print(f"Score: {result.relevance_score:.3f}")
    print(f"Document: {result.document.text}\n")
```

### BGE-Reranker (Open-Source)

BAAI (Beijing Academy of AI) выпустила серию открытых reranker-моделей.

```python
from FlagEmbedding import FlagReranker

# Загрузка модели
reranker = FlagReranker('BAAI/bge-reranker-v2.5-gemma2-lightweight', use_fp16=True)

# Reranking
query = "Какие документы нужны для оформления отпуска?"
passages = [
    "Для отпуска заполните форму Т-6...",
    "График отпусков утверждается...",
    "Командировочные расходы..."
]

# Создаём пары (query, passage)
pairs = [[query, p] for p in passages]

# Получаем скоры
scores = reranker.compute_score(pairs, normalize=True)
# scores: [0.92, 0.34, 0.08]

# Сортируем
ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
```

### Sentence-Transformers Cross-Encoder

Библиотека Sentence-Transformers предоставляет удобную обёртку:

```python
from sentence_transformers import CrossEncoder

# Загрузка модели
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Для русского/мультиязычного
model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# Пары для оценки
pairs = [
    ["Какие документы нужны для отпуска?", "Форма Т-6 и заявление руководителю"],
    ["Какие документы нужны для отпуска?", "График отпусков утверждается в декабре"],
]

scores = model.predict(pairs)
# scores: array([0.89, 0.12])
```

### Сравнение моделей reranking

| Модель | Качество | Скорость | Языки | Тип |
|--------|----------|----------|-------|-----|
| Cohere Rerank v4.0-pro | Отличное | Облако | 100+ | API |
| Cohere Rerank v4.0-fast | Хорошее | Очень быстрая | 100+ | API |
| BGE-Reranker-v2.5-gemma2 | Отличное | Быстрая | 100+ | Open-source |
| ms-marco-MiniLM | Хорошее | Очень быстрая | EN | Open-source |
| mMiniLM-reranker | Хорошее | Быстрая | Multi | Open-source |
| Jina Reranker v2 | Отличное | Средняя | Multi | Open-source |

Рекомендация: для production с русским языком — Cohere v4.0-pro (если бюджет позволяет) или BGE-Reranker-v2.5-gemma2-lightweight (open-source с отличным качеством).

---

## Часть 3: Интеграция Reranking в RAG

### LangChain ContextualCompressionRetriever

LangChain предоставляет абстракцию для reranking:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.vectorstores import Chroma

# Базовый retriever
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Reranker
compressor = CohereRerank(
    cohere_api_key="your-key",
    model="rerank-v4.0-pro",
    top_n=5
)

# Композитный retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Использование
results = compression_retriever.invoke("Какие документы нужны для отпуска?")
# Вернёт топ-5 после reranking из топ-20 retrieval
```

### Кастомный Reranker с BGE

```python
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker
from typing import List, Sequence

class BGEReranker(BaseDocumentCompressor):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reranker = FlagReranker(self.model_name, use_fp16=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str
    ) -> List[Document]:
        if not documents:
            return []

        # Создаём пары
        pairs = [[query, doc.page_content] for doc in documents]

        # Получаем скоры
        scores = self._reranker.compute_score(pairs, normalize=True)

        # Добавляем скоры в metadata и сортируем
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = score

        sorted_docs = sorted(
            documents,
            key=lambda x: x.metadata["rerank_score"],
            reverse=True
        )

        return sorted_docs[:self.top_n]

# Использование
bge_reranker = BGEReranker(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=bge_reranker,
    base_retriever=base_retriever
)
```

### Полный RAG Pipeline с Reranking

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Компоненты
llm = ChatOpenAI(model="gpt-5")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
reranker = CohereRerank(top_n=5)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

# Промпт
prompt = ChatPromptTemplate.from_template("""
Ответь на вопрос, используя только следующий контекст:

{context}

Вопрос: {question}

Ответ:
""")

# Функция форматирования контекста
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {"context": reranking_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Использование
answer = rag_chain.invoke("Какие документы нужны для оформления отпуска?")
```

---

## Часть 4: Настройка и оптимизация Reranking

### Выбор параметров

**top_k (для retrieval):** Сколько документов отбирать на первом этапе.
- Слишком мало (10) — можем упустить релевантные
- Слишком много (100) — дорого для reranking
- Рекомендация: 20-50 для большинства случаев

**top_n (для reranking):** Сколько документов оставлять после reranking.
- Зависит от размера контекстного окна LLM
- 3-5 для простых вопросов
- 5-10 для сложных, требующих синтеза

### Score Threshold: отсечение по качеству

Иногда лучше вернуть меньше документов, но более качественных:

```python
class ThresholdReranker(BaseDocumentCompressor):
    min_score: float = 0.5  # Минимальный скор релевантности

    def compress_documents(self, documents, query):
        # ... reranking logic ...

        # Фильтруем по порогу
        filtered = [
            doc for doc in sorted_docs
            if doc.metadata["rerank_score"] >= self.min_score
        ]

        # Но оставляем хотя бы один документ
        if not filtered and sorted_docs:
            return sorted_docs[:1]

        return filtered
```

### Batching для эффективности

При обработке множества запросов — батчинг ускоряет reranking:

```python
class BatchedReranker:
    def __init__(self, model_name, batch_size=32):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank_batch(self, queries, documents_list):
        """
        queries: List[str] — список запросов
        documents_list: List[List[str]] — документы для каждого запроса
        """
        all_pairs = []
        query_indices = []

        for i, (query, docs) in enumerate(zip(queries, documents_list)):
            for doc in docs:
                all_pairs.append([query, doc])
                query_indices.append(i)

        # Один большой batch
        all_scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        # Распределяем скоры обратно
        results = [[] for _ in queries]
        for idx, score in zip(query_indices, all_scores):
            results[idx].append(score)

        return results
```

---

## Часть 5: Альтернативные подходы к Reranking

### LLM-as-Reranker

Современные LLM можно использовать напрямую для reranking:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

def llm_rerank(query, documents, top_n=5):
    # Формируем промпт
    docs_text = "\n\n".join([
        f"[{i+1}] {doc}" for i, doc in enumerate(documents)
    ])

    prompt = f"""Дан вопрос и список документов. Оцени релевантность каждого
документа вопросу по шкале 0-10 и верни номера {top_n} наиболее релевантных
в порядке убывания релевантности.

Вопрос: {query}

Документы:
{docs_text}

Верни только номера через запятую, без объяснений:"""

    response = llm.invoke(prompt)
    # Парсим ответ: "1, 4, 2"
    indices = [int(x.strip()) - 1 for x in response.content.split(",")]

    return [documents[i] for i in indices[:top_n]]
```

Плюсы: можно объяснить критерии релевантности, гибкость.
Минусы: медленно, дорого, вариативность ответов.

### Listwise Reranking

Вместо оценки каждого документа отдельно (pointwise) или парами (pairwise), оцениваем весь список целиком:

```python
def listwise_rerank(query, documents):
    """LLM видит весь список и сортирует его"""
    prompt = f"""Отсортируй документы по релевантности вопросу.
Верни номера в порядке от наиболее к наименее релевантному.

Вопрос: {query}

Документы:
{format_numbered_docs(documents)}

Порядок (номера через запятую):"""

    response = llm.invoke(prompt)
    order = parse_order(response.content)
    return [documents[i] for i in order]
```

Исследования показывают, что listwise может быть точнее pointwise для небольших списков (до 20 документов).

### Ensemble Reranking

Комбинация нескольких reranker'ов:

```python
def ensemble_rerank(query, documents, rerankers, weights):
    """
    rerankers: список моделей reranking
    weights: веса для каждой модели
    """
    all_scores = []

    for reranker in rerankers:
        scores = reranker.score(query, documents)
        # Нормализация min-max
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        all_scores.append(scores)

    # Взвешенное среднее
    final_scores = sum(w * s for w, s in zip(weights, all_scores))

    # Сортировка
    ranked = sorted(
        zip(documents, final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked]
```

---

## Часть 6: Метрики и оценка качества Reranking

### Метрики ранжирования

**NDCG (Normalized Discounted Cumulative Gain):**

Учитывает не только релевантность, но и позицию. Релевантный документ на 1-й позиции ценнее, чем на 10-й.

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

где IDCG — DCG идеального ранжирования.

**MRR (Mean Reciprocal Rank):**

Позиция первого релевантного документа.

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

**Precision@k / Recall@k:**

Стандартные метрики: какая доля топ-k релевантна / какая доля всех релевантных попала в топ-k.

### A/B тестирование reranking

```python
import random

def ab_test_retrieval(query, base_retriever, reranking_retriever, p_treatment=0.5):
    """
    С вероятностью p_treatment используем reranking,
    иначе — базовый retriever
    """
    use_reranking = random.random() < p_treatment

    if use_reranking:
        results = reranking_retriever.invoke(query)
        variant = "treatment"
    else:
        results = base_retriever.invoke(query)
        variant = "control"

    # Логируем для анализа
    log_experiment(query, results, variant)

    return results
```

### Offline evaluation с golden dataset

```python
def evaluate_reranking(reranker, test_set):
    """
    test_set: List[{query, documents, relevance_labels}]
    relevance_labels: List[int] — метки релевантности (0/1 или 0-3)
    """
    ndcg_scores = []

    for item in test_set:
        # Reranking
        reranked_docs = reranker(item["query"], item["documents"])

        # Восстанавливаем метки в новом порядке
        reranked_labels = [
            item["relevance_labels"][item["documents"].index(doc)]
            for doc in reranked_docs
        ]

        # NDCG
        ndcg = compute_ndcg(reranked_labels, k=5)
        ndcg_scores.append(ndcg)

    return {
        "ndcg@5_mean": np.mean(ndcg_scores),
        "ndcg@5_std": np.std(ndcg_scores)
    }
```

---

## Часть 7: Практические рекомендации

### Когда reranking критичен

- **Высокие требования к precision:** Юридические, медицинские, финансовые домены
- **Ограниченный контекст LLM:** Нужно отобрать лучшие 3-5 из многих кандидатов
- **Разнородные документы:** Когда retrieval возвращает "похожие, но не то"
- **Сложные запросы:** Требующие понимания нюансов связи вопроса и ответа

### Когда можно обойтись без reranking

- **Простые FAQ:** Вопрос-ответ с очевидным соответствием
- **Жёсткие требования к латентности:** <100ms на запрос
- **Ограниченный бюджет:** Reranking API дорого при большом трафике

### Чек-лист внедрения reranking

1. **Baseline:** Измерьте качество без reranking (NDCG, Precision@k)
2. **Выберите модель:** Cohere для простоты, BGE для self-hosted
3. **Настройте параметры:** top_k retrieval, top_n reranking
4. **Измерьте улучшение:** A/B тест или offline evaluation
5. **Мониторинг:** Латентность, стоимость, качество в production

### Типичные ошибки

**Ошибка 1: Reranking слишком малого топ-k**

Если retrieval возвращает топ-5 для reranking топ-3 — мало пространства для улучшения. Reranking нужен для "просеивания" большого списка.

**Ошибка 2: Игнорирование латентности**

CrossEncoder на CPU без оптимизации может добавить секунды к каждому запросу. Планируйте инфраструктуру.

**Ошибка 3: Не учитывать разнообразие**

Топ-5 после reranking может быть 5 почти одинаковых документов. Иногда нужен diversity-aware reranking.

---

## Заключение: Reranking как страховка качества

Reranking — это "вторая линия обороны" в RAG-системе. Если retrieval упустил нюанс или ошибся в ранжировании, reranking может это исправить.

Ключевые идеи:

**Cross-Encoder vs Bi-Encoder:** Cross-Encoder точнее, потому что видит вопрос и документ одновременно. Но он медленный — только для небольших списков.

**Двухэтапная архитектура:** Bi-Encoder для широкого отбора (1M → 50), Cross-Encoder для точной сортировки (50 → 5).

**Модели:** Cohere Rerank для простоты, BGE-Reranker для self-hosted, LLM для максимальной гибкости.

**Метрики:** NDCG, MRR, Precision@k — измеряйте до и после внедрения.

В следующей, финальной лекции модуля мы соберём всё воедино: разберём, как правильно подавать найденный контекст в LLM и как добиться, чтобы модель цитировала источники.

---

## Вопросы для самопроверки

1. В чём принципиальное отличие Bi-Encoder от Cross-Encoder? Почему Cross-Encoder точнее?

2. Почему нельзя использовать только Cross-Encoder для retrieval среди миллиона документов?

3. Как выбрать оптимальное соотношение top_k (retrieval) и top_n (reranking)?

4. Сравните Cohere Rerank и BGE-Reranker. Когда вы бы выбрали каждый?

5. Что такое NDCG и почему эта метрика важна для оценки reranking?

6. Спроектируйте pipeline с reranking для системы с требованием латентности <500ms.

---

## Ключевые термины

| Термин | Определение |
|--------|-------------|
| **Reranking** | Повторное ранжирование результатов поиска более точной моделью |
| **Cross-Encoder** | Модель, обрабатывающая пару (вопрос, документ) совместно |
| **Bi-Encoder** | Модель, кодирующая вопрос и документ независимо |
| **NDCG** | Normalized Discounted Cumulative Gain — метрика качества ранжирования |
| **MRR** | Mean Reciprocal Rank — средняя обратная позиция первого релевантного результата |
| **Pointwise Reranking** | Оценка каждого документа независимо |
| **Listwise Reranking** | Оценка и сортировка всего списка целиком |
| **Score Threshold** | Минимальный скор для включения документа в результат |
