# Раздел 4: Подготовка данных — ETL и стратегии Chunking

## Введение: Мусор на входе — мусор на выходе

В мире машинного обучения есть старая поговорка: "Garbage in, garbage out" — мусор на входе даст мусор на выходе. Для RAG-систем эта истина особенно актуальна. Вы можете выбрать лучшую модель эмбеддингов, самую быструю векторную базу данных, тончайшим образом настроить reranking — но если ваши документы плохо подготовлены, система будет давать посредственные результаты.

Подготовка данных — невидимый герой успешного RAG. Она включает три взаимосвязанных процесса: извлечение текста из различных форматов (E — Extract), его преобразование и очистка (T — Transform), и загрузка в индекс (L — Load). Центральное место в этом процессе занимает chunking — разбиение документов на фрагменты оптимального размера и структуры.

Представьте себе, что вы готовите ингредиенты для изысканного блюда. Мало просто купить продукты — их нужно правильно нарезать. Лук мелкими кубиками для соуса, крупными полукольцами для гриля. Мясо поперёк волокон для стейка, вдоль для шашлыка. Неправильная нарезка испортит блюдо, даже если продукты первоклассные. То же самое с текстом: неправильный chunking "испортит" даже лучшую базу знаний.

---

## Часть 1: Загрузка данных — Document Loaders

### Многообразие источников

Корпоративная база знаний редко живёт в одном месте и формате. Типичная картина:

- Политики и регламенты в PDF
- Инструкции в Word и Google Docs
- База знаний в Confluence или Notion
- Технические спецификации в Markdown
- Переписка в Slack или Teams
- Таблицы в Excel
- Данные в SQL-базах
- Веб-страницы и документация

Каждый источник требует своего "загрузчика" — компонента, который умеет извлекать текст и метаданные.

### LangChain Document Loaders

LangChain предоставляет огромную коллекцию загрузчиков. Результат работы любого загрузчика — список объектов Document с полями `page_content` (текст) и `metadata` (словарь метаданных).

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    WebBaseLoader,
    ConfluenceLoader,
    NotionDBLoader
)

# PDF
pdf_loader = PyPDFLoader("report.pdf")
pdf_docs = pdf_loader.load()
# Каждая страница — отдельный Document
# metadata включает номер страницы, путь к файлу

# Word
docx_loader = Docx2txtLoader("policy.docx")
docx_docs = docx_loader.load()

# Веб-страница
web_loader = WebBaseLoader("https://example.com/docs")
web_docs = web_loader.load()

# CSV (каждая строка — документ)
csv_loader = CSVLoader("data.csv")
csv_docs = csv_loader.load()
```

### Особенности работы с PDF

PDF — один из самых сложных форматов для извлечения текста. Причина в том, что PDF хранит информацию о визуальном расположении элементов, а не о логической структуре текста.

**Проблема колонок:** Текст в две колонки может извлечься как чередование строк из левой и правой колонки, превращая связный текст в кашу.

**Проблема таблиц:** Табличные данные теряют структуру, превращаясь в набор разрозненных чисел.

**Проблема сканов:** PDF может содержать отсканированные изображения вместо текста — нужен OCR.

Разные загрузчики справляются с этими проблемами по-разному:

```python
# Простой, быстрый, но хрупкий
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("doc.pdf")

# Более умный, использует heuristics для структуры
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("doc.pdf")

# Unstructured — мощный, но медленный
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("doc.pdf", mode="elements")
# mode="elements" разбивает на структурные элементы (заголовки, параграфы, таблицы)

# Для сканов — с OCR
from langchain_community.document_loaders import UnstructuredPDFLoader
loader = UnstructuredPDFLoader("scanned.pdf", strategy="ocr_only")
```

**Рекомендация:** Для production используйте комбинацию. Начните с PyPDFLoader для простых PDF. Для сложных документов с таблицами и колонками — Unstructured или специализированные API (Adobe PDF Services, AWS Textract).

### Работа с веб-контентом

Веб-страницы несут много "шума": навигация, футеры, боковые панели, реклама. Извлечь только полезный контент — нетривиальная задача.

```python
from langchain_community.document_loaders import WebBaseLoader

# Базовая загрузка
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# С фильтрацией по CSS-селектору
loader = WebBaseLoader(
    "https://example.com/article",
    bs_kwargs={"parse_only": SoupStrainer(class_="article-content")}
)
```

Для "боевых" задач рассмотрите специализированные инструменты:

- **Trafilatura** — библиотека для извлечения основного контента
- **Newspaper3k** — заточена под новостные статьи (⚠️ фактически не поддерживается с 2024 года, нет обновлений более 12 месяцев)
- **Jina Reader API** — облачный сервис для конвертации веб-страниц в Markdown

### Обогащение метаданными

Метаданные критически важны для фильтрации при retrieval. Извлекайте максимум информации:

```python
import os
from datetime import datetime

def load_with_metadata(file_path):
    # Базовая загрузка
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Обогащение метаданными
    file_stat = os.stat(file_path)
    for doc in docs:
        doc.metadata.update({
            "filename": os.path.basename(file_path),
            "file_size": file_stat.st_size,
            "modified_date": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            "directory": os.path.dirname(file_path),
            # Можно добавить кастомные поля
            "department": extract_department_from_path(file_path),
            "document_type": classify_document_type(file_path)
        })

    return docs
```

---

## Часть 2: Chunking — искусство разбиения текста

### Почему размер имеет значение

Chunking — разбиение документов на фрагменты — определяет гранулярность поиска. Это один из самых влияющих на качество RAG факторов.

**Слишком большие чанки:**
- Больше контекста — хорошо для понимания
- Но сложнее найти конкретный ответ в "стоге сена"
- Занимают много места в контекстном окне LLM
- Выше стоимость (токены = деньги)

**Слишком маленькие чанки:**
- Теряется контекст
- Один чанк может содержать неполную мысль
- Больше чанков → медленнее поиск, больше индекс

**Идеальный чанк:**
- Содержит законченную мысль или логическую единицу
- Достаточно контекста для понимания
- Достаточно специфичен для точного поиска

К сожалению, "идеальный" размер зависит от типа документов, языка, задачи. Это параметр для экспериментов.

### Фиксированный chunking: простой старт

Простейший подход — резать по количеству символов с перекрытием (overlap).

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,      # Максимум символов в чанке
    chunk_overlap=200,     # Перекрытие между соседними чанками
    separator="\n"         # Предпочтительная точка разрыва
)

chunks = splitter.split_documents(documents)
```

**Перекрытие (overlap)** критически важно! Без него информация на границе чанков теряется. Если предложение разрезано пополам, retriever может найти одну половину, но без контекста другой половины ответ будет неполным.

Типичные значения: chunk_size 500-2000 символов, overlap 10-20% от chunk_size.

### RecursiveCharacterTextSplitter: умнее, чем кажется

LangChain предлагает "рекурсивный" сплиттер, который пытается сохранять логическую структуру.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Приоритет разделителей
)
```

Логика работы:
1. Пытается разбить по "\n\n" (двойной перенос — обычно конец параграфа)
2. Если чанки всё ещё большие, разбивает по "\n" (одинарный перенос)
3. Затем по ". " (конец предложения)
4. В крайнем случае — по пробелам или посимвольно

Результат: чанки чаще заканчиваются на границах предложений и параграфов, а не посередине слова.

### Chunking с учётом токенов

Модели эмбеддингов и LLM считают не символы, а токены. Один токен — примерно 4 символа для английского, меньше для языков с не-латинским алфавитом.

```python
from langchain_text_splitters import TokenTextSplitter

# Использует токенизатор tiktoken (OpenAI)
splitter = TokenTextSplitter(
    chunk_size=500,      # В токенах, не символах
    chunk_overlap=50
)
```

Для точного контроля над размером контекста (особенно при дорогих API) — используйте токенный сплиттер.

### Семантический chunking: по смыслу, не по размеру

Революционный подход: определять границы чанков не по длине, а по изменению темы.

**Идея:** Вычисляем эмбеддинги для скользящего окна текста. Когда сходство между соседними окнами резко падает — это граница темы, здесь и режем.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # или "standard_deviation"
    breakpoint_threshold_amount=95  # Порог для определения "скачка"
)

chunks = splitter.split_documents(documents)
```

**Преимущества:**
- Чанки соответствуют логическим единицам текста
- Меньше "разрезанных" мыслей

**Недостатки:**
- Требует дополнительных вычислений (эмбеддинги для каждого окна)
- Размер чанков непредсказуем (может быть слишком большим или маленьким)
- Сложнее отлаживать

### Chunking для структурированных документов

Markdown, HTML, код — имеют явную структуру. Используйте её!

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Разбиение по заголовкам Markdown
headers_to_split_on = [
    ("#", "Заголовок 1"),
    ("##", "Заголовок 2"),
    ("###", "Заголовок 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)

# Каждый чанк содержит метаданные с иерархией заголовков!
# chunk.metadata = {"Заголовок 1": "Введение", "Заголовок 2": "Основные понятия"}
```

Для HTML:

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
```

Для кода:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python-специфичный сплиттер
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200
)
```

Языко-специфичные сплиттеры знают о конструкциях языка (функции, классы, импорты) и стараются не разрезать их посередине.

---

## Часть 3: Продвинутые стратегии chunking

### Parent Document Retriever: лучшее из двух миров

Дилемма: маленькие чанки дают точный поиск, большие — богатый контекст. Можно ли получить оба?

Parent Document Retriever решает эту проблему двухуровневой структурой:

1. **Дочерние чанки (children):** маленькие, для точного поиска
2. **Родительские документы (parents):** большие, содержащие детей, для контекста

При поиске:
1. Ищем по дочерним чанкам (точный поиск)
2. Возвращаем родительские документы (богатый контекст)

```python
from langchain_community.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore

# Сплиттер для родителей (большие чанки)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Сплиттер для детей (маленькие чанки)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Хранилище для родительских документов
docstore = InMemoryStore()

# Векторное хранилище для детей
vectorstore = Chroma(embedding_function=embeddings)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Добавляем документы
retriever.add_documents(documents)

# При поиске получаем родительские документы
results = retriever.get_relevant_documents("Как оформить отпуск?")
```

### Контекстное обогащение чанков

Проблема: чанк может начинаться со слов "Данные правила применяются..." — но какие правила? Контекст потерян.

Решение: добавлять к каждому чанку контекстную информацию.

**Способ 1: Добавить заголовки документа**

```python
def add_context_to_chunks(chunks, document_title, section_hierarchy):
    for chunk in chunks:
        context_prefix = f"Документ: {document_title}\n"
        context_prefix += f"Раздел: {' > '.join(section_hierarchy)}\n\n"
        chunk.page_content = context_prefix + chunk.page_content
    return chunks
```

**Способ 2: Генерация summary для каждого чанка**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5")

def add_summary_to_chunks(chunks):
    for chunk in chunks:
        summary = llm.invoke(
            f"Напиши одно предложение, описывающее тему этого текста:\n\n{chunk.page_content}"
        ).content
        chunk.metadata["summary"] = summary
    return chunks
```

### Hypothetical Questions: инвертированный подход

Нестандартная, но эффективная техника: для каждого чанка сгенерировать вопросы, на которые он отвечает.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

def generate_hypothetical_questions(chunk, num_questions=3):
    prompt = f"""Прочитай текст и сгенерируй {num_questions} вопроса,
    на которые этот текст отвечает:

    {chunk.page_content}

    Вопросы (по одному на строку):"""

    response = llm.invoke(prompt)
    questions = response.content.strip().split("\n")
    return questions

# При индексации сохраняем и чанк, и вопросы
# Поиск идёт по вопросам, возвращаем чанки
```

Преимущество: вопрос пользователя сравнивается с вопросами (а не с ответами), что часто даёт лучшее совпадение.

### Таблица стратегий chunking

| Стратегия | Преимущества | Недостатки | Когда использовать |
|-----------|--------------|------------|-------------------|
| **Fixed-size** | Простота, предсказуемость | Режет по живому | Быстрый старт, однородные тексты |
| **Recursive** | Учитывает структуру | Всё ещё может резать неудачно | Общий случай |
| **Semantic** | Чанки по смыслу | Дорого, непредсказуемый размер | Разнородные документы |
| **By headers** | Сохраняет структуру | Требует структурированных документов | Markdown, HTML, DOCX |
| **Parent-child** | Точность + контекст | Сложнее реализация | Требования к качеству |
| **Hypothetical Q** | Улучшенный matching | Дорого (LLM для каждого чанка) | Критичные приложения |

---

## Часть 4: Очистка и нормализация текста

### Типичные проблемы "сырого" текста

Текст, извлечённый из документов, часто содержит артефакты:

- **Множественные пробелы и переносы:** "Это    текст   с   лишними    пробелами"
- **Специальные символы:** \u200b (zero-width space), \xa0 (non-breaking space)
- **Лишние header/footer:** "Страница 12 из 50" на каждой странице PDF
- **Битая кодировка:** "РџСЂРёРјРµСЂ" вместо "Пример"
- **Мусорные элементы:** "[email protected]", "Confidential"

### Функции очистки

```python
import re
import unicodedata

def clean_text(text):
    # Нормализация Unicode
    text = unicodedata.normalize("NFKC", text)

    # Замена множественных пробелов/переносов
    text = re.sub(r'\s+', ' ', text)

    # Удаление zero-width символов
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

    # Удаление номеров страниц (пример паттерна)
    text = re.sub(r'Страница \d+ из \d+', '', text)

    # Удаление email-адресов (если нужно)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)

    return text.strip()

def clean_documents(docs):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs
```

### Дедупликация

Одинаковые документы или очень похожие чанки — проблема для RAG:

- Засоряют результаты поиска
- Увеличивают размер индекса
- Могут "забивать" контекст одинаковой информацией

```python
from datasketch import MinHash, MinHashLSH

def deduplicate_chunks(chunks, threshold=0.9):
    """
    Удаляет почти-дубликаты с использованием MinHash LSH
    threshold: порог сходства для считания дубликатом
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique_chunks = []

    for i, chunk in enumerate(chunks):
        # Создаём MinHash для чанка
        m = MinHash(num_perm=128)
        for word in chunk.page_content.lower().split():
            m.update(word.encode('utf8'))

        # Проверяем, есть ли похожие
        duplicates = lsh.query(m)
        if not duplicates:
            lsh.insert(str(i), m)
            unique_chunks.append(chunk)

    return unique_chunks
```

---

## Часть 5: Pipeline обработки данных

### Архитектура ETL для RAG

Соберём всё вместе в единый pipeline:

```python
from dataclasses import dataclass
from typing import List
from langchain_core.documents import Document

@dataclass
class ProcessingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"

class RAGDataPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    def load(self, file_path: str) -> List[Document]:
        """Extract: загрузка документов"""
        extension = file_path.split('.')[-1].lower()

        loader_map = {
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader,
            'md': UnstructuredMarkdownLoader,
            'txt': TextLoader,
        }

        loader_class = loader_map.get(extension)
        if not loader_class:
            raise ValueError(f"Unsupported format: {extension}")

        loader = loader_class(file_path)
        return loader.load()

    def transform(self, docs: List[Document]) -> List[Document]:
        """Transform: очистка и chunking"""
        # Очистка
        docs = clean_documents(docs)

        # Chunking
        chunks = self.splitter.split_documents(docs)

        # Обогащение метаданными
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)

        # Дедупликация
        chunks = deduplicate_chunks(chunks)

        return chunks

    def process(self, file_path: str) -> List[Document]:
        """Полный pipeline"""
        docs = self.load(file_path)
        chunks = self.transform(docs)
        return chunks
```

### Обработка директории документов

```python
import os
from pathlib import Path
from tqdm import tqdm

def process_directory(directory: str, pipeline: RAGDataPipeline) -> List[Document]:
    """Обработка всех документов в директории"""
    all_chunks = []
    supported_extensions = {'.pdf', '.docx', '.md', '.txt'}

    files = list(Path(directory).rglob('*'))
    files = [f for f in files if f.suffix.lower() in supported_extensions]

    for file_path in tqdm(files, desc="Processing documents"):
        try:
            chunks = pipeline.process(str(file_path))
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Processed {len(files)} files, got {len(all_chunks)} chunks")
    return all_chunks
```

### Инкрементальная обработка

В production документы добавляются и изменяются. Полная переиндексация каждый раз — расточительно.

```python
import hashlib
from datetime import datetime

class IncrementalProcessor:
    def __init__(self, state_file: str = "processing_state.json"):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self):
        """Загрузка состояния (хэши обработанных файлов)"""
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def _file_hash(self, file_path: str) -> str:
        """Вычисление хэша файла"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def get_changed_files(self, directory: str) -> tuple:
        """Определение изменённых и удалённых файлов"""
        current_files = {}
        for file_path in Path(directory).rglob('*'):
            if file_path.is_file():
                current_files[str(file_path)] = self._file_hash(str(file_path))

        # Новые или изменённые
        to_process = []
        for path, hash_val in current_files.items():
            if path not in self.state or self.state[path] != hash_val:
                to_process.append(path)

        # Удалённые
        to_remove = [p for p in self.state if p not in current_files]

        return to_process, to_remove

    def mark_processed(self, file_path: str):
        self.state[file_path] = self._file_hash(file_path)
        self._save_state()
```

---

## Часть 6: Практические рекомендации

### Чек-лист подготовки данных

1. **Аудит источников:** Какие форматы? Какое качество? Есть ли сканы?

2. **Выбор загрузчиков:** Протестируйте несколько вариантов на репрезентативных документах

3. **Определение стратегии chunking:**
   - Начните с RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   - Измерьте recall на тестовых вопросах
   - Экспериментируйте с параметрами

4. **Очистка текста:** Изучите артефакты ваших документов, напишите специфичные правила

5. **Обогащение метаданными:** Чем больше структурированной информации — тем лучше фильтрация

6. **Дедупликация:** Особенно важно, если документы могут дублироваться

7. **Тестирование:** Создайте набор тестовых вопросов, проверяйте качество retrieval

### Типичные ошибки

**Ошибка 1: Универсальные настройки для всех документов**

Техническая документация и художественный текст требуют разных подходов. Сегментируйте по типам документов.

**Ошибка 2: Игнорирование метаданных**

"Просто текст" — потеря информации. Сохраняйте заголовки, даты, авторов, разделы.

**Ошибка 3: Отсутствие валидации качества**

"Загрузили и забыли" — рецепт проблем. Визуально проверяйте случайные чанки.

**Ошибка 4: Слишком агрессивная очистка**

Удаление "шума" может удалить полезную информацию. Важный термин или код может выглядеть как мусор.

### Метрики качества подготовки данных

Как понять, хорошо ли подготовлены данные?

**Внутренние метрики:**
- Средний/медианный размер чанка (должен быть около целевого)
- Стандартное отклонение размера (низкое = предсказуемо)
- Процент "обрезанных" предложений

**Внешние метрики (через тестирование retrieval):**
- Recall@k: находятся ли релевантные чанки?
- Качество ответов: корректны ли ответы LLM на тестовые вопросы?

---

## Заключение: Данные — фундамент успеха

Подготовка данных — не glamorous работа. Она не так эффектна, как тонкая настройка нейросетей или красивые UI. Но именно она определяет потолок качества вашей RAG-системы.

Ключевые идеи:

**Загрузка — первая линия обороны.** Неправильно извлечённый текст невозможно исправить chunking'ом. Инвестируйте в качественные загрузчики для ваших форматов.

**Chunking — искусство, не наука.** Нет универсально правильного размера. Экспериментируйте, измеряйте, адаптируйте под свои данные.

**Метаданные — недооценённый актив.** Каждый бит структурированной информации улучшает возможности фильтрации и ранжирования.

**Pipeline должен быть воспроизводимым.** От первого прогона до production — одинаковый процесс. Это позволяет отлаживать и улучшать.

---

## Вопросы для самопроверки

1. Почему извлечение текста из PDF — нетривиальная задача? Какие проблемы могут возникнуть?

2. Объясните, почему overlap между чанками важен для качества RAG.

3. Сравните RecursiveCharacterTextSplitter и семантический chunking. Когда использовать каждый?

4. Как работает Parent Document Retriever? Какую проблему он решает?

5. Зачем нужна дедупликация чанков?

6. Спроектируйте pipeline обработки для корпоративной Wiki с документами в Markdown и PDF.

---

## Ключевые термины

| Термин | Определение |
|--------|-------------|
| **ETL** | Extract-Transform-Load — процесс извлечения, преобразования и загрузки данных |
| **Document Loader** | Компонент для извлечения текста из файлов различных форматов |
| **Chunking** | Разбиение документов на фрагменты для индексации |
| **Overlap** | Перекрытие между соседними чанками для сохранения контекста |
| **Semantic Chunking** | Разбиение по смысловым границам, а не по размеру |
| **Parent Document Retriever** | Паттерн с двухуровневой структурой чанков |
| **Hypothetical Questions** | Техника генерации вопросов для чанков |
| **Дедупликация** | Удаление идентичных или почти идентичных фрагментов |
