"""MCP-сервер: аналитическая SQLite база данных.

Запуск: python mcp_db_server.py          (порт 8002)
        python mcp_db_server.py --port N  (произвольный порт)

Инструменты: query_db, list_tables, table_schema
"""

import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "analytics.db"

DEFAULT_PORT = 8002
mcp = FastMCP("Database Server", host="127.0.0.1", port=DEFAULT_PORT)


@mcp.tool()
def query_db(sql: str) -> str:
    """Выполнить SQL-запрос к аналитической базе данных (только SELECT).

    Args:
        sql: SQL-запрос (только SELECT)
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Ошибка: разрешены только SELECT-запросы."
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        if not rows:
            return "Запрос выполнен, результатов нет."
        columns = rows[0].keys()
        lines = [" | ".join(columns)]
        lines.append("-" * len(lines[0]))
        for row in rows:
            lines.append(" | ".join(str(row[col]) for col in columns))
        return "\n".join(lines)
    except sqlite3.Error as e:
        return f"SQL ошибка: {e}"
    finally:
        conn.close()


@mcp.tool()
def list_tables() -> str:
    """Показать все таблицы в базе данных."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cur.fetchall()]
        return "Таблицы: " + ", ".join(tables) if tables else "База данных пуста."
    finally:
        conn.close()


@mcp.tool()
def table_schema(table_name: str) -> str:
    """Получить схему таблицы (колонки и типы).

    Args:
        table_name: Имя таблицы
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(f"PRAGMA table_info({table_name})")
        cols = cur.fetchall()
        if not cols:
            return f"Таблица '{table_name}' не найдена."
        lines = [f"Схема таблицы '{table_name}':"]
        for col in cols:
            pk = " PRIMARY KEY" if col[5] else ""
            lines.append(f"  {col[1]} {col[2]}{pk}")
        return "\n".join(lines)
    finally:
        conn.close()


if __name__ == "__main__":
    if not DB_PATH.exists():
        from seed_db import seed
        seed()

    mcp.run(transport="sse")
