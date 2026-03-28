"""Создание и заполнение тестовой SQLite базы данных."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "analytics.db"


def seed():
    """Создать таблицы и заполнить тестовыми данными."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS framework_metrics")
    cur.execute("DROP TABLE IF EXISTS protocol_adoption")

    cur.execute("""
        CREATE TABLE framework_metrics (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            github_stars INTEGER,
            monthly_downloads INTEGER,
            release_year INTEGER,
            language TEXT,
            category TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE protocol_adoption (
            id INTEGER PRIMARY KEY,
            protocol TEXT NOT NULL,
            version TEXT,
            organizations INTEGER,
            monthly_sdk_downloads INTEGER,
            status TEXT
        )
    """)

    frameworks = [
        ("LangGraph", 18000, 5200000, 2024, "Python", "orchestration"),
        ("CrewAI", 45000, 3800000, 2023, "Python", "orchestration"),
        ("OpenAI Agents SDK", 15000, 2100000, 2025, "Python/TypeScript", "orchestration"),
        ("Google ADK", 8000, 900000, 2025, "Python/TS/Go/Java", "orchestration"),
        ("Pydantic AI", 12000, 1500000, 2025, "Python", "orchestration"),
        ("Mastra", 22000, 1300000, 2025, "TypeScript", "orchestration"),
        ("smolagents", 6000, 400000, 2025, "Python", "code-first"),
        ("Letta", 14000, 300000, 2024, "Python", "memory-centric"),
    ]
    cur.executemany(
        "INSERT INTO framework_metrics VALUES (NULL, ?, ?, ?, ?, ?, ?)",
        frameworks,
    )

    protocols = [
        ("MCP", "2025-11-25", 146, 97000000, "stable"),
        ("A2A", "1.0", 150, 8500000, "stable"),
        ("WebMCP", "draft", 4, 50000, "experimental"),
        ("AG-UI", "0.1", 12, 200000, "early"),
        ("ANP", "draft", 3, 30000, "experimental"),
    ]
    cur.executemany(
        "INSERT INTO protocol_adoption VALUES (NULL, ?, ?, ?, ?, ?)",
        protocols,
    )

    conn.commit()
    conn.close()
    print(f"Database seeded: {DB_PATH}")


if __name__ == "__main__":
    seed()
