"""MCP-сервер: поиск через DuckDuckGo.

Запуск: python mcp_search_server.py          (порт 8001)
        python mcp_search_server.py --port N  (произвольный порт)

Инструменты: search_web, search_news
"""

from mcp.server.fastmcp import FastMCP
from ddgs import DDGS

DEFAULT_PORT = 8001
mcp = FastMCP("Search Server", host="127.0.0.1", port=DEFAULT_PORT)


@mcp.tool()
def search_web(query: str) -> str:
    """Поиск информации в интернете через DuckDuckGo.

    Args:
        query: Поисковый запрос
    """
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
    if not results:
        return f"По запросу '{query}' ничего не найдено."
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['body'][:200]}")
    return "\n".join(lines)


@mcp.tool()
def search_news(query: str) -> str:
    """Поиск свежих новостей через DuckDuckGo.

    Args:
        query: Поисковый запрос для новостей
    """
    with DDGS() as ddgs:
        results = ddgs.news(query, max_results=3)
    if not results:
        return f"Новостей по запросу '{query}' не найдено."
    lines = []
    for r in results:
        date = r.get("date", "N/A")
        lines.append(f"- [{date}] {r['title']}: {r['body'][:150]}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="sse")
