"""MCP-сервер с мультиагентной системой внутри (матрёшка).

Снаружи — один инструмент: deep_analysis(query) → report.
Внутри — граф LangGraph из двух агентов: исследователь → аналитик.

Запуск: python mcp_matryoshka_server.py          (порт 8003)
        python mcp_matryoshka_server.py --port N  (произвольный порт)
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from mcp.server.fastmcp import FastMCP

from llm_config import get_llm

DEFAULT_PORT = 8003
mcp = FastMCP("Deep Analysis Server", host="127.0.0.1", port=DEFAULT_PORT)


# --- Внутренний граф: исследователь → аналитик ---


class InnerState(TypedDict):
    query: str
    research: str
    analysis: str


def _build_inner_graph():
    llm = get_llm()

    def researcher(state: InnerState) -> dict:
        resp = llm.invoke([
            SystemMessage(
                content="Найди 3-5 ключевых фактов по теме. Будь краток и конкретен."
            ),
            HumanMessage(content=state["query"]),
        ])
        return {"research": resp.content}

    def analyst(state: InnerState) -> dict:
        resp = llm.invoke([
            SystemMessage(
                content="Проанализируй факты и сформулируй краткий аналитический вывод."
            ),
            HumanMessage(content=f"Факты:\n{state['research']}"),
        ])
        return {"analysis": resp.content}

    g = StateGraph(InnerState)
    g.add_node("researcher", researcher)
    g.add_node("analyst", analyst)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", END)
    return g.compile()


inner_graph = _build_inner_graph()


@mcp.tool()
def deep_analysis(query: str) -> str:
    """Глубокий анализ темы. Внутри — мультиагентная система (исследователь + аналитик).

    Args:
        query: Тема для анализа
    """
    result = inner_graph.invoke({
        "query": query,
        "research": "",
        "analysis": "",
    })
    return result["analysis"]


if __name__ == "__main__":
    mcp.run(transport="sse")
