"""A2A-сервер: юридический рецензент.

Реализует подмножество A2A v1.0 (JSON-RPC 2.0 over HTTP):
- GET  /.well-known/agent.json  — Agent Card
- POST /                        — JSON-RPC: tasks/send, tasks/get, tasks/cancel

Запуск: python a2a_server.py          (порт 5002)
        python a2a_server.py --port N  (произвольный порт)
"""

import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, SystemMessage

from llm_config import get_llm

app = FastAPI(title="Legal Review A2A Agent")
llm = get_llm()

# --- In-memory task storage ---
tasks: dict[str, dict] = {}


# ============================================================================
# AGENT CARD
# ============================================================================
AGENT_CARD = {
    "name": "Legal Review Agent",
    "description": "Reviews documents for legal compliance and risks",
    "url": "http://localhost:5002",
    "version": "1.0",
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "skills": [
        {
            "id": "legal-review",
            "name": "Legal Compliance Review",
            "description": (
                "Analyzes text for legal risks, compliance issues, "
                "and regulatory concerns"
            ),
            "tags": ["legal", "compliance", "review"],
            "examples": [
                "Review this contract for potential risks",
                "Check compliance of this policy document",
            ],
        },
    ],
    "securitySchemes": {
        "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
    },
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain", "application/json"],
}


@app.get("/.well-known/agent.json")
@app.get("/.well-known/agent-card.json")
async def agent_card():
    """Agent Card — «визитная карточка» агента по стандарту A2A."""
    return AGENT_CARD


# ============================================================================
# JSON-RPC 2.0 HANDLER
# ============================================================================
@app.post("/")
async def jsonrpc_handler(request: dict):
    """Единая JSON-RPC точка входа для A2A-операций."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id", 1)

    handlers = {
        "tasks/send": handle_send,
        "tasks/get": handle_get,
        "tasks/cancel": handle_cancel,
        "message/send": handle_message_send,  # A2A v1.0 SDK (a2a-sdk)
    }
    handler = handlers.get(method)
    if not handler:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return await handler(params, req_id)


async def handle_message_send(params: dict, req_id: Any):
    """Обработка message/send — формат A2A v1.0 SDK (a2a-sdk).

    SDK ожидает Task со строгой структурой: contextId, history как список
    Message, status.message как Message (не строка).
    """
    # SDK шлёт: params = {"message": {...}, "configuration": {...}}
    message = params.get("message", params)
    text_parts = [
        p["text"] for p in message.get("parts", []) if p.get("kind") == "text"
    ]
    input_text = " ".join(text_parts)
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())

    # Проверяем достаточность входных данных
    if len(input_text.strip()) < 20:
        task = {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": "input-required",
                "message": {
                    "role": "agent",
                    "messageId": str(uuid.uuid4()),
                    "kind": "message",
                    "parts": [{"kind": "text", "text": (
                        "Пожалуйста, предоставьте более подробное описание "
                        "для рецензии (минимум 20 символов)."
                    )}],
                },
            },
            "artifacts": [],
            "history": [message],
        }
        tasks[task_id] = task
        return {"jsonrpc": "2.0", "id": req_id, "result": task}

    # Выполняем задачу через LLM
    response = llm.invoke([
        SystemMessage(content=(
            "Ты — юридический рецензент. Проанализируй текст на предмет "
            "юридических рисков, вопросов compliance и регуляторных проблем. "
            "Ответь кратко и структурированно на русском языке."
        )),
        HumanMessage(content=input_text),
    ])

    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "completed"},
        "artifacts": [{
            "artifactId": str(uuid.uuid4()),
            "parts": [{"kind": "text", "text": response.content}],
        }],
        "history": [message],
    }
    tasks[task_id] = task
    return {"jsonrpc": "2.0", "id": req_id, "result": task}


async def handle_send(params: dict, req_id: Any):
    """Обработка tasks/send — создание и выполнение задачи."""
    task_id = params.get("id", str(uuid.uuid4()))
    message = params.get("message", {})
    text_parts = [
        p["text"] for p in message.get("parts", []) if p.get("kind") == "text"
    ]
    input_text = " ".join(text_parts)

    # Создаём задачу: submitted
    task = {
        "id": task_id,
        "status": {"state": "submitted"},
        "artifacts": [],
        "history": [],
    }
    tasks[task_id] = task

    # submitted → working
    _transition(task, "working")

    # Проверяем достаточность входных данных
    if len(input_text.strip()) < 20:
        _transition(task, "input-required")
        task["status"]["message"] = (
            "Пожалуйста, предоставьте более подробное описание для рецензии "
            "(минимум 20 символов)."
        )
        return {"jsonrpc": "2.0", "id": req_id, "result": task}

    # Выполняем задачу через LLM
    response = llm.invoke([
        SystemMessage(content=(
            "Ты — юридический рецензент. Проанализируй текст на предмет "
            "юридических рисков, вопросов compliance и регуляторных проблем. "
            "Ответь кратко и структурированно на русском языке."
        )),
        HumanMessage(content=input_text),
    ])

    task["artifacts"].append({
        "parts": [{"kind": "text", "text": response.content}],
        "index": 0,
    })

    # working → completed
    _transition(task, "completed")
    return {"jsonrpc": "2.0", "id": req_id, "result": task}


async def handle_get(params: dict, req_id: Any):
    """Обработка tasks/get — получение статуса задачи."""
    task_id = params.get("id", "")
    task = tasks.get(task_id)
    if not task:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Task not found: {task_id}"},
        }
    return {"jsonrpc": "2.0", "id": req_id, "result": task}


async def handle_cancel(params: dict, req_id: Any):
    """Обработка tasks/cancel — отмена задачи."""
    task_id = params.get("id", "")
    task = tasks.get(task_id)
    if not task:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Task not found: {task_id}"},
        }
    _transition(task, "canceled")
    return {"jsonrpc": "2.0", "id": req_id, "result": task}


def _transition(task: dict, new_state: str):
    """Переход задачи в новое состояние с записью в историю."""
    old_state = task["status"]["state"]
    task["status"]["state"] = new_state
    task["history"].append({"from": old_state, "to": new_state})


# ============================================================================
# ЗАПУСК
# ============================================================================
if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
