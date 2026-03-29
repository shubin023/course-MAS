from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RequestType(str, Enum):
    QUESTION = "question"
    TASK = "task"
    SMALL_TALK = "small_talk"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class MemoryStrategy(str, Enum):
    BUFFER = "buffer"
    SUMMARY = "summary"


class Classification(BaseModel):
    request_type: RequestType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AssistantResponse(BaseModel):
    content: str
    request_type: RequestType
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_used: int = Field(ge=0)
