from pydantic import BaseModel
from typing import Optional


class NithinGenerateRequest(BaseModel):
    platform: str  # "x" or "linkedin"
    context: str
    facts: list[str] = []
    angle: Optional[str] = None
    cta: Optional[str] = None
    thread: bool = False
    variants: int = 3
    max_chars: Optional[int] = None
    allow_research: bool = True
    research_query: Optional[str] = None
    auto_research: bool = True
    proofread: bool = True


class NithinGenerateResponse(BaseModel):
    text: str
    warnings: list[str]
    metadata: dict
