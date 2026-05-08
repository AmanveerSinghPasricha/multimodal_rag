from typing import List, Optional
from pydantic import BaseModel, Field

class RAGResponse(BaseModel):
    answer: str = Field(description="The primary response text")
    citations: List[str] = Field(description="List of 'Filename (Page X)' sources used")
    confidence: float = Field(description="Recalibrated 0.0-1.0 score")
    memory_update: Optional[str] = Field(default=None, description="New fact to remember")