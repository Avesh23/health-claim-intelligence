from typing import Optional
from pydantic import BaseModel


class FileClassificationResult(BaseModel):
    filename: str
    category: str
    confidence: Optional[float] = None


class BatchClassificationResponse(BaseModel):
    results: list[FileClassificationResult]
