from typing import Optional
from pydantic import BaseModel


class FileClassificationResult(BaseModel):
    filename: str
    category: str
    confidence: Optional[float] = None


class BatchClassificationResponse(BaseModel):
    results: list[FileClassificationResult]


# --- Data Extraction Models ---

class LineItem(BaseModel):
    description: Optional[str] = None
    amount: Optional[str] = None
    quantity: Optional[str] = None


class ExtractedData(BaseModel):
    member_id: Optional[str] = None
    policy_number: Optional[str] = None
    claim_date: Optional[str] = None
    treatment_date: Optional[str] = None
    claimed_amount: Optional[str] = None
    line_items: Optional[list[LineItem]] = None
    signature: Optional[str] = None
    location: Optional[str] = None
    bank_amount: Optional[str] = None


class FileExtractionResult(BaseModel):
    filename: str
    data: ExtractedData


class BatchExtractionResponse(BaseModel):
    results: list[FileExtractionResult]
