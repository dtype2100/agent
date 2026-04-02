from typing import Optional
from pydantic import BaseModel, Field


class IndexTextsRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="색인할 텍스트 목록")


class IndexResponse(BaseModel):
    indexed: int = Field(..., description="색인된 청크 수")
    filename: Optional[str] = None
