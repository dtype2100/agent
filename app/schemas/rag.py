from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="검색 쿼리")


class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list, description="검색된 청크 텍스트")
