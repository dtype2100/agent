from fastapi import APIRouter, Depends

from app.api.endpoints import agent, documents, rag
from app.core.security import verify_api_key_dependency

api_router = APIRouter(dependencies=[Depends(verify_api_key_dependency)])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
