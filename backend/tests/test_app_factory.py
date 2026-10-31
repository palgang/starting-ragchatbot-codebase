"""
Builds a FastAPI app that mirrors the routes in backend/app.py, but with an
injected RAG system and a caller-supplied frontend directory.

Why this exists: backend/app.py builds a real RAGSystem at import time (Chroma,
sentence-transformers, Anthropic client) and mounts "../frontend" relative to the
working directory, which does not exist when pytest runs from the repo root.
This factory lets API tests run without any of those dependencies.

If you change a route, request/response model, or middleware in backend/app.py,
update the matching code here.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system, frontend_dir: Path) -> FastAPI:
    """Create an app with the same API surface as backend/app.py."""
    app = FastAPI(title="Course Materials RAG System (test)")

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()

            answer, sources = rag_system.query(request.query, session_id)

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

    return app
